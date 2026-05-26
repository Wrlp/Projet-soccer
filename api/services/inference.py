from __future__ import annotations

import logging
import pickle
from functools import lru_cache

import numpy as np

from api.config import DEFAULT_CONTEXT_FRAMES, FPS, MODEL_NAME, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def load_artifacts():
    paths = [MODELS_DIR / f"{MODEL_NAME}.pkl", MODELS_DIR / "scaler.pkl", MODELS_DIR / "idx_to_label.pkl"]
    if not all(p.exists() for p in paths):
        raise ModelNotFoundError("Modèle absent — lancez : uv run main.py")
    model = pickle.load(open(paths[0], "rb"))
    scaler = pickle.load(open(paths[1], "rb"))
    idx_to_label = pickle.load(open(paths[2], "rb"))
    n_feat = int(getattr(scaler, "n_features_in_", 512))
    ctx = max(0, (n_feat // 512 - 1) // 2) if n_feat % 512 == 0 else 0
    return model, scaler, idx_to_label, ctx, n_feat


def resolve_context_frames(requested: int) -> int:
    _, _, _, model_ctx, _ = load_artifacts()
    if requested != model_ctx:
        logger.warning("context_frames %s → %s (selon le modèle)", requested, model_ctx)
    return model_ctx


def _vectors(features: np.ndarray, ctx: int) -> np.ndarray:
    n, target = features.shape[0], ctx * 2 + 1
    rows = []
    for i in range(n):
        s, e = max(0, i - ctx), min(n, i + ctx + 1)
        block = features[s:e]
        if len(block) < target:
            block = np.vstack([block, np.zeros((target - len(block), 512))])
        rows.append(block.flatten())
    return np.array(rows, dtype=np.float32)


def run_inference(features: np.ndarray, half: int, *, threshold: float, context_frames: int) -> list[dict]:
    model, scaler, idx_to_label, ctx, _ = load_artifacts()
    ctx = resolve_context_frames(context_frames)
    X = scaler.transform(_vectors(features, ctx))
    probas = model.predict_proba(X)
    events = []
    for i, p in enumerate(probas):
        j = int(np.argmax(p))
        c = float(p[j])
        if c < threshold:
            continue
        events.append(
            {
                "timestamp": int(i / FPS),
                "half": half,
                "label": idx_to_label[int(model.classes_[j])],
                "confidence": round(c, 4),
            }
        )
    events.sort(key=lambda e: e["timestamp"])
    out = []
    for ev in events:
        if out and out[-1]["label"] == ev["label"] and ev["timestamp"] - out[-1]["timestamp"] <= 4:
            if ev["confidence"] > out[-1]["confidence"]:
                out[-1] = ev
        else:
            out.append(ev)
    return out


def predict_on_video_features(halves, *, threshold: float, context_frames: int) -> list[dict]:
    all_ev = []
    for half, feats in halves:
        all_ev.extend(run_inference(feats, half, threshold=threshold, context_frames=context_frames))
    return sorted(all_ev, key=lambda e: (e["half"], e["timestamp"]))
