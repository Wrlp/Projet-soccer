"""Inférence Random Forest (ResNet152 + PCA512) — pipeline legacy."""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from api.config import FPS, MODEL_NAME, MODELS_DIR
from api.services.feature_extractor import extract_pca512_from_video
from api.services.models.common import (
    DEFAULT_STRIDE_SEC,
    build_inference_meta,
    merge_events,
    resolve_half_range,
)
from api.services.models.registry import ModelSpec

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
MERGE_GAP_SEC = 4.0
INFER_FPS = int(FPS)
WINDOW_SEC = 0.5


@lru_cache(maxsize=1)
def _load_artifacts():
    paths = [
        MODELS_DIR / f"{MODEL_NAME}.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "idx_to_label.pkl",
    ]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError("Modèle Random Forest absent — lancez : uv run main.py")
    model = pickle.load(open(paths[0], "rb"))
    scaler = pickle.load(open(paths[1], "rb"))
    idx_to_label = pickle.load(open(paths[2], "rb"))
    n_feat = int(getattr(scaler, "n_features_in_", 512))
    ctx = max(0, (n_feat // 512 - 1) // 2) if n_feat % 512 == 0 else 0
    return model, scaler, idx_to_label, ctx


def is_model_ready() -> bool:
    paths = [
        MODELS_DIR / f"{MODEL_NAME}.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "idx_to_label.pkl",
    ]
    return all(p.exists() for p in paths)


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


def predict_video(
    video_path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    half: str | int = "auto",
    skip_ball_out: bool = True,
) -> tuple[list[dict], dict]:
    del skip_ball_out, stride_sec
    model, scaler, idx_to_label, ctx = _load_artifacts()
    start_sec, duration_sec, label_half = resolve_half_range(video_path, half)
    features = extract_pca512_from_video(
        video_path,
        fps=FPS,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
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
                "timestamp": int(i / FPS) + int(start_sec),
                "half": label_half,
                "label": idx_to_label[int(model.classes_[j])],
                "confidence": round(c, 4),
            }
        )
    events.sort(key=lambda e: e["timestamp"])
    merged = merge_events(events, gap_sec=MERGE_GAP_SEC)

    meta = build_inference_meta(
        model_name="Random Forest",
        checkpoint=str(MODELS_DIR / MODEL_NAME),
        infer_fps=INFER_FPS,
        window_sec=WINDOW_SEC,
        stride_sec=1 / FPS,
        n_frames=features.shape[0],
        n_windows=features.shape[0],
        threshold=threshold,
        candidates=events,
        used_fallback=False,
        merged_count=len(merged),
        analysis_start_sec=start_sec,
        label_half=label_half,
    )
    return merged, meta


def make_sklearn_spec() -> ModelSpec:
    return ModelSpec(
        id="random_forest",
        name="Random Forest",
        description="ResNet152 + PCA512 + Random Forest — 2 fps",
        path=MODELS_DIR / f"{MODEL_NAME}.pkl",
        default_threshold=DEFAULT_THRESHOLD,
        window_seconds=WINDOW_SEC,
        infer_fps=INFER_FPS,
        merge_gap_sec=MERGE_GAP_SEC,
        is_ready=is_model_ready,
        predict=predict_video,
    )
