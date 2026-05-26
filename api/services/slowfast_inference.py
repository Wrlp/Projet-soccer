"""
Inférence SlowFast (SlowFast/checkpoints/best.pth) sur extraits vidéo.
Fenêtres glissantes ~4 s @ 16 fps, comme à l'entraînement.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import SLOWFAST_ALPHA, SLOWFAST_CKPT, SLOWFAST_IMAGE_SIZE, SLOWFAST_SPLITS, SLOWFAST_T_S

# Clip training ≈ 5 s ; fenêtre fast = T_s * alpha frames @ 16 fps
INFER_FPS = 16
WINDOW_SEC = (SLOWFAST_T_S * SLOWFAST_ALPHA) / INFER_FPS
DEFAULT_STRIDE_SEC = 1.0
DEFAULT_THRESHOLD = 0.18
MIN_CONFIDENCE = 0.10
TOP_K_FALLBACK = 8

_DISPLAY_NAMES = {
    "Ball_out_of_play": "Ball out of play",
    "Yellow_red_card": "Yellow->red card",
    "Shots_off_target": "Shots off target",
    "Shots_on_target": "Shots on target",
    "Direct_free-kick": "Direct free-kick",
    "Indirect_free-kick": "Indirect free-kick",
}


def _display_label(class_name: str) -> str:
    return _DISPLAY_NAMES.get(class_name, class_name.replace("_", " "))


def _load_class_map() -> tuple[int, dict[int, str]]:
    with open(SLOWFAST_SPLITS, encoding="utf-8") as f:
        splits = json.load(f)
    idx_to_name: dict[int, str] = {}
    for split in splits.values():
        for e in split:
            idx_to_name[int(e["label"])] = e["class_name"]
    num_classes = max(idx_to_name.keys()) + 1
    return num_classes, idx_to_name


@lru_cache(maxsize=1)
def _get_model():
    if not SLOWFAST_CKPT.exists():
        raise FileNotFoundError(
            f"Checkpoint SlowFast introuvable : {SLOWFAST_CKPT}\n"
            "Entraînez avec : python SlowFast/train.py"
        )
    from SlowFast.model import SlowFastSimple

    num_classes, idx_to_name = _load_class_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlowFastSimple(num_classes=num_classes, pretrained_backbones=False)
    ckpt = torch.load(SLOWFAST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info("SlowFast chargé (%s, %d classes, %s)", SLOWFAST_CKPT.name, num_classes, device)
    return model, device, idx_to_name


def _load_video_frames(video_path: Path, size: int = SLOWFAST_IMAGE_SIZE) -> torch.Tensor:
    """Tensor (T, C, H, W) @ 16 fps, comme SoccerClipDataset."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={INFER_FPS},scale={size}:{size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg : {proc.stderr.decode(errors='replace')[:500]}")
    raw = proc.stdout
    if not raw:
        raise ValueError(f"Vidéo vide ou illisible : {video_path.name}")

    frame_size = size * size * 3
    n_frames = len(raw) // frame_size
    arr = np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8).reshape(n_frames, size, size, 3)
    return torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).contiguous()


def _temporal_sample(frames: torch.Tensor, t_needed: int) -> torch.Tensor:
    t_total = frames.shape[0]
    if t_total >= t_needed:
        idx = np.linspace(0, t_total - 1, num=t_needed, dtype=int)
        return frames[idx]
    pad = [t_total - 1] * (t_needed - t_total)
    idx = list(range(t_total)) + pad
    return frames[idx]


def _to_slow_fast(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """frames (T_f, C, H, W) → slow, fast (C, T, H, W)."""
    tfm = T.Compose([T.ConvertImageDtype(torch.float), T.Resize((SLOWFAST_IMAGE_SIZE, SLOWFAST_IMAGE_SIZE))])
    t_f = SLOWFAST_T_S * SLOWFAST_ALPHA
    fast = _temporal_sample(frames, t_f)
    slow_idx = fast[::SLOWFAST_ALPHA][: SLOWFAST_T_S]

    def stack_path(sel: torch.Tensor) -> torch.Tensor:
        proc = [tfm(f) for f in sel]
        return torch.stack(proc, dim=1)

    return stack_path(slow_idx), stack_path(fast)


def _merge_events(events: list[dict], gap_sec: float = 2.5) -> list[dict]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e["timestamp"])
    out = [events[0]]
    for ev in events:
        last = out[-1]
        if ev["label"] == last["label"] and ev["timestamp"] - last["timestamp"] <= gap_sec:
            if ev["confidence"] > last["confidence"]:
                out[-1] = ev
        else:
            out.append(ev)
    return out


def predict_video(
    video_path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    half: int = 1,
    skip_ball_out: bool = True,
) -> tuple[list[dict], dict]:
    """
    Prédictions par fenêtres glissantes.
    Chaque fenêtre ≈ 4 s ; timestamp = centre de la fenêtre (relatif au clip).
    """
    model, device, idx_to_name = _get_model()
    frames = _load_video_frames(video_path)
    t_f = SLOWFAST_T_S * SLOWFAST_ALPHA
    stride = max(1, int(stride_sec * INFER_FPS))
    duration_sec = frames.shape[0] / INFER_FPS

    if frames.shape[0] < t_f:
        windows = [(0, frames.shape[0])]
    else:
        windows = []
        for start in range(0, frames.shape[0] - t_f + 1, stride):
            windows.append((start, start + t_f))
        if not windows:
            windows = [(0, min(t_f, frames.shape[0]))]

    candidates: list[dict] = []
    with torch.no_grad():
        for start, end in windows:
            clip = frames[start:end]
            if clip.shape[0] < 2:
                continue
            slow, fast = _to_slow_fast(clip)
            slow = slow.unsqueeze(0).to(device)
            fast = fast.unsqueeze(0).to(device)
            logits = model(slow, fast)
            prob = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = float(prob.max().item()), int(prob.argmax().item())
            if conf < MIN_CONFIDENCE:
                continue
            class_key = idx_to_name.get(pred_idx, str(pred_idx))
            center_frame = start + (end - start) / 2
            ts = int(center_frame / INFER_FPS)
            candidates.append(
                {
                    "timestamp": ts,
                    "half": half,
                    "label": _display_label(class_key),
                    "confidence": round(conf, 4),
                    "classKey": class_key,
                }
            )

    def _filter(cands: list[dict], th: float, skip_ball: bool) -> list[dict]:
        out = []
        for c in cands:
            if c["confidence"] < th:
                continue
            if skip_ball and c.get("classKey") == "Ball_out_of_play":
                continue
            out.append(c)
        return out

    events = _filter(candidates, threshold, skip_ball_out)
    used_fallback = False
    if not events:
        # SlowFast est peu calibré en probabilité sur vidéos hors clips d'entraînement
        pool = _filter(candidates, MIN_CONFIDENCE, skip_ball_out)
        if not pool and candidates:
            pool = [c for c in candidates if c.get("classKey") != "Ball_out_of_play"] or candidates
        pool.sort(key=lambda c: c["confidence"], reverse=True)
        events = pool[:TOP_K_FALLBACK]
        used_fallback = bool(events)

    merged = _merge_events(events)
    meta = {
        "model": "SlowFast",
        "checkpoint": str(SLOWFAST_CKPT),
        "inferFps": INFER_FPS,
        "windowSeconds": round(WINDOW_SEC, 2),
        "strideSeconds": stride_sec,
        "framesDecoded": int(frames.shape[0]),
        "durationSeconds": round(duration_sec, 1),
        "durationMinutes": round(duration_sec / 60, 2),
        "windowsScored": len(windows),
        "thresholdRequested": threshold,
        "usedFallback": used_fallback,
        "maxConfidence": round(max((c["confidence"] for c in candidates), default=0), 4),
    }
    if used_fallback:
        meta["detectionHint"] = (
            f"Aucun score ≥ {threshold:.0%} — affichage des {len(merged)} meilleures fenêtres "
            f"(max {meta['maxConfidence']:.0%}). Baissez le seuil pour filtrer davantage."
        )
    return merged, meta
