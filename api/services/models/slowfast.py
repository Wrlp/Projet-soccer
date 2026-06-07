"""Inférence SlowFast — fenêtres glissantes ~4 s @ 16 fps."""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from api.config import SLOWFAST_ALPHA, SLOWFAST_CKPT, SLOWFAST_IMAGE_SIZE, SLOWFAST_SPLITS, SLOWFAST_T_S
from api.services.models.common import (
    DEFAULT_STRIDE_SEC,
    INFER_BATCH_SIZE,
    apply_threshold_with_fallback,
    build_inference_meta,
    center_timestamp,
    load_video_frames_rgb,
    make_candidate,
    merge_events,
    resolve_half_range,
    sliding_windows,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INFER_FPS = 16
WINDOW_FRAMES = SLOWFAST_T_S * SLOWFAST_ALPHA
WINDOW_SEC = WINDOW_FRAMES / INFER_FPS
DEFAULT_THRESHOLD = 0.18
MIN_CONFIDENCE = 0.10
MERGE_GAP_SEC = 2.5


def is_model_ready() -> bool:
    return SLOWFAST_CKPT.exists()


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


def _temporal_sample_torch(frames: torch.Tensor, t_needed: int) -> torch.Tensor:
    t_total = frames.shape[0]
    if t_total >= t_needed:
        idx = np.linspace(0, t_total - 1, num=t_needed, dtype=int)
        return frames[idx]
    pad = [t_total - 1] * (t_needed - t_total)
    idx = list(range(t_total)) + pad
    return frames[idx]


def _to_slow_fast(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tfm = T.Compose([T.ConvertImageDtype(torch.float), T.Resize((SLOWFAST_IMAGE_SIZE, SLOWFAST_IMAGE_SIZE))])
    fast = _temporal_sample_torch(frames, WINDOW_FRAMES)
    slow_idx = fast[::SLOWFAST_ALPHA][: SLOWFAST_T_S]

    def stack_path(sel: torch.Tensor) -> torch.Tensor:
        proc = [tfm(f) for f in sel]
        return torch.stack(proc, dim=1)

    return stack_path(slow_idx), stack_path(fast)


def _score_windows_batched(
    model,
    device,
    idx_to_name: dict[int, str],
    frames: torch.Tensor,
    windows: list[tuple[int, int]],
    *,
    label_half: int,
    batch_size: int = INFER_BATCH_SIZE,
) -> list[dict]:
    candidates: list[dict] = []
    with torch.no_grad():
        for batch_start in range(0, len(windows), batch_size):
            batch = windows[batch_start : batch_start + batch_size]
            slow_batch: list[torch.Tensor] = []
            fast_batch: list[torch.Tensor] = []
            spans: list[tuple[int, int]] = []
            for start, end in batch:
                clip = frames[start:end]
                if clip.shape[0] < 2:
                    continue
                slow, fast = _to_slow_fast(clip)
                slow_batch.append(slow)
                fast_batch.append(fast)
                spans.append((start, end))
            if not slow_batch:
                continue

            slow = torch.stack(slow_batch, dim=0).to(device)
            fast = torch.stack(fast_batch, dim=0).to(device)
            logits = model(slow, fast)
            probs = torch.softmax(logits, dim=1)
            for i, (start, end) in enumerate(spans):
                conf = float(probs[i].max().item())
                pred_idx = int(probs[i].argmax().item())
                if conf < MIN_CONFIDENCE:
                    continue
                class_key = idx_to_name.get(pred_idx, str(pred_idx))
                candidates.append(
                    make_candidate(
                        half=label_half,
                        class_key=class_key,
                        confidence=conf,
                        timestamp=center_timestamp(start, end, INFER_FPS),
                    )
                )
    return candidates


def predict_video(
    video_path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    half: str | int = "auto",
    skip_ball_out: bool = True,
) -> tuple[list[dict], dict]:
    model, device, idx_to_name = _get_model()
    start_sec, duration_sec, label_half = resolve_half_range(video_path, half)
    rgb = load_video_frames_rgb(
        video_path,
        fps=INFER_FPS,
        size=SLOWFAST_IMAGE_SIZE,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    frames = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).contiguous()
    stride = max(1, int(stride_sec * INFER_FPS))
    windows = sliding_windows(frames.shape[0], WINDOW_FRAMES, stride)

    candidates = _score_windows_batched(
        model,
        device,
        idx_to_name,
        frames,
        windows,
        label_half=label_half,
    )

    events, used_fallback = apply_threshold_with_fallback(
        candidates,
        threshold,
        skip_ball_out=skip_ball_out,
        min_confidence=MIN_CONFIDENCE,
    )
    merged = merge_events(events, gap_sec=MERGE_GAP_SEC)
    meta = build_inference_meta(
        model_name="SlowFast",
        checkpoint=str(SLOWFAST_CKPT),
        infer_fps=INFER_FPS,
        window_sec=WINDOW_SEC,
        stride_sec=stride_sec,
        n_frames=frames.shape[0],
        n_windows=len(windows),
        threshold=threshold,
        candidates=candidates,
        used_fallback=used_fallback,
        merged_count=len(merged),
        analysis_start_sec=start_sec,
        label_half=label_half,
    )
    return merged, meta
