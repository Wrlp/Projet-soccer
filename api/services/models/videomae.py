"""Inférence VideoMAE — fenêtres glissantes de 16 frames @ 16 fps (~1 s)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

from api.config import VIDEOMAE_MODEL_DIR
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
    temporal_sample_rgb,
)

logger = logging.getLogger(__name__)

INFER_FPS = 16
NUM_FRAMES = 16
IMAGE_SIZE = 224
WINDOW_SEC = NUM_FRAMES / INFER_FPS
DEFAULT_THRESHOLD = 0.35
MIN_CONFIDENCE = 0.12
MERGE_GAP_SEC = 2.0


def is_model_ready() -> bool:
    try:
        _check_weights()
        return VIDEOMAE_MODEL_DIR.exists()
    except FileNotFoundError:
        return False


def _check_weights() -> None:
    weights = VIDEOMAE_MODEL_DIR / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"Poids VideoMAE introuvables : {weights}\n"
            "Placez les fichiers dans outputs/models/videomae_soccernet/best_model/"
        )
    if weights.stat().st_size < 1024:
        head = weights.read_bytes()[:64]
        if b"git-lfs" in head:
            raise FileNotFoundError(
                "Poids VideoMAE non téléchargés (pointeur Git LFS).\n"
                "Lancez : git lfs pull"
            )


@lru_cache(maxsize=1)
def _get_model():
    if not VIDEOMAE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Modèle VideoMAE introuvable : {VIDEOMAE_MODEL_DIR}")
    _check_weights()
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoMAEImageProcessor.from_pretrained(str(VIDEOMAE_MODEL_DIR))
    model = VideoMAEForVideoClassification.from_pretrained(str(VIDEOMAE_MODEL_DIR))
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    logger.info("VideoMAE chargé (%s, %d classes, %s)", VIDEOMAE_MODEL_DIR.name, len(id2label), device)
    return model, processor, device, id2label


def _score_windows_batched(
    model,
    processor,
    device,
    id2label: dict[int, str],
    frames,
    windows: list[tuple[int, int]],
    *,
    label_half: int,
    batch_size: int = INFER_BATCH_SIZE,
) -> list[dict]:
    candidates: list[dict] = []
    with torch.no_grad():
        for batch_start in range(0, len(windows), batch_size):
            batch = windows[batch_start : batch_start + batch_size]
            clips: list[list] = []
            spans: list[tuple[int, int]] = []
            for start, end in batch:
                clip = frames[start:end]
                if clip.shape[0] < 2:
                    continue
                sampled = temporal_sample_rgb(clip, NUM_FRAMES)
                clips.append([sampled[i] for i in range(sampled.shape[0])])
                spans.append((start, end))
            if not clips:
                continue

            inputs = processor(clips, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            for i, (start, end) in enumerate(spans):
                conf = float(probs[i].max().item())
                pred_idx = int(probs[i].argmax().item())
                if conf < MIN_CONFIDENCE:
                    continue
                class_key = id2label.get(pred_idx, str(pred_idx))
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
    model, processor, device, id2label = _get_model()
    start_sec, duration_sec, label_half = resolve_half_range(video_path, half)
    frames = load_video_frames_rgb(
        video_path,
        fps=INFER_FPS,
        size=IMAGE_SIZE,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    stride = max(1, int(stride_sec * INFER_FPS))
    windows = sliding_windows(frames.shape[0], NUM_FRAMES, stride)

    candidates = _score_windows_batched(
        model,
        processor,
        device,
        id2label,
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
        model_name="VideoMAE",
        checkpoint=str(VIDEOMAE_MODEL_DIR),
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
