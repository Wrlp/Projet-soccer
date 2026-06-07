"""Utilitaires partagés pour l'inférence par fenêtres glissantes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

DISPLAY_NAMES: dict[str, str] = {
    "Ball_out_of_play": "Ball out of play",
    "Yellow_red_card": "Yellow->red card",
    "Shots_off_target": "Shots off target",
    "Shots_on_target": "Shots on target",
    "Direct_free-kick": "Direct free-kick",
    "Indirect_free-kick": "Indirect free-kick",
}

DEFAULT_STRIDE_SEC = 1.0
INFER_BATCH_SIZE = 8
TOP_K_FALLBACK = 8
BALL_OUT_CLASS = "Ball_out_of_play"
# ~42 min : en dessous = extrait (pas une mi-temps complète)
EXCERPT_MAX_SEC = 42 * 60


def display_label(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " "))


def merge_events(events: list[dict], gap_sec: float = 2.0) -> list[dict]:
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


def filter_candidates(
    candidates: list[dict],
    threshold: float,
    *,
    skip_ball_out: bool = True,
) -> list[dict]:
    out = []
    for c in candidates:
        if c["confidence"] < threshold:
            continue
        if skip_ball_out and c.get("classKey") == BALL_OUT_CLASS:
            continue
        out.append(c)
    return out


def apply_threshold_with_fallback(
    candidates: list[dict],
    threshold: float,
    *,
    skip_ball_out: bool = True,
    min_confidence: float = 0.12,
    top_k_fallback: int = TOP_K_FALLBACK,
) -> tuple[list[dict], bool]:
    events = filter_candidates(candidates, threshold, skip_ball_out=skip_ball_out)
    if events:
        return events, False

    pool = filter_candidates(candidates, min_confidence, skip_ball_out=skip_ball_out)
    if not pool and candidates:
        pool = [c for c in candidates if c.get("classKey") != BALL_OUT_CLASS] or candidates
    pool.sort(key=lambda c: c["confidence"], reverse=True)
    events = pool[:top_k_fallback]
    return events, bool(events)


def video_duration_sec(video_path: Path) -> float:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(float(out.stdout.strip()), 0.0)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def resolve_half_range(
    video_path: Path,
    half: str | int | None,
    *,
    excerpt_max_sec: float = EXCERPT_MAX_SEC,
) -> tuple[float, float | None, int]:
    """
    Détermine la portion de vidéo à analyser.

    Retourne (start_sec, duration_sec ou None, half pour libellés).
    Pour un extrait court ou half=auto, analyse la vidéo entière.
    Pour un match long + half=1|2, n'analyse que la mi-temps choisie.
    """
    dur = video_duration_sec(video_path)
    if dur <= 0:
        return 0.0, None, 1

    if half in (None, "", "auto"):
        return 0.0, None, 1

    label_half = 2 if str(half) == "2" else 1
    if dur < excerpt_max_sec:
        return 0.0, None, label_half

    half_dur = dur / 2.0
    if label_half == 1:
        return 0.0, half_dur, 1
    return half_dur, half_dur, 2


def sliding_windows(n_frames: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if n_frames < window_size:
        return [(0, n_frames)]
    windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        windows.append((start, start + window_size))
    return windows or [(0, min(window_size, n_frames))]


def load_video_frames_rgb(
    video_path: Path,
    *,
    fps: int,
    size: int,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> np.ndarray:
    """Frames uint8 (T, H, W, C) @ fps donné, optionnellement sur un segment."""
    cmd = ["ffmpeg", "-v", "error"]
    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(["-i", str(video_path)])
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", str(duration_sec)])
    cmd.extend(
        [
            "-vf",
            f"fps={fps},scale={size}:{size}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg : {proc.stderr.decode(errors='replace')[:500]}")
    raw = proc.stdout
    if not raw:
        raise ValueError(f"Vidéo vide ou illisible : {video_path.name}")

    frame_size = size * size * 3
    n_frames = len(raw) // frame_size
    return np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8).reshape(n_frames, size, size, 3)


def temporal_sample_rgb(frames: np.ndarray, t_needed: int) -> np.ndarray:
    t_total = frames.shape[0]
    if t_total >= t_needed:
        idx = np.linspace(0, t_total - 1, num=t_needed, dtype=int)
        return frames[idx]
    pad = [t_total - 1] * (t_needed - t_total)
    idx = list(range(t_total)) + pad
    return frames[idx]


def center_timestamp(start: int, end: int, fps: int) -> int:
    center_frame = start + (end - start) / 2
    return int(center_frame / fps)


def make_candidate(
    *,
    half: int,
    class_key: str,
    confidence: float,
    timestamp: int,
) -> dict:
    return {
        "timestamp": timestamp,
        "half": half,
        "label": display_label(class_key),
        "confidence": round(confidence, 4),
        "classKey": class_key,
    }


def build_inference_meta(
    *,
    model_name: str,
    checkpoint: str,
    infer_fps: int,
    window_sec: float,
    stride_sec: float,
    n_frames: int,
    n_windows: int,
    threshold: float,
    candidates: list[dict],
    used_fallback: bool,
    merged_count: int,
    analysis_start_sec: float = 0.0,
    label_half: int = 1,
) -> dict:
    duration_sec = n_frames / infer_fps
    max_conf = round(max((c["confidence"] for c in candidates), default=0), 4)
    meta = {
        "model": model_name,
        "checkpoint": checkpoint,
        "inferFps": infer_fps,
        "windowSeconds": round(window_sec, 2),
        "strideSeconds": stride_sec,
        "framesDecoded": int(n_frames),
        "durationSeconds": round(duration_sec, 1),
        "durationMinutes": round(duration_sec / 60, 2),
        "windowsScored": n_windows,
        "thresholdRequested": threshold,
        "usedFallback": used_fallback,
        "maxConfidence": max_conf,
        "analysisStartSec": round(analysis_start_sec, 1),
        "labelHalf": label_half,
    }
    if used_fallback:
        meta["detectionHint"] = (
            f"Aucun score ≥ {threshold:.0%} — affichage des {merged_count} meilleures fenêtres "
            f"(max {max_conf:.0%}). Baissez le seuil pour filtrer davantage."
        )
    return meta
