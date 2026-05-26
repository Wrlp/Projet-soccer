"""
Prépare une version MP4 lisible dans le navigateur (H.264).
Les clips SoccerNet (extract_clips.py) sont souvent en MPEG-4 part 2 (mp4v), non supporté par <video>.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PLAYBACK_DIR_NAME = ".playback"
# Codecs souvent illisibles dans Chrome/Firefox/Safari
_UNSUPPORTED_VIDEO = frozenset({"mpeg4", "msmpeg4v3", "msmpeg4v2", "wmv3", "flv1", "theora"})


def _probe_video_codec(path: Path) -> str | None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().lower() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def needs_browser_transcode(path: Path) -> bool:
    codec = _probe_video_codec(path)
    if codec is None:
        return True
    if codec in _UNSUPPORTED_VIDEO:
        return True
    return codec != "h264"


def ensure_playback_mp4(source: Path, job_id: str, cache_root: Path) -> Path:
    """
    Retourne un MP4 H.264 (yuv420p) pour lecture HTML5.
    Met en cache sous cache_root/.playback/{job_id}.mp4
    """
    cache_dir = cache_root / PLAYBACK_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{job_id}.mp4"

    if cached.exists() and cached.stat().st_mtime >= source.stat().st_mtime:
        return cached

    if not needs_browser_transcode(source):
        return source

    logger.info("Transcodage lecture navigateur : %s → %s", source.name, cached.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-an",
        str(cached),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace")[:400]
        logger.error("ffmpeg transcode failed: %s", err)
        raise RuntimeError(f"Transcodage vidéo impossible : {err}") from e

    return cached
