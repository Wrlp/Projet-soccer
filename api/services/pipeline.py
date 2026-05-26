from __future__ import annotations

import json
import logging
from pathlib import Path

from api.services import jobs
from api.services.metrics import build_full_result
from api.services.slowfast_inference import predict_video
from api.services.video_segments import EXCERPT_MAX_SEC as _EXCERPT_SEC
from api.config import JOBS_DIR

logger = logging.getLogger(__name__)


def _segment_meta(duration_sec: float, half: int, model_meta: dict) -> dict:
    is_excerpt = duration_sec < _EXCERPT_SEC
    return {
        "framesAnalyzed": model_meta.get("framesDecoded", 0),
        "fps": model_meta.get("inferFps", 16),
        "durationSeconds": round(duration_sec, 1),
        "durationMinutes": round(duration_sec / 60, 2),
        "isExcerpt": is_excerpt,
        "analysisMode": "excerpt" if is_excerpt else ("match" if duration_sec >= 85 * 60 else "half"),
        "segmentLabel": "Extrait vidéo" if is_excerpt else "Séquence vidéo",
        "model": "SlowFast",
        **model_meta,
    }


def run_analysis(job_id: str, video_path: Path) -> None:
    job = jobs.get_job(job_id)
    if not job:
        raise KeyError(job_id)
    params = job.get("params", {})
    threshold = float(params.get("threshold", 0.18))
    half_param = params.get("half")
    half = 1 if half_param in ("auto", None, "", "1", 1) else int(half_param)

    try:
        jobs.update_job(job_id, status="processing", progress=15)
        predictions, model_meta = predict_video(
            video_path,
            threshold=threshold,
            half=half,
        )
        jobs.update_job(job_id, progress=85)
        dur = float(model_meta.get("durationSeconds", 0))
        seg_meta = {
            **_segment_meta(dur, half, model_meta),
            "thresholdRequested": threshold,
        }
        result = build_full_result(job, predictions, seg_meta, 0)
        (JOBS_DIR / f"{job_id}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        jobs.update_job(job_id, status="completed", progress=100)
        logger.info("Job %s OK — %d événements (SlowFast)", job_id, len(predictions))
    except Exception as e:
        logger.exception("Job %s", job_id)
        jobs.update_job(job_id, status="failed", error=str(e), progress=0)
        raise
