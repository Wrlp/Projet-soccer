from __future__ import annotations

import json
import logging
from pathlib import Path

from api.services import jobs
from api.services.metrics import build_full_result
from api.services.models import default_threshold, get
from api.services.models.common import DEFAULT_STRIDE_SEC
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
        "model": model_meta.get("model", "VideoMAE"),
        **model_meta,
    }


def run_analysis(job_id: str, video_path: Path) -> None:
    job = jobs.get_job(job_id)
    if not job:
        raise KeyError(job_id)
    params = job.get("params", {})
    model_key = str(params.get("model", "videomae")).lower()
    spec = get(model_key)
    threshold = float(params.get("threshold", default_threshold(model_key)))
    half_param = params.get("half", "auto")
    stride_sec = float(params.get("stride_sec", DEFAULT_STRIDE_SEC))

    try:
        jobs.update_job(job_id, status="processing", progress=15)
        predictions, model_meta = spec.predict(
            video_path,
            threshold=threshold,
            half=half_param,
            stride_sec=stride_sec,
        )
        jobs.update_job(job_id, progress=85)
        dur = float(model_meta.get("durationSeconds", 0))
        label_half = int(model_meta.get("labelHalf", 1))
        seg_meta = {
            **_segment_meta(dur, label_half, model_meta),
            "thresholdRequested": threshold,
            "strideSeconds": stride_sec,
            "analysisStartSec": model_meta.get("analysisStartSec", 0),
        }
        result = build_full_result(job, predictions, seg_meta, 0)
        (JOBS_DIR / f"{job_id}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        jobs.update_job(job_id, status="completed", progress=100)
        logger.info("Job %s OK — %d événements (%s)", job_id, len(predictions), model_meta.get("model", model_key))
    except Exception as e:
        logger.exception("Job %s", job_id)
        jobs.update_job(job_id, status="failed", error=str(e), progress=0)
        raise
