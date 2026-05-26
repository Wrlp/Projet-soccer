from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import JOBS_DIR


def _path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def create_job(video_name: str, params: dict) -> str:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    data = {
        "id": job_id,
        "status": "pending",
        "videoName": video_name,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
        "params": params,
        "error": None,
    }
    _path(job_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return job_id


def get_job(job_id: str) -> dict | None:
    p = _path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def update_job(job_id: str, **kwargs: Any) -> dict:
    data = get_job(job_id)
    if not data:
        raise KeyError(job_id)
    data.update(kwargs)
    _path(job_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def load_result(job_id: str) -> dict | None:
    p = JOBS_DIR / f"{job_id}_result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_video_path(job_id: str, uploads_dir: Path) -> Path | None:
    """Retourne le fichier vidéo uploadé pour ce job (job-{id}.ext)."""
    for p in sorted(uploads_dir.glob(f"{job_id}.*")):
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
            return p
    return None
