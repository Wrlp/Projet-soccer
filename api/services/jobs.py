from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import JOBS_DIR


def _path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_json(path: Path, data: dict) -> None:
    """Écriture atomique pour éviter les lectures partielles (race avec le polling front)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, retries: int = 5) -> dict | None:
    if not path.exists():
        return None
    for attempt in range(retries):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise json.JSONDecodeError("empty file", text, 0)
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)
    return None


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
    _write_json(_path(job_id), data)
    return job_id


def get_job(job_id: str) -> dict | None:
    return _read_json(_path(job_id))


def update_job(job_id: str, **kwargs: Any) -> dict:
    data = get_job(job_id)
    if not data:
        raise KeyError(job_id)
    data.update(kwargs)
    _write_json(_path(job_id), data)
    return data


def load_result(job_id: str) -> dict | None:
    return _read_json(JOBS_DIR / f"{job_id}_result.json")


def find_video_path(job_id: str, uploads_dir: Path) -> Path | None:
    """Retourne le fichier vidéo uploadé pour ce job (job-{id}.ext)."""
    for p in sorted(uploads_dir.glob(f"{job_id}.*")):
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
            return p
    return None
