"""SportInsight AI — API"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.config import MODELS_DIR, UPLOADS_DIR
from api.services import jobs
from api.services.pipeline import run_analysis
from api.services.video_playback import ensure_playback_mp4

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="SportInsight AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process(job_id: str, video: Path):
    try:
        run_analysis(job_id, video)
    except Exception:
        pass


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    threshold: float = Form(0.18),
    context_frames: int = Form(0),
    half: str = Form("auto"),
):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
        raise HTTPException(400, f"Format non supporté : {ext}")
    job_id = jobs.create_job(video.filename or "video", {"threshold": threshold, "context_frames": context_frames, "half": half})
    dest = UPLOADS_DIR / f"{job_id}{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(video.file, f)
    background_tasks.add_task(_process, job_id, dest)
    return {"jobId": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    return {"status": job["status"], "progress": job.get("progress", 0), "error": job.get("error")}


_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    if not jobs.get_job(job_id):
        raise HTTPException(404, "Job introuvable")
    path = jobs.find_video_path(job_id, UPLOADS_DIR)
    if not path:
        raise HTTPException(404, "Vidéo introuvable pour ce job")
    try:
        playback = ensure_playback_mp4(path, job_id, UPLOADS_DIR)
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    media = _VIDEO_MIME.get(playback.suffix.lower(), "video/mp4")
    return FileResponse(playback, media_type=media, filename=playback.name)


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    if job["status"] == "failed":
        raise HTTPException(400, job.get("error", "Échec"))
    if job["status"] != "completed":
        raise HTTPException(202, "En cours")
    result = jobs.load_result(job_id)
    if not result:
        raise HTTPException(404, "Résultats indisponibles")
    return result


@app.get("/api/health")
def health():
    from api.config import SLOWFAST_CKPT

    ok = SLOWFAST_CKPT.exists()
    return {
        "status": "ok",
        "modelReady": ok,
        "model": "SlowFast",
        "checkpoint": str(SLOWFAST_CKPT),
    }
