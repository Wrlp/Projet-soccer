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

from api.config import UPLOADS_DIR
from api.services import jobs
from api.services.models.common import DEFAULT_STRIDE_SEC
from api.services.models import default_model_id, get, list_models
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


@app.get("/api/models")
def api_list_models():
    return {
        "default": default_model_id(),
        "models": [spec.to_api_dict() for spec in list_models()],
    }


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    threshold: float = Form(0.35),
    context_frames: int = Form(0),
    stride_sec: float = Form(DEFAULT_STRIDE_SEC),
    half: str = Form("auto"),
    model: str = Form(default_model_id()),
):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
        raise HTTPException(400, f"Format non supporté : {ext}")
    model_key = (model or default_model_id()).lower()
    try:
        spec = get(model_key)
    except KeyError:
        raise HTTPException(400, f"Modèle inconnu : {model}") from None
    if not spec.is_ready():
        raise HTTPException(
            503,
            f"Modèle {spec.name} indisponible — poids manquants ou non téléchargés (git lfs pull). "
            f"Chemin : {spec.path}",
        )
    if stride_sec < 0.5 or stride_sec > 10:
        raise HTTPException(400, "stride_sec doit être entre 0.5 et 10 secondes")
    job_id = jobs.create_job(
        video.filename or "video",
        {
            "threshold": threshold,
            "context_frames": context_frames,
            "stride_sec": stride_sec,
            "half": half,
            "model": model_key,
        },
    )
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
    specs = list_models()
    ready = {spec.id: spec.is_ready() for spec in specs}
    return {
        "status": "ok",
        "modelReady": any(ready.values()),
        "defaultModel": default_model_id(),
        "models": {spec.id: {"ready": ready[spec.id], "path": str(spec.path)} for spec in specs},
    }
