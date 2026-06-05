from pathlib import Path

# Racine du projet (parent de api/)
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "outputs" / "models"
UPLOADS_DIR = ROOT / "outputs" / "uploads"
JOBS_DIR = ROOT / "outputs" / "jobs"
DATA_DIR = ROOT / "data" / "soccernet"
PROCESSED_DIR = ROOT / "outputs" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_NAME = "event_detection_model"
FPS = 2.0
DEFAULT_CONTEXT_FRAMES = 0

# Modèles d'inférence
VIDEOMAE_MODEL_DIR = MODELS_DIR / "videomae_soccernet" / "best_model"
DEFAULT_MODEL = "videomae"

# SlowFast (api/services/models/slowfast.py)
SLOWFAST_CKPT = ROOT / "SlowFast" / "checkpoints" / "best.pth"
SLOWFAST_SPLITS = ROOT / "SlowFast" / "splits.json"
SLOWFAST_T_S = 8
SLOWFAST_ALPHA = 8
SLOWFAST_IMAGE_SIZE = 112

for d in (MODELS_DIR, UPLOADS_DIR, JOBS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
