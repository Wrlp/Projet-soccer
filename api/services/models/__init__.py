"""Modèles d'inférence enregistrés — découverte auto dans outputs/models/ + SlowFast."""

from api.config import MODELS_DIR, SLOWFAST_CKPT
from api.services.models import discovery, slowfast
from api.services.models.registry import (
    ModelSpec,
    default_model_id,
    default_threshold,
    ensure_ready,
    get,
    list_models,
    register,
)

for spec in discovery.discover_models(MODELS_DIR):
    register(spec)

register(
    ModelSpec(
        id="slowfast",
        name="SlowFast",
        description="SlowFast — fenêtres ~4 s @ 16 fps",
        path=SLOWFAST_CKPT,
        default_threshold=slowfast.DEFAULT_THRESHOLD,
        window_seconds=slowfast.WINDOW_SEC,
        infer_fps=slowfast.INFER_FPS,
        merge_gap_sec=slowfast.MERGE_GAP_SEC,
        is_ready=slowfast.is_model_ready,
        predict=slowfast.predict_video,
    )
)

__all__ = [
    "ModelSpec",
    "default_model_id",
    "default_threshold",
    "ensure_ready",
    "get",
    "list_models",
    "register",
]
