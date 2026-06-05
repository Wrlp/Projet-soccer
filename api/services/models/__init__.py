"""Modèles d'inférence enregistrés — ajouter un module + register() pour en étendre la liste."""

from api.config import SLOWFAST_CKPT, VIDEOMAE_MODEL_DIR
from api.services.models import slowfast, videomae
from api.services.models.registry import (
    ModelSpec,
    default_model_id,
    default_threshold,
    ensure_ready,
    get,
    list_models,
    register,
)

register(
    ModelSpec(
        id="videomae",
        name="VideoMAE",
        description="VideoMAE SoccerNet — fenêtres ~1 s @ 16 fps",
        path=VIDEOMAE_MODEL_DIR,
        default_threshold=videomae.DEFAULT_THRESHOLD,
        window_seconds=videomae.WINDOW_SEC,
        infer_fps=videomae.INFER_FPS,
        merge_gap_sec=videomae.MERGE_GAP_SEC,
        is_ready=videomae.is_model_ready,
        predict=videomae.predict_video,
    )
)
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
