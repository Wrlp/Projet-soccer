"""Registre central des modèles d'inférence — point d'extension pour nouveaux modèles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from api.config import DEFAULT_MODEL

PredictFn = Callable[..., tuple[list[dict], dict]]
ReadyFn = Callable[[], bool]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    path: Path
    default_threshold: float
    window_seconds: float
    infer_fps: int
    merge_gap_sec: float
    is_ready: ReadyFn
    predict: PredictFn

    def to_api_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "available": self.is_ready(),
            "path": str(self.path),
            "defaultThreshold": self.default_threshold,
            "windowSeconds": self.window_seconds,
            "inferFps": self.infer_fps,
        }


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    _REGISTRY[spec.id] = spec


def get(model_id: str) -> ModelSpec:
    key = model_id.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(aucun)"
        raise KeyError(f"Modèle inconnu : {model_id}. Disponibles : {known}")
    return _REGISTRY[key]


def get_optional(model_id: str) -> ModelSpec | None:
    return _REGISTRY.get(model_id.lower())


def list_models() -> list[ModelSpec]:
    return list(_REGISTRY.values())


def default_model_id() -> str:
    if DEFAULT_MODEL in _REGISTRY:
        return DEFAULT_MODEL
    return next(iter(_REGISTRY.values())).id if _REGISTRY else DEFAULT_MODEL


def default_threshold(model_id: str) -> float:
    return get(model_id).default_threshold


def ensure_ready(model_id: str) -> None:
    spec = get(model_id)
    if not spec.is_ready():
        raise FileNotFoundError(f"Modèle {spec.name} indisponible : {spec.path}")
