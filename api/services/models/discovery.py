"""Découverte automatique des modèles dans outputs/models/."""

from __future__ import annotations

from pathlib import Path

from api.config import DEFAULT_MODEL, MODELS_DIR
from api.services.models import sklearn_rf, videomae
from api.services.models.registry import ModelSpec

# Alias rétrocompatibles : dossier → id API
_FOLDER_ID_ALIASES: dict[str, str] = {
    "videomae_soccernet": "videomae",
}


def discover_models(models_dir: Path | None = None) -> list[ModelSpec]:
    root = Path(models_dir or MODELS_DIR)
    if not root.exists():
        return []

    specs: list[ModelSpec] = []
    seen_ids: set[str] = set()

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        best = entry / "best_model"
        weights = best / "model.safetensors"
        if not weights.exists():
            continue

        folder = entry.name
        model_id = _FOLDER_ID_ALIASES.get(folder, videomae.slug_from_folder(folder))
        if model_id in seen_ids:
            model_id = videomae.slug_from_folder(folder)
        seen_ids.add(model_id)

        name = videomae.display_name_from_folder(folder)
        specs.append(videomae.make_videomae_spec(model_id, name, best))

    if sklearn_rf.is_model_ready():
        specs.append(sklearn_rf.make_sklearn_spec())

    return specs


def default_from_discovered(specs: list[ModelSpec]) -> str:
    ids = {s.id for s in specs}
    if DEFAULT_MODEL in ids:
        return DEFAULT_MODEL
    return specs[0].id if specs else DEFAULT_MODEL
