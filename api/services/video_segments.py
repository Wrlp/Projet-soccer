"""
Découpe et métadonnées pour extraits courts ou matchs longs.
"""

from __future__ import annotations

import numpy as np

from api.config import FPS

# ~42 min : en dessous = extrait (pas une mi-temps SoccerNet complète)
EXCERPT_MAX_SEC = 42 * 60
# ~85 min : au dessus = on suppose 2 mi-temps dans un seul fichier
MATCH_MIN_SEC = 85 * 60


def duration_seconds(n_frames: int, fps: float = FPS) -> float:
    return n_frames / fps if fps > 0 else 0.0


def resolve_segments(
    features: np.ndarray,
    half_param: str | int,
    fps: float = FPS,
) -> tuple[list[tuple[int, np.ndarray]], dict]:
    """
    Retourne [(half, features_slice), ...] et métadonnées pour l'UI.

    - extrait court → 1 segment, temps 0 = début du clip
    - mi-temps choisie → 1 segment
    - auto + vidéo longue → 2 segments (coupe au milieu)
    """
    n = int(features.shape[0])
    dur = duration_seconds(n, fps)

    meta: dict = {
        "framesAnalyzed": n,
        "fps": fps,
        "durationSeconds": round(dur, 1),
        "durationMinutes": round(dur / 60, 2),
        "isExcerpt": dur < EXCERPT_MAX_SEC,
        "analysisMode": "excerpt",
    }

    if half_param == 1:
        meta["analysisMode"] = "half" if dur >= EXCERPT_MAX_SEC else "excerpt"
        meta["segmentLabel"] = "1ère mi-temps" if dur >= EXCERPT_MAX_SEC else "Extrait vidéo"
        return [(1, features)], meta

    if half_param == 2:
        meta["analysisMode"] = "half" if dur >= EXCERPT_MAX_SEC else "excerpt"
        meta["segmentLabel"] = "2ème mi-temps" if dur >= EXCERPT_MAX_SEC else "Extrait vidéo"
        return [(2, features)], meta

    # auto
    if dur < EXCERPT_MAX_SEC:
        meta["analysisMode"] = "excerpt"
        meta["isExcerpt"] = True
        meta["segmentLabel"] = "Extrait vidéo"
        return [(1, features)], meta

    if dur >= MATCH_MIN_SEC:
        mid = n // 2
        meta["analysisMode"] = "match"
        meta["isExcerpt"] = False
        meta["segmentLabel"] = "Match (2 mi-temps)"
        if mid <= 0:
            return [(1, features)], meta
        return [(1, features[:mid]), (2, features[mid:])], meta

    # Durée intermédiaire : une mi-temps ou un long extrait — une seule timeline
    meta["analysisMode"] = "half"
    meta["isExcerpt"] = True
    meta["segmentLabel"] = "Séquence unique"
    return [(1, features)], meta
