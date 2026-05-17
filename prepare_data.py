import numpy as np
import json
import pickle
from pathlib import Path

# Configuration
DATA_PATH = Path("data/soccernet")
OUTPUT_PATH = Path("outputs/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Chargement léger : uniquement les annotations + chemins
def load_match_light(match_path):
    with open(match_path / "Labels-v2.json") as f:
        labels = json.load(f)

    events = [
        {
            "label": a["label"],
            "half": int(a["gameTime"].split(" - ")[0]),
            "time_seconds": (
                int(a["gameTime"].split(" - ")[1].split(":")[0]) * 60
                + int(a["gameTime"].split(" - ")[1].split(":")[1])
            )
        }
        for a in labels["annotations"]
    ]

    return {
        "match_id": str(match_path),
        "features_path_1": str(match_path / "1_ResNET_TF2_PCA512.npy"),
        "features_path_2": str(match_path / "2_ResNET_TF2_PCA512.npy"),
        "events": events
        # les features ne sont PAS chargées ici, juste le chemin
    }

# Traitement 
match_dirs = [p.parent for p in DATA_PATH.rglob("Labels-v2.json")]
print(f"Matchs trouvés : {len(match_dirs)}")

all_matches = []
for match_path in match_dirs:
    try:
        match = load_match_light(match_path)
        all_matches.append(match)
        print(f"OK : {match_path.name}")
    except Exception as e:
        print(f"Erreur sur {match_path.name} : {e}")

# Sauvegarde 
output_file = OUTPUT_PATH / "matches.pkl"
with open(output_file, "wb") as f:
    pickle.dump(all_matches, f)

print(f"\nSauvegardé : {output_file} ({len(all_matches)} matchs)")
print("Préparation terminée.")