import numpy as np
import json
import pickle
from pathlib import Path

# Configuration
DATA_PATH = Path("data/soccernet")
OUTPUT_PATH = Path("outputs/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Chargement d'un match 
def load_match(match_path):
    features_1 = np.load(match_path / "1_ResNET_TF2_PCA512.npy")
    features_2 = np.load(match_path / "2_ResNET_TF2_PCA512.npy")

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
        "features_1": features_1,   # shape (N, 512)
        "features_2": features_2,   # shape (M, 512)
        "events": events
    }

# Traitement de tous les matchs 
def process_all(split="train"):
    split_path = DATA_PATH / split
    if not split_path.exists():
        print(f"Dossier introuvable : {split_path}")
        return []

    all_matches = []
    match_dirs = [p.parent for p in split_path.rglob("Labels-v2.json")]

    print(f"Matchs trouvés ({split}) : {len(match_dirs)}")

    for match_path in match_dirs:
        try:
            match = load_match(match_path)
            all_matches.append(match)
        except Exception as e:
            print(f"Erreur sur {match_path.name} : {e}")

    return all_matches

# Sauvegarde 
for split in ["train", "valid", "test"]:
    matches = process_all(split)

    if matches:
        output_file = OUTPUT_PATH / f"{split}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(matches, f)
        print(f"Sauvegardé : {output_file} ({len(matches)} matchs)")

print("\nPréparation terminée. Données dans outputs/processed/")