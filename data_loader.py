import pickle
import numpy as np
from pathlib import Path
from collections import Counter

OUTPUT_PATH = Path("outputs/processed")


def load_matches():
    """Charger les données préparées."""
    print("Chargement de matches.pkl...")
    with open(OUTPUT_PATH / "matches.pkl", "rb") as f:
        matches = pickle.load(f)
    print(f"{len(matches)} matchs chargés\n")
    return matches


def explore_data(matches):
    """Explorer et afficher les statistiques des données."""
    print("EXPLORATION DES DONNÉES")
    print("=" * 50)
    
    # Nombre de matchs
    print(f"Nombre de matchs: {len(matches)}")
    
    # Structure d'un match
    print(f"\nStructure d'un match:")
    match_sample = matches[0]
    for key in match_sample:
        if key != "events":
            print(f"  • {key}: {match_sample[key]}")
        else:
            print(f"  • {key}: {len(match_sample['events'])} événements")
    
    # Collectionner tous les événements
    all_events = []
    for match in matches:
        all_events.extend(match["events"])
    
    # Statistiques des événements
    event_labels = [e["label"] for e in all_events]
    event_counter = Counter(event_labels)
    
    print(f"\nStatistiques des événements:")
    print(f"  - Total d'événements: {len(all_events)}")
    print(f"  - Nombre de classes: {len(event_counter)}")
    print(f"\n  Top 10 événements:")
    for label, count in event_counter.most_common(10):
        pct = (count / len(all_events)) * 100
        print(f"    - {label:25s}: {count:4d} ({pct:5.1f}%)")
    
    # Vérifier les features
    print(f"\nVérification des features .npy:")
    valid_matches = 0
    for match in matches[:5]:  # Vérifier les 5 premiers
        try:
            features_1 = np.load(match["features_path_1"])
            features_2 = np.load(match["features_path_2"])
            valid_matches += 1
            print(f"  Match {valid_matches}: features_1={features_1.shape}, features_2={features_2.shape}")
        except FileNotFoundError as e:
            print(f"  Fichier manquant: {e}")
    
    print(f"\nExploration terminée\n")
    return matches


if __name__ == "__main__":
    matches = load_matches()
    explore_data(matches)
