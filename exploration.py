import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Configuration 
DATA_PATH = Path("data/soccernet")
OUTPUT_PATH = Path("outputs/exploration")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Charger un match exemple
match_path = next(DATA_PATH.rglob("Labels-v2.json")).parent
print(f"Match chargé : {match_path}")

features_1 = np.load(match_path / "1_ResNET_TF2_PCA512.npy")
features_2 = np.load(match_path / "2_ResNET_TF2_PCA512.npy")

with open(match_path / "Labels-v2.json") as f:
    labels = json.load(f)

# Infos de base 
print(f"\nFeatures 1ère mi-temps : {features_1.shape}")  # (N, 512)
print(f"Features 2ème mi-temps : {features_2.shape}")  # (M, 512)
print(f"Nombre d'événements    : {len(labels['annotations'])}")
print(f"\nExemple d'annotation :")
print(json.dumps(labels["annotations"][0], indent=2))

# Graphique 1 : distribution des classes 
all_labels = [a["label"] for a in labels["annotations"]]
counter = Counter(all_labels)

plt.figure(figsize=(12, 5))
plt.bar(counter.keys(), counter.values(), color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.title("Distribution des événements")
plt.ylabel("Nombre d'occurrences")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "distribution_classes.png")
plt.close()
print("\nGraphique sauvegardé : distribution_classes.png")

# Graphique 2 : timeline des événements du match 
def parse_time(game_time_str):
    # Format : "1 - 23:45" → (half, seconds)
    parts = game_time_str.split(" - ")
    half = int(parts[0])
    mm, ss = parts[1].split(":")
    return half, int(mm) * 60 + int(ss)

events = []
for a in labels["annotations"]:
    try:
        half, seconds = parse_time(a["gameTime"])
        events.append({"label": a["label"], "half": half, "seconds": seconds})
    except:
        continue

colors = {
    "Goal": "gold", "Yellow card": "yellow", "Red card": "red",
    "Corner": "blue", "Substitution": "green", "Foul": "orange",
    "Shots on target": "purple", "Shots off target": "mediumpurple",
    "Direct free-kick": "cyan", "Indirect free-kick": "lightblue",
    "Offside": "pink", "Clearance": "gray", "Throw-in": "lightgray",
    "Ball out of play": "darkgray", "Kick-off": "lime",
    "Penalty": "crimson", "Yellow->red card": "darkorange"
}

plt.figure(figsize=(14, 4))
for e in events:
    offset = 0 if e["half"] == 1 else 5400  # 90 min offset pour la 2e mi-temps
    x = e["seconds"] + offset
    color = colors.get(e["label"], "black")
    plt.axvline(x=x, color=color, alpha=0.7, linewidth=1.5, label=e["label"])

# Légende sans doublons
handles, lbls = plt.gca().get_legend_handles_labels()
by_label = dict(zip(lbls, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=6, loc="upper right", ncol=2)
plt.axvline(x=5400, color="black", linewidth=2, linestyle="--", label="Mi-temps")
plt.xlabel("Temps (secondes)")
plt.title("Timeline des événements du match")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "timeline_match.png")
plt.close()
print("Graphique sauvegardé : timeline_match.png")

# Graphique 3 : valeurs moyennes des features dans le temps
plt.figure(figsize=(12, 4))
mean_features = features_1.mean(axis=1)  # moyenne sur les 512 dims
plt.plot(mean_features, color="steelblue", linewidth=0.8)
plt.title("Moyenne des features ResNET au fil du temps (1ère mi-temps)")
plt.xlabel("Frame (1 frame = 0.5s à 2fps)")
plt.ylabel("Valeur moyenne")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "features_temporelles.png")
plt.close()
print("Graphique sauvegardé : features_temporelles.png")

print("\nExploration terminée. Graphiques dans outputs/exploration/")