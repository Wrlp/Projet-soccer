# Données SoccerNet 

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Crée un fichier `.env` à la racine :
```
SOCCERNET_PASSWORD=ton_mot_de_passe_ici
```
Le mot de passe s'obtient en remplissant le formulaire NDA sur soccer-net.org.

## Étapes
1. python download_data.py   -> télécharge les données
2. python exploration.py     -> génère les graphiques dans outputs/exploration/
3. python prepare_data.py    -> génère outputs/processed/matches.pkl

## Format de matches.pkl
Liste de dictionnaires, un par match :
- match_id        : chemin du match
- features_path_1 : chemin vers les features 1ère mi-temps (N, 512)
- features_path_2 : chemin vers les features 2ème mi-temps (M, 512)
- events          : liste d'événements
    - label        : type d'événement (Goal, Yellow card, Corner...)
    - half         : mi-temps (1 ou 2)
    - time_seconds : timestamp en secondes

## Charger les données 
import pickle
import numpy as np

with open("outputs/processed/matches.pkl", "rb") as f:
    matches = pickle.load(f)

match = matches[0]
features = np.load(match["features_path_1"])  # shape (N, 512)
events = match["events"]

## Notes
- `.env`, `data/` et `outputs/` sont dans le `.gitignore` - non versionnés
- Les features `.npy` suffisent pour l'entraînement baseline 
- Les vidéos `.mkv` sont utiles pour l'option B (GPU requis)
- Vidéos et données soumises au NDA SoccerNet — ne pas partager

# Supprimer uniquement les vidéos
find data/soccernet -name "*.mkv" -delete