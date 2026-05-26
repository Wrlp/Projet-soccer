import os
import json
import numpy as np
import cv2
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

# Configuration 
DATA_PATH = Path("data/soccernet")
CLIPS_PATH = Path("outputs/clips")
CLIPS_PATH.mkdir(parents=True, exist_ok=True)

SECONDS_BEFORE = 3
SECONDS_AFTER = 2

# Max clips par classe basé sur les vraies occurrences du dataset
MAX_CLIPS_PER_CLASS = {
    "Yellow->red card":    46,
    "Red card":            55,
    "Penalty":            173,
    "Goal":               500,
    "Yellow card":        500,
    "Offside":            500,
    "Direct free-kick":   500,
    "Corner":             500,
    "Shots off target":   500,
    "Shots on target":    500,
    "Indirect free-kick": 500,
    "Foul":               500,
}

CLASSES = list(MAX_CLIPS_PER_CLASS.keys())
clip_counter = defaultdict(int)

# Fonctions utilitaires
def all_classes_done():
    return all(clip_counter[c] >= MAX_CLIPS_PER_CLASS[c] for c in CLASSES)

def needs_more_clips(label):
    if label not in MAX_CLIPS_PER_CLASS:
        return False
    return clip_counter[label] < MAX_CLIPS_PER_CLASS[label]

def extract_clip(video_path, time_seconds, npy_path, mp4_path):
    """Extrait un clip de SECONDS_BEFORE avant à SECONDS_AFTER après l'événement"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        cap.release()
        return False

    # Convertir secondes en frames
    frames_before = int(SECONDS_BEFORE * video_fps)
    frames_after = int(SECONDS_AFTER * video_fps)

    center_frame = int(time_seconds * video_fps)
    start_frame = max(0, center_frame - frames_before)
    end_frame = center_frame + frames_after

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (224, 224))
        # frame = cv2.resize(frame, (112, 112))
        frame = cv2.resize(frame, (720, 720))
        frames.append(frame)

    cap.release()

    if not frames:
        return False

    # Sauvegarde .npy (personne 2 : entraînement)
    np.save(npy_path, np.array(frames))

    # Sauvegarde .mp4 (personne 3 : interface)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(str(mp4_path), fourcc, video_fps, (224, 224))
    # out = cv2.VideoWriter(str(mp4_path), fourcc, video_fps, (112, 112))
    out = cv2.VideoWriter(str(mp4_path), fourcc, video_fps, (720, 720))
    for frame in frames:
        out.write(frame)
    out.release()

    return True

def parse_time(game_time_str):
    """Convertit '1 - 23:45' en (half, seconds)"""
    parts = game_time_str.split(" - ")
    half = int(parts[0])
    mm, ss = parts[1].split(":")
    return half, int(mm) * 60 + int(ss)

# Téléchargement et extraction match par match 
downloader = SoccerNetDownloader(LocalDirectory=str(DATA_PATH))
downloader.password = os.getenv("SOCCERNET_PASSWORD")

games = getListGames(split=["train", "valid", "test"])
print(f"Total matchs disponibles : {len(games)}")
print(f"Total clips à extraire   : {sum(MAX_CLIPS_PER_CLASS.values())}")
print(f"Durée des clips          : {SECONDS_BEFORE}s avant + {SECONDS_AFTER}s après\n")

for i, game in enumerate(games):
    if all_classes_done():
        print("\nToutes les classes sont complètes — arrêt.")
        break

    match_path = DATA_PATH / game
    label_file = match_path / "Labels-v2.json"

    if not label_file.exists():
        print(f"[{i+1}/{len(games)}] Skip {game} — pas d'annotations")
        continue

    with open(label_file) as f:
        labels = json.load(f)

    # Vérifier si ce match a des événements utiles
    useful_events = [
        a for a in labels["annotations"]
        if needs_more_clips(a["label"])
    ]

    if not useful_events:
        print(f"[{i+1}/{len(games)}] Skip {game} — rien d'utile")
        continue

    # Télécharger les vidéos du match
    print(f"\n[{i+1}/{len(games)}] Téléchargement : {game}")
    try:
        downloader.downloadGame(
            files=["1_224p.mkv", "2_224p.mkv"],
            game=game
        )
    except Exception as e:
        print(f"Erreur téléchargement : {e}")
        continue

    # Extraire les clips
    for event in useful_events:
        label = event["label"]
        if not needs_more_clips(label):
            continue

        try:
            half, time_seconds = parse_time(event["gameTime"])
        except:
            continue

        video_file = match_path / f"{half}_224p.mkv"
        if not video_file.exists():
            continue

        # Créer les dossiers par classe
        class_name = label.replace(" ", "_").replace("->", "_")
        npy_dir = CLIPS_PATH / "npy" / class_name
        mp4_dir = CLIPS_PATH / "mp4" / class_name
        npy_dir.mkdir(parents=True, exist_ok=True)
        mp4_dir.mkdir(parents=True, exist_ok=True)

        idx = clip_counter[label]
        npy_file = npy_dir / f"{idx:04d}.npy"
        mp4_file = mp4_dir / f"{idx:04d}.mp4"

        if extract_clip(video_file, time_seconds, npy_file, mp4_file):
            clip_counter[label] += 1
            print(f"  ✓ [{label}] {clip_counter[label]}/{MAX_CLIPS_PER_CLASS[label]}")

    # Supprimer les vidéos après extraction
    for half in [1, 2]:
        video_file = match_path / f"{half}_224p.mkv"
        if video_file.exists():
            video_file.unlink()

    # Afficher l'état global toutes les 10 vidéos
    if (i + 1) % 10 == 0:
        print("\n── État actuel ──────────────────────────────")
        for c in CLASSES:
            bar = "█" * clip_counter[c] + "░" * (MAX_CLIPS_PER_CLASS[c] - clip_counter[c])
            # Limiter la barre à 20 caractères
            ratio = clip_counter[c] / MAX_CLIPS_PER_CLASS[c]
            filled = int(ratio * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {c:<25} {bar} {clip_counter[c]}/{MAX_CLIPS_PER_CLASS[c]}")
        print()

# Résumé final
print("\n── Résumé final")
total_extracted = sum(clip_counter.values())
total_expected = sum(MAX_CLIPS_PER_CLASS.values())
for c in CLASSES:
    status = "✓" if clip_counter[c] >= MAX_CLIPS_PER_CLASS[c] else "✗"
    print(f"  {status} {c:<25} {clip_counter[c]}/{MAX_CLIPS_PER_CLASS[c]}")

print(f"\nTotal : {total_extracted}/{total_expected} clips")
print(f"Clips sauvegardés dans : {CLIPS_PATH}")