import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter


def _extract_window(features, center_idx, window_size_frames):
    """Extraire une fenêtre temporelle de taille fixe autour d'un index central."""
    n_frames = len(features)

    if window_size_frames <= 0:
        raise ValueError("window_size_frames doit être strictement positif")

    start_idx = center_idx - window_size_frames // 2
    end_idx = start_idx + window_size_frames

    if start_idx < 0:
        start_idx = 0
        end_idx = min(window_size_frames, n_frames)

    if end_idx > n_frames:
        end_idx = n_frames
        start_idx = max(0, end_idx - window_size_frames)

    window = features[start_idx:end_idx]

    if len(window) < window_size_frames:
        padding = np.zeros((window_size_frames - len(window), features.shape[1]))
        window = np.vstack([window, padding])

    return window


def build_dataset(matches, context_frames=3, window_size_frames=None):
    """
    Construire X (features) et y (labels) pour l'entraînement.
    
    Pour chaque événement d'un match:
    - Charge la feature .npy correspondante
    - Extrait un CONTEXTE de frames autour de l'événement (avant/après)
    - La couple avec le label (type d'événement)
    
    Args:
        matches: liste des matchs
        context_frames: nombre de frames avant/après à inclure dans le contexte
        window_size_frames: taille exacte de la fenêtre temporelle à extraire
    """
    print("Construction du dataset (avec contexte temporel)")
    print("=" * 50)

    if window_size_frames is None:
        window_size_frames = context_frames * 2 + 1
        print(f"Contexte: {context_frames} frames avant/après l'événement")
        print(f"Taille de fenêtre effective: {window_size_frames} frames\n")
    else:
        print(f"Taille de fenêtre fixe: {window_size_frames} frames\n")
    
    # Créer l'encodeur de labels
    all_events = []
    for match in matches:
        all_events.extend(match["events"])
    
    unique_labels = sorted(set(e["label"] for e in all_events))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Nombre de classes: {len(unique_labels)}")
    print(f"Classes: {', '.join(list(label_to_idx.keys())[:5])}...\n")
    
    # Construire X et y
    X_list = []
    y_list = []
    skipped = 0
    
    for match_idx, match in enumerate(matches):
        try:
            # Charger les features ResNet-PCA 512-dim
            features_1 = np.load(match["features_path_1"])
            features_2 = np.load(match["features_path_2"])
            
            n_frames_1 = features_1.shape[0]
            n_frames_2 = features_2.shape[0]
            
            # Pour chaque événement, extraire le contexte temporel
            for event in match["events"]:
                # Approximation 2 fps
                if event["half"] == 1:
                    center_idx = min(int(event["time_seconds"] / 2), n_frames_1 - 1)
                    features = features_1
                else:
                    center_idx = min(int(event["time_seconds"] / 2), n_frames_2 - 1)
                    features = features_2
                
                # Extraire une fenêtre temporelle fixe autour de l'événement
                context_features = _extract_window(features, center_idx, window_size_frames)
                feature_vector = context_features.flatten()
                
                X_list.append(feature_vector)
                y_list.append(label_to_idx[event["label"]])
        
        except FileNotFoundError:
            skipped += 1
            continue
        except Exception as e:
            print(f" Erreur match {match_idx}: {e}")
            skipped += 1
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Dataset créé:")
    print(f"  • {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  • {len(np.unique(y))} classes")
    print(f"  • Matchs ignorés: {skipped}\n")
    
    # Afficher la distribution
    print("Distribution des classes:")
    for label_idx, count in sorted(Counter(y).items(), key=lambda x: x[1], reverse=True)[:10]:
        label_name = idx_to_label[label_idx]
        pct = (count / len(y)) * 100
        print(f"  • {label_name:25s}: {count:4d} ({pct:5.1f}%)")
    
    return X, y, label_to_idx, idx_to_label


def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """Diviser en train/test et normaliser."""
    print(f"\nSéparation train/test (test_size={test_size})")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normaliser
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {X_train_scaled.shape[0]} samples")
    print(f"Test:  {X_test_scaled.shape[0]} samples")
    print(f"Données normalisées (StandardScaler)\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
