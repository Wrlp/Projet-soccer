"""
Entraîner un modèle Random Forest pour la classification d'événements
"""

import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS_PATH = Path("outputs/models")
MODELS_PATH.mkdir(parents=True, exist_ok=True)


def train_random_forest(X_train_scaled, y_train):
    """Entraîner un Random Forest avec hyperparamètres optimisés (v2 conservative)."""
    print("Entraînement du Random Forest (v4 - Baseline+ClassWeight)")
    print("=" * 50)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("Paramètres (v4 - Baseline + ClassWeight):")
    print(f"  • n_estimators: 100")
    print(f"  • max_depth: 15")
    print(f"  • min_samples_split: 5")
    print(f"  • min_samples_leaf: 2")
    print()
    
    model.fit(X_train_scaled, y_train)
    print("Modèle entraîné\n")
    
    return model


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Évaluer le modèle sur train et test."""
    print("ÉVALUATION DU MODÈLE")
    print("=" * 50)
    
    # Prédictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Métriques de train
    train_acc = accuracy_score(y_train, y_pred_train)
    train_prec = precision_score(y_train, y_pred_train, average="weighted", zero_division=0)
    train_rec = recall_score(y_train, y_pred_train, average="weighted", zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
    
    # Métriques de test
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    
    print("\nTRAIN:")
    print(f"  Accuracy  : {train_acc:.4f}")
    print(f"  Precision : {train_prec:.4f}")
    print(f"  Recall    : {train_rec:.4f}")
    print(f"  F1-Score  : {train_f1:.4f}")
    
    print("\nTEST:")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    
    print(f"\nÉvaluation terminée\n")

    summary = {
        "train": {"acc": train_acc, "prec": train_prec, "rec": train_rec, "f1": train_f1},
        "test": {"acc": test_acc, "prec": test_prec, "rec": test_rec, "f1": test_f1},
    }
    import json
    with open(MODELS_PATH / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Métriques exportées : {MODELS_PATH / 'metrics_summary.json'}\n")

    return summary


def save_model(model, scaler, label_to_idx, idx_to_label, model_name="event_detection_model"):
    """Sauvegarder le modèle et les artéfacts."""
    print("SAUVEGARDE")
    print("=" * 50)
    
    pickle.dump(model, open(MODELS_PATH / f"{model_name}.pkl", "wb"))
    pickle.dump(scaler, open(MODELS_PATH / "scaler.pkl", "wb"))
    pickle.dump(label_to_idx, open(MODELS_PATH / "label_encoder.pkl", "wb"))
    pickle.dump(idx_to_label, open(MODELS_PATH / "idx_to_label.pkl", "wb"))

    meta_path = MODELS_PATH / "model_meta.json"
    import json

    n_feat = int(getattr(scaler, "n_features_in_", 512))
    ctx = max(0, (n_feat // 512 - 1) // 2) if n_feat % 512 == 0 else 0
    with open(meta_path, "w") as f:
        json.dump({"n_features": n_feat, "context_frames": ctx}, f, indent=2)
    
    print(f"Modèle sauvegardé dans {MODELS_PATH}/\n")
