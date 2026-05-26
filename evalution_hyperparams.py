import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import pickle

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import ParameterGrid



COLORS = {
    "primary":   "#1a6fbc",
    "secondary": "#0f2d4a",
    "goal":      "#185fa5",
    "card":      "#854f0b",
    "sub":       "#3b6d11",
    "shot":      "#993556",
    "corner":    "#3c3489",
}

EVENT_CLASSES = ["But", "Carton jaune", "Carton rouge", "Remplacement", "Tir", "Corner"]

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#Chargement model

def load_model_from_person2(model_path="model.pkl"):
    """
    Charge le modèle, le scaler et les labels

    Retourne : model, scaler, label_to_idx, idx_to_label
    """
    if not os.path.exists(model_path):
        print(f"[WARN] Fichier '{model_path}' introuvable → mode données simulées activé.")
        return None, None, None, None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model        = bundle["model"]
    scaler       = bundle["scaler"]
    label_to_idx = bundle["label_to_idx"]
    idx_to_label = bundle["idx_to_label"]

    print(f"[OK] Modèle chargé depuis '{model_path}'")
    print(f"     Classes : {list(label_to_idx.keys())}")
    return model, scaler, label_to_idx, idx_to_label


def load_test_data(test_path="test_data.pkl"):
    """
    Charge X_test et y_test
    Si le fichier n'existe pas → données simulées.
    """
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            data = pickle.load(f)
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"[OK] Données de test chargées : {X_test.shape[0]} échantillons")
        return X_test, y_test

    # ── Données simulées ──
    print("[WARN] test_data.pkl introuvable → données simulées.")
    np.random.seed(42)
    n = 200
    X_test = np.random.rand(n, 10)
    y_test = np.random.choice(range(len(EVENT_CLASSES)), size=n)
    return X_test, y_test


#Prédiction 

def get_predictions(model, scaler, X_test, idx_to_label):
    """
    Applique le scaler puis le modèle sur X_test.
    Retourne y_pred (indices) et y_pred_labels (noms des classes).
    """
    if model is None:
        # Données simulées si pas de modèle
        y_pred = np.random.choice(range(len(EVENT_CLASSES)), size=len(X_test))
    else:
        X_scaled = scaler.transform(X_test)
        y_pred   = model.predict(X_scaled)

    # Convertit les indices en noms de classes
    if idx_to_label:
        y_pred_labels = [idx_to_label[i] for i in y_pred]
    else:
        y_pred_labels = [EVENT_CLASSES[i % len(EVENT_CLASSES)] for i in y_pred]

    return y_pred, y_pred_labels


def idx_to_class_names(y_indices, idx_to_label):
    """Convertit un tableau d'indices en liste de noms de classes."""
    if idx_to_label:
        return [idx_to_label[i] for i in y_indices]
    return [EVENT_CLASSES[i % len(EVENT_CLASSES)] for i in y_indices]


# Hyperparamètres

def hyperparameter_grid():
    """
    Grille d'hyperparamètres pour le Random Forest
    Ces valeurs peuvent être passées à GridSearchCV.
    """
    param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2"],
        "class_weight":      ["balanced", None],
    }

    all_configs = list(ParameterGrid(param_grid))
    print(f"[INFO] {len(all_configs)} configurations possibles dans la grille.")
    return param_grid, all_configs


def plot_hyperparam_impact(save=True):
    """
    Figure 1 — Impact de n_estimators et max_depth sur la mAP.
    Remplace map_scores par tes vraies valeurs après GridSearch.
    """
    n_estimators_list = [50, 100, 200]
    max_depth_list    = [10, 20, None]

    # Résultats simulés — remplace par tes vraies valeurs
    map_scores = np.array([
        [44.1, 47.3, 45.8],  # n_estimators=50,  max_depth 10/20/None
        [51.2, 58.6, 57.1],  # n_estimators=100
        [53.7, 61.2, 60.9],  # n_estimators=200
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(max_depth_list))
    width = 0.25

    for i, n in enumerate(n_estimators_list):
        bars = ax.bar(x + i * width, map_scores[i], width,
                      label=f"n_estimators={n}",
                      color=list(COLORS.values())[i], alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3)

    ax.set_xlabel("max_depth", fontsize=12)
    ax.set_ylabel("mAP (%)", fontsize=12)
    ax.set_title("Impact des hyperparamètres (Random Forest) sur la mAP",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["depth=10", "depth=20", "depth=None"])
    ax.legend(title="n_estimators")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig1_hyperparam_impact.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_threshold_impact(model, scaler, X_test, y_test, idx_to_label, save=True):
    """
    Figure 2 — Précision vs Rappel selon le seuil de probabilité.
    Utilise predict_proba() du Random Forest.
    """
    thresholds = np.linspace(0.1, 0.9, 40)
    precisions, recalls = [], []

    for thresh in thresholds:
        if model is not None:
            X_scaled = scaler.transform(X_test)
            proba    = model.predict_proba(X_scaled)
            y_pred   = np.array([
                model.classes_[np.argmax(p)] if np.max(p) >= thresh else -1
                for p in proba
            ])
            mask   = y_pred != -1
            y_p    = y_pred[mask]
            y_t    = np.array(y_test)[mask]
        else:
            # Simulé
            y_p = np.random.choice(range(len(EVENT_CLASSES)), size=len(y_test))
            y_t = np.array(y_test)

        if len(y_p) == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            continue

        precisions.append(precision_score(y_t, y_p, average="macro",
                                          zero_division=0,
                                          labels=list(range(len(EVENT_CLASSES)))))
        recalls.append(recall_score(y_t, y_p, average="macro",
                                    zero_division=0,
                                    labels=list(range(len(EVENT_CLASSES)))))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, color=COLORS["primary"], linewidth=2, label="Précision")
    ax.plot(thresholds, recalls,    color=COLORS["card"],    linewidth=2, label="Rappel")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Seuil = 0.5")
    ax.set_xlabel("Seuil de détection", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Précision vs Rappel selon le seuil", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig2_threshold_precision_recall.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


# Metrics

def compute_metrics(y_true_labels, y_pred_labels):
    """
    Calcule précision, rappel et F1 à partir des noms de classes.
    """
    precision = precision_score(y_true_labels, y_pred_labels,
                                average="macro", zero_division=0,
                                labels=EVENT_CLASSES)
    recall    = recall_score(y_true_labels, y_pred_labels,
                             average="macro", zero_division=0,
                             labels=EVENT_CLASSES)
    f1        = f1_score(y_true_labels, y_pred_labels,
                         average="macro", zero_division=0,
                         labels=EVENT_CLASSES)

    print("\n" + "="*45)
    print("   MÉTRIQUES — SportInsight AI")
    print("="*45)
    print(f"  Précision macro  : {precision*100:.1f}%")
    print(f"  Rappel macro     : {recall*100:.1f}%")
    print(f"  F1-score macro   : {f1*100:.1f}%")
    print("="*45 + "\n")

    return {"precision": precision, "recall": recall, "f1": f1}


# Figures

def plot_map_per_class(y_true_labels, y_pred_labels, save=True):
    """
    Figure 3 — F1-score par classe (approximation de la mAP par classe).
    """
    scores = {}
    for cls in EVENT_CLASSES:
        y_t = [1 if y == cls else 0 for y in y_true_labels]
        y_p = [1 if y == cls else 0 for y in y_pred_labels]
        scores[cls] = f1_score(y_t, y_p, zero_division=0) * 100

    classes = list(scores.keys())
    values  = list(scores.values())
    colors  = [COLORS["goal"], COLORS["card"], "#a32d2d",
               COLORS["sub"], COLORS["shot"], COLORS["corner"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(classes, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=6, fontsize=10)
    ax.axvline(x=np.mean(values), color="gray", linestyle="--",
               label=f"Moyenne : {np.mean(values):.1f}%")
    ax.set_xlabel("F1-score (%)", fontsize=12)
    ax.set_title("F1-score par classe d'événement", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig3_score_per_class.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_confusion_matrix(y_true_labels, y_pred_labels, save=True):
    """
    Figure 4 — Matrice de confusion.
    """
    cm   = confusion_matrix(y_true_labels, y_pred_labels, labels=EVENT_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EVENT_CLASSES)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
    ax.set_title("Matrice de confusion — Détection d'événements",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig4_confusion_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_metrics_table(metrics, save=True):
    """
    Figure 5 — Tableau récapitulatif des métriques.
    (Remplace la courbe AUC comme demandé dans les commentaires du code)
    """
    rows   = [["Précision macro", f"{metrics['precision']*100:.1f}%"],
              ["Rappel macro",    f"{metrics['recall']*100:.1f}%"],
              ["F1-score macro",  f"{metrics['f1']*100:.1f}%"]]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Métrique", "Valeur"],
        cellLoc="center",
        loc="center",
        colColours=["#0f2d4a", "#0f2d4a"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.4, 2)

    # Couleurs header blanc
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f6ff" if r % 2 == 0 else "white")

    ax.set_title("Résultats d'évaluation — SportInsight AI",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig5_metrics_table.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_training_curves(train_losses=None, val_losses=None, val_maps=None, save=True):
    """
    Figure 6 — Courbes d'entraînement.
    Passe tes vraies listes train_losses, val_losses, val_maps si disponibles.
    """
    epochs = np.arange(1, 51)
    if train_losses is None:
        np.random.seed(0)
        train_losses = 1.8 * np.exp(-0.07 * epochs) + np.random.normal(0, 0.03, 50)
        val_losses   = 1.9 * np.exp(-0.06 * epochs) + np.random.normal(0, 0.04, 50) + 0.1
        val_maps     = 60  * (1 - np.exp(-0.08 * epochs)) + np.random.normal(0, 1, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, color=COLORS["primary"], linewidth=2, label="Train loss")
    ax1.plot(epochs, val_losses,   color=COLORS["card"],    linewidth=2, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Courbe de loss", fontweight="bold")
    ax1.legend(); ax1.grid(linestyle="--", alpha=0.4)

    ax2.plot(epochs, val_maps, color=COLORS["sub"], linewidth=2)
    ax2.fill_between(epochs, val_maps, alpha=0.15, color=COLORS["sub"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP (%)")
    ax2.set_title("mAP sur validation", fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.suptitle("Courbes d'entraînement — SportInsight AI",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig6_training_curves.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()



if __name__ == "__main__":


    # Charger le modèle sauvegardé 
    print("[1/6] Chargement du modèle...")
    model, scaler, label_to_idx, idx_to_label = load_model_from_person2("model.pkl")

    #Charger les données de test 
    print("[2/6] Chargement des données de test...")
    X_test, y_test = load_test_data("test_data.pkl")

    # Générer les prédictions 
    print("[3/6] Génération des prédictions...")
    y_pred, y_pred_labels = get_predictions(model, scaler, X_test, idx_to_label)
    y_true_labels = idx_to_class_names(y_test, idx_to_label)

    # Métriques
    print("[4/6] Calcul des métriques...")
    metrics = compute_metrics(y_true_labels, y_pred_labels)

    # Hyperparamètres 
    print("[5/6] Grille d'hyperparamètres...")
    param_grid, all_configs = hyperparameter_grid()

    # Figures 
    print("[6/6] Génération des figures...")
    plot_hyperparam_impact()
    plot_threshold_impact(model, scaler, X_test, y_test, idx_to_label)
    plot_map_per_class(y_true_labels, y_pred_labels)
    plot_confusion_matrix(y_true_labels, y_pred_labels)
    plot_metrics_table(metrics)
    plot_training_curves()


