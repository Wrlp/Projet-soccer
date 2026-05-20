import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, average_precision_score
)
from sklearn.model_selection import ParameterGrid
import json
import os

#config

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



#simulate data loading
def load_predictions(filepath=None):
    """
    Charge les prédictions du modèle (personne 2).
    Format attendu : liste de dicts avec 'timestamp', 'class', 'confidence'
    
    Si filepath=None → données simulées pour tester le code.
    """
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    # ── Données simulées ──
    print("[INFO] Utilisation de données simulées. Remplace par les vraies prédictions.")
    np.random.seed(42)
    n = 60
    predictions = []
    ground_truth = []

    for i in range(n):
        cls = np.random.choice(EVENT_CLASSES)
        t = np.random.uniform(0, 90)
        predictions.append({"timestamp": round(t, 2), "class": cls, "confidence": round(np.random.uniform(0.5, 1.0), 3)})
        # Vérité terrain : parfois le même événement, parfois différent
        gt_cls = cls if np.random.rand() > 0.3 else np.random.choice(EVENT_CLASSES)
        ground_truth.append({"timestamp": round(t + np.random.uniform(-2, 2), 2), "class": gt_cls})

    return predictions, ground_truth


# Hyperparams

def hyperparameter_grid():
    """
    Définit et affiche la grille d'hyperparamètres à tester.
    La personne 2 utilise ces valeurs pour entraîner le modèle.
    """
    param_grid = {
        "learning_rate":    [1e-4, 1e-3, 5e-3],
        "batch_size":       [8, 16, 32],
        "num_epochs":       [10, 20, 50],
        "window_size_sec":  [5, 10, 15],    # fenêtre temporelle autour de l'événement
        "threshold":        [0.3, 0.5, 0.7], # seuil de détection
    }

    all_configs = list(ParameterGrid(param_grid))
    print(f"[INFO] Nombre total de configurations : {len(all_configs)}")
    print(f"[INFO] Exemple de config #1 : {all_configs[0]}")
    print(f"[INFO] Exemple de config #2 : {all_configs[len(all_configs)//2]}")

    return param_grid, all_configs


def plot_hyperparam_impact(save=True):
    """
    Figure 1 — Impact des hyperparamètres sur la mAP.
    Montre comment le learning rate et le batch size influencent les résultats.
    """
    learning_rates = [1e-4, 1e-3, 5e-3]
    batch_sizes = [8, 16, 32]

    # Résultats simulés (mAP en %) — à remplacer par tes vraies valeurs
    map_scores = np.array([
        [42.1, 47.3, 44.8],  # lr=1e-4, batch 8/16/32
        [51.2, 58.6, 55.1],  # lr=1e-3
        [48.7, 53.2, 50.9],  # lr=5e-3
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(batch_sizes))
    width = 0.25

    for i, lr in enumerate(learning_rates):
        bars = ax.bar(x + i * width, map_scores[i], width,
                      label=f"lr = {lr}", color=list(COLORS.values())[i], alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3)

    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("mAP (%)", fontsize=12)
    ax.set_title("Impact du learning rate et du batch size sur la mAP", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"batch={b}" for b in batch_sizes])
    ax.legend(title="Learning rate")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig1_hyperparam_impact.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_threshold_impact(save=True):
    """
    Figure 2 — Impact du seuil de détection sur précision vs rappel.
    Courbe classique précision/rappel selon le threshold.
    """
    thresholds = np.linspace(0.1, 0.9, 50)
    # Simulé — à remplacer par tes vraies valeurs
    precision = 0.4 + 0.55 * thresholds + np.random.normal(0, 0.02, len(thresholds))
    recall    = 0.95 - 0.75 * thresholds + np.random.normal(0, 0.02, len(thresholds))
    precision = np.clip(precision, 0, 1)
    recall    = np.clip(recall, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision, color=COLORS["primary"], linewidth=2, label="Précision")
    ax.plot(thresholds, recall,    color=COLORS["card"],    linewidth=2, label="Rappel")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Seuil recommandé (0.5)")
    ax.fill_between(thresholds, precision, recall, alpha=0.08, color=COLORS["primary"])

    ax.set_xlabel("Seuil de détection (threshold)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Précision vs Rappel selon le seuil de détection", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig2_threshold_precision_recall.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


# Evaluation

def compute_metrics(predictions, ground_truth):
    """
    Calcule précision, rappel, F1 et mAP approximative par classe.
    
    predictions  : liste de dicts {"timestamp", "class", "confidence"}
    ground_truth : liste de dicts {"timestamp", "class"}
    """
    y_pred = [p["class"] for p in predictions]
    y_true = [g["class"] for g in ground_truth]

    # Aligne les longueurs si nécessaire
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=EVENT_CLASSES)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=EVENT_CLASSES)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=EVENT_CLASSES)

    print("\n" + "="*45)
    print("   MÉTRIQUES D'ÉVALUATION — SportInsight AI")
    print("="*45)
    print(f"  Précision macro  : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Rappel macro     : {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  F1-score macro   : {f1:.4f}  ({f1*100:.1f}%)")
    print("="*45 + "\n")

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_temporal_error(predictions, ground_truth, tolerance_sec=5):
    """
    Calcule l'erreur temporelle moyenne entre prédictions et vérité terrain.
    Un événement est considéré correct si dans ±tolerance_sec secondes.
    """
    errors = []
    for pred in predictions:
        same_class = [g for g in ground_truth if g["class"] == pred["class"]]
        if same_class:
            closest = min(same_class, key=lambda g: abs(g["timestamp"] - pred["timestamp"]))
            errors.append(abs(closest["timestamp"] - pred["timestamp"]))

    mean_err = np.mean(errors) if errors else 0
    within_tolerance = sum(1 for e in errors if e <= tolerance_sec) / len(errors) if errors else 0

    print(f"  Erreur temporelle moyenne : {mean_err:.2f} secondes")
    print(f"  Événements dans ±{tolerance_sec}s     : {within_tolerance*100:.1f}%\n")

    return mean_err, within_tolerance


#Figure

def plot_map_per_class(save=True):
    """
    Figure 3 — mAP par classe d'événement.
    """
    # Simulé — remplace par tes vraies valeurs
    map_per_class = {
        "But":           72.4,
        "Carton jaune":  55.1,
        "Carton rouge":  61.3,
        "Remplacement":  48.7,
        "Tir":           43.2,
        "Corner":        38.9,
    }

    classes = list(map_per_class.keys())
    values  = list(map_per_class.values())
    colors  = [COLORS["goal"], COLORS["card"], "#a32d2d",
               COLORS["sub"], COLORS["shot"], COLORS["corner"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(classes, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=6, fontsize=10)
    ax.axvline(x=np.mean(values), color="gray", linestyle="--",
               label=f"mAP moyenne : {np.mean(values):.1f}%")

    ax.set_xlabel("mAP (%)", fontsize=12)
    ax.set_title("mAP par classe d'événement", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig3_map_per_class.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_confusion_matrix(predictions, ground_truth, save=True):
    """
    Figure 4 — Matrice de confusion par classe.
    """
    y_pred = [p["class"] for p in predictions]
    y_true = [g["class"] for g in ground_truth]
    min_len = min(len(y_pred), len(y_true))

    cm = confusion_matrix(y_true[:min_len], y_pred[:min_len], labels=EVENT_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EVENT_CLASSES)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
    ax.set_title("Matrice de confusion — Détection d'événements", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig4_confusion_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_training_curves(save=True):
    """
    Figure 5 — Courbes d'entraînement (loss et mAP au fil des epochs).
    À remplacer par les vraies courbes de la personne 2.
    """
    epochs = np.arange(1, 51)
    # Simulé
    train_loss = 1.8 * np.exp(-0.07 * epochs) + np.random.normal(0, 0.03, len(epochs))
    val_loss   = 1.9 * np.exp(-0.06 * epochs) + np.random.normal(0, 0.04, len(epochs)) + 0.1
    val_map    = 60 * (1 - np.exp(-0.08 * epochs)) + np.random.normal(0, 1, len(epochs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, color=COLORS["primary"], label="Train loss", linewidth=2)
    ax1.plot(epochs, val_loss,   color=COLORS["card"],    label="Val loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Courbe de loss", fontweight="bold")
    ax1.legend()
    ax1.grid(linestyle="--", alpha=0.4)

    # mAP
    ax2.plot(epochs, val_map, color=COLORS["sub"], linewidth=2)
    ax2.fill_between(epochs, val_map, alpha=0.15, color=COLORS["sub"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP (%)")
    ax2.set_title("mAP sur validation", fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.suptitle("Courbes d'entraînement — SportInsight AI", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig5_training_curves.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()


def plot_timeline_example(predictions, save=True):
    """
    Figure 6 — Timeline visuelle des événements détectés vs vérité terrain.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

    color_map = {
        "But":           COLORS["goal"],
        "Carton jaune":  COLORS["card"],
        "Carton rouge":  "#a32d2d",
        "Remplacement":  COLORS["sub"],
        "Tir":           COLORS["shot"],
        "Corner":        COLORS["corner"],
    }

    # Prédictions
    for p in predictions[:20]:
        c = color_map.get(p["class"], "gray")
        ax1.scatter(p["timestamp"], 0.5, color=c, s=80, zorder=3)
    ax1.set_yticks([0.5])
    ax1.set_yticklabels(["Prédictions"])
    ax1.axhline(0.5, color="lightgray", linewidth=0.8)
    ax1.set_xlim(0, 90)

    # Vérité terrain (simulée)
    gt_sample = [{"timestamp": np.random.uniform(0, 90), "class": np.random.choice(EVENT_CLASSES)} for _ in range(15)]
    for g in gt_sample:
        c = color_map.get(g["class"], "gray")
        ax2.scatter(g["timestamp"], 0.5, color=c, s=80, marker="D", zorder=3)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["Vérité terrain"])
    ax2.axhline(0.5, color="lightgray", linewidth=0.8)
    ax2.set_xlabel("Minute de match")

    # Légende
    legend_patches = [mpatches.Patch(color=c, label=cls) for cls, c in color_map.items()]
    fig.legend(handles=legend_patches, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("Timeline — Prédictions vs Vérité terrain", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = f"{OUTPUT_DIR}/fig6_timeline_comparison.png"
        fig.savefig(path, dpi=150)
        print(f"[SAVED] {path}")
    plt.show()




if __name__ == "__main__":

    print("\n=== SportInsight AI — Évaluation & Figures ===\n")

    # 1. Hyperparamètres
    print("[1/6] Grille d'hyperparamètres...")
    param_grid, all_configs = hyperparameter_grid()

    # 2. Charger les données
    #    Quand la personne 2 te donne son fichier JSON :
    #    predictions, ground_truth = load_predictions("predictions.json")
    print("\n[2/6] Chargement des prédictions...")
    predictions, ground_truth = load_predictions()

    # 3. Métriques
    print("[3/6] Calcul des métriques...")
    metrics = compute_metrics(predictions, ground_truth)
    mean_err, within_tol = compute_temporal_error(predictions, ground_truth)

    # 4. Figures
    print("[4/6] Génération des figures...\n")
    plot_hyperparam_impact()
    plot_threshold_impact()
    plot_map_per_class()
    plot_confusion_matrix(predictions, ground_truth)
    plot_training_curves()
    plot_timeline_example(predictions)

    print(f"\nTerminé ! Toutes les figures sont dans le dossier '{OUTPUT_DIR}/'")
    print("   Remplace les données simulées par les vraies prédictions de la personne 2.")