import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

import torch
from torch.utils.data import DataLoader

# Chemins principaux
DATA_ROOT      = Path(r"/mnt/e/SOCCER")
MODEL_DIR      = Path("outputs/models/videomae_soccernet_112/best_model")
OUTPUT_DIR     = Path("outputs/figures/videomae_soccernet_112")
LABELS_FILE    = MODEL_DIR / "labels.json"
METRICS_FILE   = Path("outputs/figures/videomae_soccernet_112/metrics.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "primary":   "#1a6fbc",
    "secondary": "#0f2d4a",
    "goal":      "#185fa5",
    "card":      "#854f0b",
    "sub":       "#3b6d11",
    "shot":      "#993556",
    "corner":    "#3c3489",
}


# Chargement du modele et des labels depuis le dossier best_model

def load_model_and_labels():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Dossier modele introuvable : {MODEL_DIR}\n"
            "Assurez-vous que l'entrainement a ete complete et que best_model/ existe."
        )

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    label_to_id = labels_data["label_to_id"]
    id_to_label = {int(k): v for k, v in labels_data["id_to_label"].items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]

    image_processor = VideoMAEImageProcessor.from_pretrained(str(MODEL_DIR))
    model = VideoMAEForVideoClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Modele charge depuis : {MODEL_DIR}")
    print(f"Classes : {class_names}")
    return model, image_processor, label_to_id, id_to_label, class_names, device


# Chargement des clips de validation depuis HDD

def load_validation_clips(class_names, label_to_id, num_frames=16, image_size=224):
    # Import local au projet (meme logique que train.py)
    from videomae_finetune.dataset import SoccerVideoDataset, discover_samples, split_samples
    from torch.utils.data import DataLoader

    clips_root = DATA_ROOT / "clips" / "mp4"
    if not clips_root.exists():
        raise FileNotFoundError(
            f"Dossier de clips introuvable : {clips_root}\n"
            "Verifiez que E:\\SOCCER\\outputs\\clips\\mp4\\ existe."
        )

    samples, _ = discover_samples(str(clips_root))
    _, val_samples = split_samples(samples, test_size=0.2, random_state=42)

    dataset = SoccerVideoDataset(val_samples, num_frames=num_frames, image_size=image_size)
    print(f"Echantillons de validation : {len(dataset)}")
    return dataset


def build_collate_fn(image_processor):
    def collate_fn(batch):
        videos, labels, class_names_batch, paths = zip(*batch)
        processed = image_processor(images=list(videos), return_tensors="pt")
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return processed["pixel_values"], labels_tensor, class_names_batch, paths
    return collate_fn


# Inference sur le jeu de validation

@torch.no_grad()
def run_inference(model, dataset, image_processor, device, batch_size=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=build_collate_fn(image_processor),
        pin_memory=torch.cuda.is_available(),
    )

    all_preds  = []
    all_labels = []

    for pixel_values, labels, _, _ in loader:
        pixel_values = pixel_values.to(device)
        outputs      = model(pixel_values=pixel_values)
        preds        = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    print(f"Inference terminee : {len(all_preds)} predictions")
    return np.array(all_labels), np.array(all_preds)


# Calcul et affichage des metriques globales

def compute_and_print_metrics(y_true, y_pred, class_names):
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("=" * 45)
    print("   METRIQUES — SportInsight AI (VideoMAE)")
    print("=" * 45)
    print(f"  Precision macro  : {precision * 100:.1f}%")
    print(f"  Rappel macro     : {recall * 100:.1f}%")
    print(f"  F1-score macro   : {f1 * 100:.1f}%")
    print("=" * 45)

    return {"precision": precision, "recall": recall, "f1": f1}


# Figure 1 : F1-score par classe

def plot_f1_per_class(y_true, y_pred, class_names, save=True):
    scores = {}
    for i, cls in enumerate(class_names):
        y_t = (y_true == i).astype(int)
        y_p = (y_pred == i).astype(int)
        scores[cls] = f1_score(y_t, y_p, zero_division=0) * 100

    classes = list(scores.keys())
    values  = list(scores.values())
    cmap    = plt.get_cmap("tab20")
    colors  = [cmap(i) for i in range(len(classes))]

    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.5)))
    bars = ax.barh(classes, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=6, fontsize=9)
    ax.axvline(
        x=np.mean(values),
        color="gray",
        linestyle="--",
        label=f"Moyenne : {np.mean(values):.1f}%",
    )
    ax.set_xlabel("F1-score (%)", fontsize=12)
    ax.set_title("F1-score par classe d'evenement", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig1_f1_per_class.png"
        fig.savefig(path, dpi=150)
        print(f"Figure sauvegardee : {path}")
    plt.show()


# Figure 2 : Matrice de confusion

def plot_confusion_matrix(y_true, y_pred, class_names, save=True):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(7, len(class_names) - 1)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(
        "Matrice de confusion — Detection d'evenements",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig2_confusion_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"Figure sauvegardee : {path}")
    plt.show()


# Figure 3 : Tableau recapitulatif des metriques

def plot_metrics_table(metrics, save=True):
    rows = [
        ["Precision macro", f"{metrics['precision'] * 100:.1f}%"],
        ["Rappel macro",    f"{metrics['recall'] * 100:.1f}%"],
        ["F1-score macro",  f"{metrics['f1'] * 100:.1f}%"],
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metrique", "Valeur"],
        cellLoc="center",
        loc="center",
        colColours=["#0f2d4a", "#0f2d4a"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.4, 2)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f6ff" if r % 2 == 0 else "white")

    ax.set_title(
        "Resultats d'evaluation — SportInsight AI",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig3_metrics_table.png"
        fig.savefig(path, dpi=150)
        print(f"Figure sauvegardee : {path}")
    plt.show()


# Figure 4 : Courbes d'entrainement depuis metrics.json

def plot_training_curves(save=True):
    if not METRICS_FILE.exists():
        print(f"Fichier metrics.json introuvable ({METRICS_FILE}), figure ignoree.")
        return

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        print("Historique vide dans metrics.json, figure ignoree.")
        return

    epochs      = [e["epoch"] for e in history]
    train_loss  = [e["train_loss"] for e in history]
    val_loss    = [e["val_loss"] for e in history]
    val_f1      = [e["val_macro_f1"] for e in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, color=COLORS["primary"], linewidth=2, label="Train loss")
    ax1.plot(epochs, val_loss,   color=COLORS["card"],    linewidth=2, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Courbe de loss", fontweight="bold")
    ax1.legend()
    ax1.grid(linestyle="--", alpha=0.4)

    ax2.plot(epochs, val_f1, color=COLORS["sub"], linewidth=2)
    ax2.fill_between(epochs, val_f1, alpha=0.15, color=COLORS["sub"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1 sur validation", fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.suptitle(
        "Courbes d'entrainement — SportInsight AI",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig4_training_curves.png"
        fig.savefig(path, dpi=150)
        print(f"Figure sauvegardee : {path}")
    plt.show()


# Figure 5 : Precision vs Rappel selon le seuil de confiance

def plot_threshold_impact(model, dataset, image_processor, device, save=True):
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=build_collate_fn(image_processor),
        pin_memory=torch.cuda.is_available(),
    )

    all_proba  = []
    all_labels = []

    with torch.no_grad():
        for pixel_values, labels, _, _ in loader:
            pixel_values = pixel_values.to(device)
            logits       = model(pixel_values=pixel_values).logits
            proba        = torch.softmax(logits, dim=-1).cpu().numpy()
            all_proba.extend(proba)
            all_labels.extend(labels.tolist())

    all_proba  = np.array(all_proba)
    all_labels = np.array(all_labels)
    n_classes  = all_proba.shape[1]
    thresholds = np.linspace(0.1, 0.9, 40)

    precisions = []
    recalls    = []

    for thresh in thresholds:
        y_pred = np.array([
            np.argmax(p) if np.max(p) >= thresh else -1
            for p in all_proba
        ])
        mask = y_pred != -1
        y_p  = y_pred[mask]
        y_t  = all_labels[mask]

        if len(y_p) == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            continue

        precisions.append(
            precision_score(y_t, y_p, average="macro", zero_division=0,
                            labels=list(range(n_classes)))
        )
        recalls.append(
            recall_score(y_t, y_p, average="macro", zero_division=0,
                         labels=list(range(n_classes)))
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, color=COLORS["primary"], linewidth=2, label="Precision")
    ax.plot(thresholds, recalls,    color=COLORS["card"],    linewidth=2, label="Rappel")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Seuil = 0.5")
    ax.set_xlabel("Seuil de confiance", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision vs Rappel selon le seuil", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig5_threshold_precision_recall.png"
        fig.savefig(path, dpi=150)
        print(f"Figure sauvegardee : {path}")
    plt.show()


# Point d'entree principal

def run_evaluation():
    print("Chargement du modele...")
    model, image_processor, label_to_id, id_to_label, class_names, device = load_model_and_labels()

    print("Chargement des clips de validation...")
    dataset = load_validation_clips(class_names, label_to_id)

    print("Inference en cours...")
    y_true, y_pred = run_inference(model, dataset, image_processor, device)

    print("Calcul des metriques...")
    metrics = compute_and_print_metrics(y_true, y_pred, class_names)

    print("Generation des figures...")
    plot_f1_per_class(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_metrics_table(metrics)
    plot_training_curves()
    plot_threshold_impact(model, dataset, image_processor, device)

    print("Evaluation terminee. Figures dans :", OUTPUT_DIR)
    return metrics


if __name__ == "__main__":
    run_evaluation()