from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    get_cosine_schedule_with_warmup,
)

from videomae_finetune.dataset import (
    SoccerVideoDataset,
    compute_class_weights,
    discover_samples,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE on the SOCCER clip dataset")
    parser.add_argument("--data-root", type=str, default="SOCCER/outputs/clips/mp4")
    parser.add_argument("--output-dir", type=str, default="outputs/models/videomae_soccernet")
    parser.add_argument("--model-name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-per-folder", type=int, default=300)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(image_processor: VideoMAEImageProcessor):
    def collate_fn(batch):
        videos, labels, class_names, paths = zip(*batch)
        processed = image_processor(images=list(videos), return_tensors="pt")
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return processed["pixel_values"], labels_tensor, class_names, paths

    return collate_fn


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for pixel_values, labels, _, _ in data_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        if outputs.loss is not None:
            total_loss += outputs.loss.item() * labels.size(0)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        total_examples += labels.size(0)

    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    mean_loss = total_loss / max(1, total_examples)
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    samples, class_names = discover_samples(args.data_root, max_per_folder=args.max_per_folder)
    train_samples, val_samples = split_samples(samples, test_size=args.test_size, random_state=args.seed)

    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]

    label_to_id = {class_name: index for index, class_name in enumerate(class_names)}
    id_to_label = {str(index): class_name for class_name, index in label_to_id.items()}

    image_processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    train_dataset = SoccerVideoDataset(train_samples, num_frames=args.num_frames, image_size=args.image_size)
    val_dataset = SoccerVideoDataset(val_samples, num_frames=args.num_frames, image_size=args.image_size)

    collate_fn = build_collate_fn(image_processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        args.model_name,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True,
    )
    model.config.num_frames = args.num_frames
    model.config.image_size = args.image_size
    model.config.label2id = label_to_id
    model.config.id2label = id_to_label

    if args.freeze_backbone:
        for parameter in model.videomae.parameters():
            parameter.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = compute_class_weights(train_samples, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    best_macro_f1 = -1.0
    best_metrics: dict[str, float] | None = None
    metrics_history: list[dict[str, float]] = []

    def save_artifacts(target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target_dir)
        image_processor.save_pretrained(target_dir)
        with open(target_dir / "labels.json", "w", encoding="utf-8") as handle:
            json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, handle, indent=2, ensure_ascii=False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0

        for pixel_values, labels, _, _ in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with autocast_context:
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)

            optimizer_stepped = True
            if use_amp:
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= previous_scale
            else:
                loss.backward()
                optimizer.step()

            if optimizer_stepped:
                scheduler.step()

            running_loss += loss.item() * labels.size(0)
            running_examples += labels.size(0)

        train_loss = running_loss / max(1, running_examples)
        val_metrics = evaluate(model, val_loader, device)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        metrics_history.append(epoch_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
            save_artifacts(best_model_dir)

        if epoch % 10 == 0:
            checkpoint_dir = checkpoints_dir / f"epoch_{epoch:03d}"
            save_artifacts(checkpoint_dir)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best": best_metrics,
                "history": metrics_history,
                "class_names": class_names,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Best model saved in: {best_model_dir}")
    print(f"Checkpoints saved in: {checkpoints_dir}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    train()
