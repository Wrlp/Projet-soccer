from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


@dataclass(frozen=True)
class VideoSample:
    path: str
    label: int
    class_name: str


def discover_samples(
    root_dir: str | Path,
    max_per_folder: int = 300,
) -> tuple[list[VideoSample], list[str]]:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    class_names = sorted([entry.name for entry in root_path.iterdir() if entry.is_dir()])
    samples: list[VideoSample] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = root_path / class_name
        video_paths = [
            video_path
            for video_path in sorted(class_dir.iterdir())
            if video_path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if max_per_folder > 0:
            video_paths = video_paths[:max_per_folder]
        for video_path in video_paths:
            samples.append(VideoSample(str(video_path), class_index, class_name))

    if not samples:
        raise RuntimeError(f"No video clips found under {root_path}")

    return samples, class_names


def split_samples(
    samples: list[VideoSample],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[VideoSample], list[VideoSample]]:
    labels = [sample.label for sample in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return list(train_samples), list(val_samples)


def load_video_clip(video_path: str | Path, num_frames: int = 16, image_size: int = 224) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")

    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
        selected_frames = [frames[index] for index in indices]
    else:
        selected_frames = frames + [frames[-1]] * (num_frames - len(frames))

    return np.stack(selected_frames, axis=0)


def compute_class_weights(samples: list[VideoSample], num_classes: int) -> torch.Tensor:
    counts = np.bincount([sample.label for sample in samples], minlength=num_classes)
    safe_counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * safe_counts)
    weights = weights.astype(np.float32)
    weights[counts == 0] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


class SoccerVideoDataset(Dataset):
    def __init__(self, samples: list[VideoSample], num_frames: int = 16, image_size: int = 224):
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        video = load_video_clip(sample.path, num_frames=self.num_frames, image_size=self.image_size)
        return video, sample.label, sample.class_name, sample.path
