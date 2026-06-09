from __future__ import annotations

import logging
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from api.config import ARTIFACTS_DIR, DATA_DIR, FPS

logger = logging.getLogger(__name__)
_BRIDGE_PATH = ARTIFACTS_DIR / "resnet_to_pca512.pkl"
_BRIDGE = None


def _read_opencv(video_path: Path, fps: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(native / fps)), 1)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is not None and frame.size and i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def _read_ffmpeg(
    video_path: Path,
    fps: float,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> list[np.ndarray]:
    if not shutil.which("ffmpeg"):
        return []
    frames = []
    with tempfile.TemporaryDirectory(prefix="si_") as tmp:
        pattern = str(Path(tmp) / "f_%06d.jpg")
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if start_sec > 0:
            cmd.extend(["-ss", str(start_sec)])
        cmd.extend(["-i", str(video_path)])
        if duration_sec is not None and duration_sec > 0:
            cmd.extend(["-t", str(duration_sec)])
        cmd.extend(["-vf", f"fps={fps}", pattern])
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.warning("ffmpeg: %s", r.stderr)
            return []
        for p in sorted(Path(tmp).glob("f_*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def read_video_frames(
    video_path: Path,
    fps: float = FPS,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> list[np.ndarray]:
    video_path = Path(video_path)
    if start_sec > 0 or duration_sec is not None:
        frames = _read_ffmpeg(video_path, fps, start_sec=start_sec, duration_sec=duration_sec)
    else:
        frames = _read_opencv(video_path, fps) or _read_ffmpeg(video_path, fps)
    if not frames:
        raise ValueError(f"Impossible de lire {video_path.name} (essayez MP4 H.264)")
    return frames


def _resnet2048(frames: list[np.ndarray]) -> np.ndarray:
    import torch
    from torchvision import models, transforms

    w = models.ResNet152_Weights.IMAGENET1K_V1
    net = models.resnet152(weights=w)
    net.fc = torch.nn.Identity()
    net.eval()
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    chunks = []
    with torch.no_grad():
        for i in range(0, len(frames), 16):
            b = torch.stack([tfm(f) for f in frames[i : i + 16]])
            chunks.append(net(b).cpu().numpy())
    return np.vstack(chunks)


def _to_pca512(raw: np.ndarray) -> np.ndarray:
    global _BRIDGE
    if _BRIDGE_PATH.exists():
        if _BRIDGE is None:
            with open(_BRIDGE_PATH, "rb") as f:
                _BRIDGE = pickle.load(f)
        return _BRIDGE.predict(raw).astype(np.float32)
    from sklearn.decomposition import PCA

    if raw.shape[0] >= 512:
        return PCA(512, random_state=42).fit_transform(raw).astype(np.float32)
    pad = np.zeros((512 - raw.shape[0], raw.shape[1]))
    pca = PCA(512, random_state=42)
    pca.fit(np.vstack([raw, pad]))
    return pca.transform(raw).astype(np.float32)


def extract_pca512_from_video(
    video_path: Path,
    fps: float = FPS,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> np.ndarray:
    frames = read_video_frames(
        video_path,
        fps,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    return _to_pca512(_resnet2048(frames))


# Découpe : voir video_segments.resolve_segments
