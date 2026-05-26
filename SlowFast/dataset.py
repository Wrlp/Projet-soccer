import os
import json
import random
import numpy as np
import subprocess
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def list_video_files(root_dir):
    root = Path(root_dir)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    items = []
    for idx, cls in enumerate(classes):
        cls_dir = root / cls
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                items.append({'path': str(p), 'label': idx, 'class_name': cls})
    return items, classes


class SoccerClipDataset(Dataset):
    """
    Dataset that returns Slow / Fast tensors for each clip.

    Returns:
      slow: Tensor (C, T_s, H, W)
      fast: Tensor (C, T_f, H, W)
      label: int
    """

    def __init__(self, root_dir='SOCCER/outputs/clips/mp4', split_list=None,
                 T_s=8, alpha=8, size=224, transforms=None, use_npy=False,
                 npy_root='SOCCER/outputs/clips/npy'):
        self.root_dir = Path(root_dir)
        self.T_s = T_s
        self.alpha = alpha
        self.T_f = T_s * alpha
        self.size = size
        self.use_npy = use_npy
        self.npy_root = Path(npy_root)

        if split_list is None:
            items, classes = list_video_files(self.root_dir)
        else:
            items = []
            classes = set()
            for entry in split_list:
                items.append({'path': entry['path'], 'label': entry['label'], 'class_name': entry.get('class_name','')})
                classes.add(entry.get('class_name',''))
            classes = sorted([c for c in classes if c])

        self.items = items
        self.classes = classes

        # default transforms applied per-frame (on PIL or tensor images)
        if transforms is None:
            self.transforms = T.Compose([
                T.ConvertImageDtype(torch.float),
                T.Resize((self.size, self.size)),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def _load_video(self, path):
        """Load a clip with ffmpeg and return a tensor of shape (T, C, H, W).

        We avoid torchvision video backends because the available build may not
        include `read_video`. Frames are sampled at a fixed rate and resized to
        the target input size to keep decoding simple and dependency-free.
        """
        ffmpeg_cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-vf", f"fps=16,scale={self.size}:{self.size}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        raw = proc.stdout
        if not raw:
            raise RuntimeError(f'Empty video: {path}')

        frame_size = self.size * self.size * 3
        n_frames = len(raw) // frame_size
        if n_frames <= 0:
            raise RuntimeError(f'Could not decode frames from video: {path}')

        arr = np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8)
        arr = arr.reshape(n_frames, self.size, self.size, 3).copy()
        v = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return v

    def _load_npy(self, path_npy):
        arr = np.load(path_npy)
        # assume arr shape (T, H, W, C) or (T, C, H, W) or features
        if arr.ndim == 4 and arr.shape[-1] == 3:
            # (T, H, W, C)
            arr = torch.from_numpy(arr).permute(0, 3, 1, 2)
        else:
            # features: convert to fake frames by repeating
            arr = torch.from_numpy(arr)
            arr = arr.unsqueeze(1).unsqueeze(1)  # (T,1,1,D)
        return arr

    def _temporal_sample(self, frames, T_needed):
        T_total = frames.shape[0]
        if T_total == 0:
            raise RuntimeError('Video has no frames')
        if T_total >= T_needed:
            indices = np.linspace(0, T_total - 1, num=T_needed, dtype=int)
        else:
            # pad by repeating last frame
            indices = list(range(T_total)) + [T_total - 1] * (T_needed - T_total)
            indices = np.array(indices, dtype=int)
        sampled = frames[indices]
        return sampled

    def _process_frames(self, frames):
        # frames: Tensor (T, C, H, W)
        # apply transforms per frame and return (C, T, H, W)
        proc = []
        for f in frames:
            # f: (C,H,W)
            img = f
            img = self.transforms(img)
            proc.append(img)
        proc = torch.stack(proc, dim=1)  # (C, T, H, W)
        return proc

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item['path']
        label = int(item['label'])

        if self.use_npy:
            # attempt to find corresponding npy
            p = Path(path)
            cls = p.parent.name
            name = p.stem + '.npy'
            npy_path = self.npy_root / cls / name
            frames = self._load_npy(str(npy_path))
        else:
            frames = self._load_video(path)  # (T, C, H, W)

        # sample fast and slow
        fast = self._temporal_sample(frames, self.T_f)
        slow = fast[::self.alpha][:self.T_s]

        fast_proc = self._process_frames(fast)
        slow_proc = self._process_frames(slow)

        return slow_proc, fast_proc, label


if __name__ == '__main__':
    # quick smoke test listing
    items, classes = list_video_files('SOCCER/outputs/clips/mp4')
    print('Found', len(items), 'clips across', len(classes), 'classes')
