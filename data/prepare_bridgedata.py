"""
BridgeData V2 preprocessing pipeline.

Edit the Config dataclass at the top to change any setting, then run:
    python data/prepare_bridgedata.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import random
from io import BytesIO
from pathlib import Path

import datasets.config
import fsspec
import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import Video, load_dataset
from PIL import Image
from tqdm import tqdm

from cosmos.datasets.video_dataset import make_dataloaders

from dataclasses import dataclass


@dataclass
class Config:
    out_dir:          Path = Path("local_storage/dataset/bridgedata_clips")
    num_trajectories: int  = 500
    batch_size:       int  = 8
    num_workers:      int  = 4
    visualize_only:   bool = False

    # clip params
    target_h:    int   = 64
    target_w:    int   = 64
    clip_len:    int   = 8
    target_fps:  float = 6.0

    # quality filters
    static_thresh: float = 0.02   # mean abs diff below this → static clip
    shot_thresh:   float = 0.30   # mean abs diff above this → shot change
    val_fraction:  float = 0.05


CFG = Config()

# ── torchcodec workaround ─────────────────────────────────────────────────────
# torchcodec is installed but missing FFmpeg shared libs; disable it so
# datasets falls back to its plain-bytes Video path.
datasets.config.TORCHCODEC_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_raw_video(video_field: dict) -> bytes:
    raw = video_field.get("bytes")
    if not raw:
        with fsspec.open(video_field["path"], "rb") as f:
            raw = f.read()
    return raw


def decode_frames(raw: bytes) -> list[np.ndarray]:
    import decord
    vr = decord.VideoReader(BytesIO(raw))
    step = max(1, round(vr.get_avg_fps() / CFG.target_fps))
    frames = vr.get_batch(list(range(0, len(vr), step))).asnumpy()  # [N, H, W, 3]
    return [frames[i] for i in range(len(frames))]


def resize_and_normalize(frame: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 → float32 3xHxW in [-1, 1]."""
    img = TF.resize(Image.fromarray(frame), [CFG.target_h, CFG.target_w],
                    interpolation=TF.InterpolationMode.BILINEAR)
    return TF.to_tensor(img) * 2.0 - 1.0


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0))


def slice_into_clips(frames: list[np.ndarray]) -> list[list[np.ndarray]]:
    n = CFG.clip_len
    return [frames[s : s + n] for s in range(0, len(frames) - n + 1, n)]


def is_bad_clip(frames: list[np.ndarray]) -> bool:
    if len(frames) < CFG.clip_len:
        return True
    diffs = [frame_diff(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    if np.mean(diffs) < CFG.static_thresh:
        return True
    if any(d > CFG.shot_thresh for d in diffs):
        return True
    return False


def clip_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """List of T HxWx3 frames → [T, 3, H, W] float32 in [-1, 1]."""
    return torch.stack([resize_and_normalize(f) for f in frames])


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


# ── main processing ───────────────────────────────────────────────────────────

def process(seed: int = 42):
    random.seed(seed)
    out_dir   = CFG.out_dir
    train_dir = out_dir / "train"
    val_dir   = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "VyoJ/BridgeData-V2-Scripted-Videos",
        streaming=True,
    )["train"].cast_column("video", Video(decode=False))

    n_ok = n_bad = n_corrupt = 0

    for i, sample in enumerate(tqdm(ds, total=CFG.num_trajectories, desc="trajectories")):
        if i >= CFG.num_trajectories:
            break

        traj_name = sample.get("trajectory_name", f"traj_{i:06d}")

        try:
            raw    = _load_raw_video(sample["video"])
            frames = decode_frames(raw)
        except Exception as e:
            tqdm.write(f"[skip] {traj_name}: decode error — {e}")
            n_corrupt += 1
            continue

        for clip_idx, clip_frames in enumerate(slice_into_clips(frames)):
            if is_bad_clip(clip_frames):
                n_bad += 1
                continue

            tensor    = clip_to_tensor(clip_frames)   # [T, 3, H, W]
            split_dir = val_dir if (stable_hash(traj_name) % 100) < int(CFG.val_fraction * 100) else train_dir
            torch.save(tensor, split_dir / f"{traj_name}_clip{clip_idx:03d}.pt")
            n_ok += 1

    train_files = sorted(train_dir.glob("*.pt"))
    val_files   = sorted(val_dir.glob("*.pt"))
    (out_dir / "train.txt").write_text("\n".join(str(f) for f in train_files))
    (out_dir / "val.txt").write_text("\n".join(str(f) for f in val_files))

    print(f"\ndone — saved {n_ok} clips  |  dropped {n_bad} bad  |  {n_corrupt} corrupt")
    print(f"  train: {len(train_files)}  val: {len(val_files)}")
    print(f"  manifests: {out_dir}/train.txt  {out_dir}/val.txt")


# ── batch visualizer ──────────────────────────────────────────────────────────

def visualize_batch(batch: torch.Tensor, max_rows: int = 4):
    """batch: [B, T, 3, H, W] in [-1, 1]. Shows first frame of each clip."""
    import matplotlib.pyplot as plt

    B = min(batch.size(0), max_rows)
    T_show = min(batch.size(1), 8)
    fig, axes = plt.subplots(B, T_show, figsize=(T_show * 2, B * 2))
    axes = np.array(axes).reshape(B, T_show)

    for b in range(B):
        for t in range(T_show):
            frame = batch[b, t]                        # [3, H, W]
            frame = (frame * 0.5 + 0.5).clamp(0, 1)   # [-1,1] → [0,1]
            axes[b, t].imshow(frame.permute(1, 2, 0).numpy())
            axes[b, t].axis("off")
            if b == 0:
                axes[b, t].set_title(f"t={t}")

    plt.suptitle(f"Batch shape: {tuple(batch.shape)}")
    plt.tight_layout()
    plt.savefig("batch_preview.png", dpi=120)
    plt.show()
    print("saved batch_preview.png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not CFG.visualize_only:
        process()

    train_dl, val_dl = make_dataloaders(
        CFG.out_dir / "train.txt", CFG.out_dir / "val.txt",
        batch_size=CFG.batch_size, num_workers=CFG.num_workers,
    )
    batch = next(iter(train_dl))
    print(f"batch shape: {batch.shape}")   # [B, T, 3, H, W]
    visualize_batch(batch)
