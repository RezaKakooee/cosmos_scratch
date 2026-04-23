"""
Visualization utilities for tokenizer v1.

All functions accept tensors in [-1, 1] with shape [B, C, T, H, W]
or [C, T, H, W] (single clip).
"""

from pathlib import Path

import torch
import numpy as np
from PIL import Image


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float in [-1,1] → HxWx3 uint8."""
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def save_frame_grid(
    x: torch.Tensor,
    path: str | Path,
    label: str = "",
) -> None:
    """
    Save one clip as a horizontal strip of frames.

    x: [C, T, H, W] or [1, C, T, H, W]
    """
    if x.dim() == 5:
        x = x[0]
    C, T, H, W = x.shape
    frames = [_to_uint8(x[:, t]) for t in range(T)]
    grid = np.concatenate(frames, axis=1)   # [H, T*W, 3]
    img = Image.fromarray(grid)
    if label:
        from PIL import ImageDraw
        ImageDraw.Draw(img).text((4, 4), label, fill=(255, 255, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_recon_grid(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    path: str | Path,
    max_clips: int = 4,
) -> None:
    """
    Save a side-by-side comparison grid.

    Layout: rows alternate  [input row] [recon row]  for each clip.

    x, x_hat: [B, C, T, H, W]
    """
    B = min(x.size(0), max_clips)
    rows = []
    for b in range(B):
        input_row = np.concatenate([_to_uint8(x[b, :, t])     for t in range(x.size(2))], axis=1)
        recon_row = np.concatenate([_to_uint8(x_hat[b, :, t]) for t in range(x.size(2))], axis=1)
        rows.extend([input_row, recon_row])

    grid = np.concatenate(rows, axis=0)   # stack rows vertically
    img = Image.fromarray(grid)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"saved recon grid → {path}")


def save_gif(
    x: torch.Tensor,
    path: str | Path,
    fps: int = 6,
) -> None:
    """
    Save a single clip as an animated GIF.

    x: [C, T, H, W] or [1, C, T, H, W]
    """
    if x.dim() == 5:
        x = x[0]
    frames = [Image.fromarray(_to_uint8(x[:, t])) for t in range(x.size(1))]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"saved gif → {path}")


def save_recon_gif(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    path: str | Path,
    fps: int = 6,
) -> None:
    """
    Save a side-by-side [input | reconstruction] GIF for one clip.

    x, x_hat: [C, T, H, W] or [1, C, T, H, W]
    """
    if x.dim() == 5:
        x, x_hat = x[0], x_hat[0]
    T = x.size(1)
    frames = []
    for t in range(T):
        left  = _to_uint8(x[:, t])
        right = _to_uint8(x_hat[:, t])
        frames.append(Image.fromarray(np.concatenate([left, right], axis=1)))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"saved recon gif → {path}")
