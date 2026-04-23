import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    """
    Loads pre-processed clip tensors saved by data/prepare_bridgedata.py.

    Each .pt file contains a single clip: [T, 3, H, W] float32 in [-1, 1].
    A manifest txt file (train.txt / val.txt) lists one clip path per line,
    which makes it easy to reproduce splits without moving files around.
    """

    def __init__(self, manifest: str | Path):
        """
        Args:
            manifest: path to a txt file where each line is an absolute or
                      relative path to a .pt clip file.
        """
        manifest = Path(manifest)
        self.files = [
            Path(line.strip())
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        if not self.files:
            raise ValueError(f"No clips found in manifest {manifest}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # [T, 3, H, W]  float32  in [-1, 1]
        return torch.load(self.files[idx], weights_only=True)


def make_dataloaders(
    train_manifest: str | Path,
    val_manifest: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = VideoDataset(train_manifest)
    val_ds   = VideoDataset(val_manifest)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"train clips: {len(train_ds)}  val clips: {len(val_ds)}")
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = make_dataloaders(
        "local_storage/dataset/bridgedata_clips/train.txt",
        "local_storage/dataset/bridgedata_clips/val.txt",
        batch_size=4,
        num_workers=0,
    )

    batch = next(iter(train_dl))                          # [B, T, 3, H, W]
    B, T, C, H, W = batch.shape

    print(f"batch shape : {tuple(batch.shape)}")
    print(f"dtype       : {batch.dtype}")
    print(f"min / max   : {batch.min():.3f} / {batch.max():.3f}")

    assert batch.dtype == torch.float32,      "expected float32"
    assert C == 3,                            "expected 3 channels"
    assert batch.min() >= -1.0 - 1e-4,       "values below -1"
    assert batch.max() <=  1.0 + 1e-4,       "values above  1"
    assert batch.shape == (B, T, 3, H, W),   "unexpected shape"

    print("all assertions passed")
