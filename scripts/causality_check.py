"""
Causality sanity check for TokenizerV1.

Procedure:
  1. Take one clip x of shape [1, 3, T, H, W]
  2. Copy it to x2
  3. Corrupt frames after time t in x2 with random noise
  4. Compare z[:, :, :t+1] from encode(x) vs encode(x2)

Expected: latents for times <= t must be identical (or within float tolerance).
If they differ, the encoder is leaking future information.

Run:
    python scripts/causality_check.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dataclasses import dataclass
from cosmos.datasets.video_dataset import VideoDataset
from cosmos.tokenizers.tokenizer_v1 import TokenizerV1


@dataclass
class Config:
    manifest:  Path = Path("local_storage/dataset/bridgedata_clips/train.txt")
    ckpt:      Path | None = None   # set to a checkpoint path to test a trained model
    device:    str  = "cuda" if torch.cuda.is_available() else "cpu"
    split_t:   int  = 3             # corrupt frames after this index (0-based)
    threshold: float = 1e-5         # max allowed diff in past latents


CFG = Config()


def main():
    # ── model ─────────────────────────────────────────────────────────────────
    model = TokenizerV1().to(CFG.device)
    if CFG.ckpt is not None:
        ckpt = torch.load(CFG.ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"loaded checkpoint: {CFG.ckpt}")
    model.eval()

    # ── one clip ──────────────────────────────────────────────────────────────
    ds = VideoDataset(CFG.manifest)
    # [T, 3, H, W] → [1, 3, T, H, W]
    x = ds[0].permute(1, 0, 2, 3).unsqueeze(0).to(CFG.device)
    T = x.size(2)
    print(f"clip shape : {tuple(x.shape)}")
    print(f"split at t : {CFG.split_t}  (corrupt frames {CFG.split_t+1}..{T-1})")

    # ── corrupt future frames ─────────────────────────────────────────────────
    x2 = x.clone()
    x2[:, :, CFG.split_t + 1:] = torch.rand_like(x2[:, :, CFG.split_t + 1:]) * 2 - 1

    # ── encode both ───────────────────────────────────────────────────────────
    with torch.no_grad():
        z1 = model.encode(x)    # [1, 16, T, H/4, W/4]
        z2 = model.encode(x2)

    print(f"latent shape: {tuple(z1.shape)}")

    # ── compare latents up to and including split_t ───────────────────────────
    z1_past = z1[:, :, :CFG.split_t + 1]
    z2_past = z2[:, :, :CFG.split_t + 1]

    max_diff  = (z1_past - z2_past).abs().max().item()
    mean_diff = (z1_past - z2_past).abs().mean().item()

    print(f"\npast latents (t <= {CFG.split_t})")
    print(f"  max  |z1 - z2| = {max_diff:.2e}")
    print(f"  mean |z1 - z2| = {mean_diff:.2e}")

    # ── also check future latents changed (sanity) ───────────────────────────
    z1_future = z1[:, :, CFG.split_t + 1:]
    z2_future = z2[:, :, CFG.split_t + 1:]
    future_diff = (z1_future - z2_future).abs().mean().item()
    print(f"\nfuture latents (t > {CFG.split_t})")
    print(f"  mean |z1 - z2| = {future_diff:.2e}  (should be > 0)")

    # ── verdict ───────────────────────────────────────────────────────────────
    if max_diff < CFG.threshold:
        print(f"\n✓  PASSED — past latents unchanged (max diff {max_diff:.2e} < {CFG.threshold})")
    else:
        print(f"\n✗  FAILED — past latents differ (max diff {max_diff:.2e} >= {CFG.threshold})")
        print("   The encoder is leaking future information.")

    assert future_diff > 0, "future latents should differ after corruption"
    print("✓  future latents correctly changed after corruption")


if __name__ == "__main__":
    main()
