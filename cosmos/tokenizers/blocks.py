import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos.tokenizers.causal_conv import CausalConv3d, FactorizedCausalConv3d, make_causal_conv


class ResBlock3D(nn.Module):
    """
    Causal residual block using factorized spatial+temporal convolutions.

    conv_type controls the convolution variant (see make_causal_conv):
      "factorized" — spatial pass then temporal pass (Day 2 default, fewer params)
      "full"       — joint 3D kernel  (Day 1 style, more expressive)

    Layout:
        GroupNorm → SiLU → CausalConv [factorized or full]
        GroupNorm → SiLU → CausalConv [factorized or full]
        + residual

    GroupNorm is applied per-frame (B*T treated as batch) so that future
    frames do not contaminate past-frame statistics — same causal-norm fix
    used in CausalSpatiotemporalAttention.

    Input / output: [B, C, T, H, W]  — channels, T, H, W all unchanged.
    """

    def __init__(self, channels: int, num_groups: int = 8, conv_type: str = "factorized"):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = make_causal_conv(channels, channels, kernel_size=3, conv_type=conv_type)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = make_causal_conv(channels, channels, kernel_size=3, conv_type=conv_type)

    def _norm(self, norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
        # Apply GroupNorm per-frame so future frames do not affect past-frame stats.
        # Merge B and T → [B*T, C, H, W], normalise, then split back.
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = norm(x)
        return x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self._norm(self.norm1, x)))
        x = self.conv2(F.silu(self._norm(self.norm2, x)))
        return residual + x


# ── Spatial down / up ─────────────────────────────────────────────────────────

class DownsampleSpatial(nn.Module):
    """
    Halves H and W using a strided factorized causal conv. T is unchanged.

    Stride (1, 2, 2) in the spatial pass of FactorizedCausalConv3d skips every
    other pixel in H and W, halving spatial resolution without pooling.

    [B, C_in, T, H,   W  ] → [B, C_out, T, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Spatial conv with stride 2 in H and W; temporal conv keeps stride 1.
        # We build this directly with CausalConv3d using stride (1,2,2) since
        # FactorizedCausalConv3d does not expose per-axis stride — strided
        # downsampling is already a simple single-layer operation.
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleSpatial(nn.Module):
    """
    Doubles H and W. T is unchanged.

    Nearest-neighbour interpolation + causal conv:
      - Interpolation avoids the checkerboard artifacts of transposed convs.
      - The following conv learns to blend the upsampled values and change
        channels.

    [B, C_in, T, H,   W  ] → [B, C_out, T, H*2, W*2]
    """

    def __init__(self, in_channels: int, out_channels: int, conv_type: str = "factorized"):
        super().__init__()
        self.conv = make_causal_conv(in_channels, out_channels, kernel_size=3, conv_type=conv_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale_factor=(1,2,2): leave T alone, double H and W
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


# ── Temporal down / up ────────────────────────────────────────────────────────

class DownsampleTemporal(nn.Module):
    """
    Halves T using a strided causal conv on the time axis. H and W unchanged.

    A causal conv with kernel_t=2 and stride_t=2 looks at pairs of frames
    (t-1, t) and maps each pair to one output frame.  Because it is causal
    (left-padded), the output at position t depends only on frames ≤ t.

    Why stride instead of pooling?
      Strided conv is learnable — the model can decide what to keep from each
      pair of frames, rather than just averaging or taking the max.

    [B, C_in, T,   H, W] → [B, C_out, T/2, H, W]
    T must be even.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_t: int = 2):
        super().__init__()
        # kernel (kernel_t, 1, 1): only the time axis, no spatial mixing.
        # stride (2, 1, 1): halve T, keep H and W.
        # time_pad = kernel_t - 1: left-only causal padding (same as CausalConv3d).
        self.time_pad = kernel_t - 1
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kernel_t, 1, 1),
            stride=(2, 1, 1),
            padding=(0, 0, 0),   # manual causal padding applied in forward()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad time so the first output frame sees only past frames.
        if self.time_pad > 0:
            pad = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([pad, x], dim=2)
        return self.conv(x)


class UpsampleTemporal(nn.Module):
    """
    Doubles T using nearest-neighbour interpolation + causal conv.

    Same philosophy as UpsampleSpatial: interpolation provides the upsampled
    grid cheaply; the conv learns how to fill in plausible intermediate frames.

    [B, C_in, T,   H, W] → [B, C_out, T*2, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # kernel (3, 1, 1): temporal smoothing after interpolation, no spatial change.
        self.time_pad = 2   # kernel_t - 1 = 3 - 1
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(3, 1, 1),
            padding=(0, 0, 0),   # manual causal padding in forward()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale_factor=(2,1,1): double T, leave H and W alone
        x = F.interpolate(x, scale_factor=(2, 1, 1), mode="nearest")
        # Causal left-pad before temporal conv
        pad = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
        x = torch.cat([pad, x], dim=2)
        return self.conv(x)


if __name__ == "__main__":
    B, T, H, W = 2, 8, 32, 32

    # ── ResBlock3D ────────────────────────────────────────────────────────────
    x = torch.randn(B, 64, T, H, W)
    blk = ResBlock3D(64)
    y = blk(x)
    assert y.shape == x.shape, f"ResBlock3D shape mismatch: {y.shape}"
    print(f"ResBlock3D        {tuple(x.shape)} → {tuple(y.shape)}")

    # ── Spatial down / up ─────────────────────────────────────────────────────
    x = torch.randn(B, 64, T, H, W)
    y = DownsampleSpatial(64, 128)(x)
    assert y.shape == (B, 128, T, H // 2, W // 2)
    print(f"DownsampleSpatial {tuple(x.shape)} → {tuple(y.shape)}")

    y = UpsampleSpatial(64, 32)(x)
    assert y.shape == (B, 32, T, H * 2, W * 2)
    print(f"UpsampleSpatial   {tuple(x.shape)} → {tuple(y.shape)}")

    # ── Temporal down / up ────────────────────────────────────────────────────
    x = torch.randn(B, 64, T, H, W)
    y = DownsampleTemporal(64, 64)(x)
    assert y.shape == (B, 64, T // 2, H, W)
    print(f"DownsampleTemporal{tuple(x.shape)} → {tuple(y.shape)}")

    y = UpsampleTemporal(64, 64)(x)
    assert y.shape == (B, 64, T * 2, H, W)
    print(f"UpsampleTemporal  {tuple(x.shape)} → {tuple(y.shape)}")

    # ── Causality check on ResBlock3D ─────────────────────────────────────────
    blk.eval()
    x = torch.randn(B, 64, T, H, W)
    x_corrupt = x.clone()
    x_corrupt[:, :, T // 2:] = torch.randn_like(x_corrupt[:, :, T // 2:])
    with torch.no_grad():
        diff = (blk(x) - blk(x_corrupt))[:, :, :T // 2].abs().max().item()
    print(f"ResBlock3D causality max diff: {diff:.2e}")
    assert diff < 1e-5, f"causality violated: {diff:.2e}"

    print("all assertions passed")
