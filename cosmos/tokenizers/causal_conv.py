import torch
import torch.nn as nn


class CausalConv3d(nn.Module):
    """
    3D convolution that is causal in time.

    Pads (kernel_t - 1) frames on the LEFT of the time axis only,
    so output at time t depends only on frames <= t.
    Height and width are padded symmetrically as normal.

    Input / output shape: [B, C, T, H, W]
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int | tuple[int, int, int] = 3,
        stride:       int | tuple[int, int, int] = 1,
        bias:         bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        kt, kh, kw = kernel_size

        # spatial symmetric padding, zero temporal padding (we pad manually)
        # spatial padding is just to preserve image size while letting the kernel see nearby spatial neighbors.
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2

        self.time_pad = kt - 1          # left-pad this many frames in time
        # left padding keeps past-context access; no right padding prevents future leakage
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, pad_h, pad_w),  # no built-in time padding
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]

        # ── THIS is where causality is enforced ───────────────────────────────
        #
        # A Conv3d with kernel_t=3 and NO padding would look at positions
        # [t-1, t, t+1] — it leaks one future frame into every output.
        #
        # PyTorch's built-in symmetric padding makes it worse:
        # padding=1 adds one zero on each side → still looks at [t-1, t, t+1].
        #
        # Instead we manually add (kernel_t - 1) = 2 past frames on the LEFT
        # and set Conv3d temporal padding = 0 so it never adds anything on
        # the right.
        #
        # Timeline with kernel_t=3, time_pad=2:
        #
        #   Original:   [  f0,  f1,  f2,  f3,  f4 ]
        #   After pad:  [ f0*, f0*, f0,  f1,  f2,  f3,  f4 ]   (* = replicated)
        #                  ↑    ↑
        #                  prepended copies of the first frame
        #
        #   Conv window at output t=0  →  sees [ f0*, f0*, f0 ]  (all past/present)
        #   Conv window at output t=1  →  sees [ f0*, f0,  f1 ]  (past/present)
        #   Conv window at output t=2  →  sees [ f0,  f1,  f2 ]  (past/present)
        #
        #   No output ever sees a frame AFTER its own time index → causal ✓
        #
        # We replicate f0 rather than padding with zeros so that the boundary
        # frame gets the same distribution as interior frames (avoids a sharp
        # discontinuity that the model would have to learn to ignore).
        # ─────────────────────────────────────────────────────────────────────
        if self.time_pad > 0:
            pad = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([pad, x], dim=2)   # [B, C, T + time_pad, H, W]
        return self.conv(x)


class FactorizedCausalConv3d(nn.Module):
    """
    Factorized causal 3D convolution: spatial conv followed by temporal conv.

    Why factorize?
      A full 3D conv with kernel (kt, kh, kw) mixes space and time in one step.
      Factorizing into two separate passes has two advantages:
        1. Fewer parameters: (1·kh·kw + kt·1·1) × C²  vs  (kt·kh·kw) × C²
        2. Cleaner inductive bias: the model explicitly learns "what is at this
           location" (spatial pass) before "how does it change over time"
           (temporal pass).  This mirrors how the Cosmos paper treats space and
           time as separate axes in the attention block too.

    Two-pass design:
      Pass 1 — Spatial conv  kernel (1, k, k):
        - No temporal mixing at all (kt=1 means no time axis is touched).
        - Symmetric spatial padding preserves H and W.
        - Maps in_channels → out_channels.

      Pass 2 — Temporal conv  kernel (k, 1, 1):
        - No spatial mixing (kh=kw=1 means H, W are untouched).
        - Left-only temporal padding: replicate the first frame (k-1) times so
          the output at time t only sees frames ≤ t  (causal guarantee).
        - Keeps channels at out_channels → out_channels.

    Shape:
      Input  [B, in_channels,  T, H, W]
      Output [B, out_channels, T, H, W]   — T, H, W are preserved
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,   # applied to both spatial (kh=kw) and temporal (kt)
        bias:         bool = True,
    ):
        super().__init__()

        # ── Pass 1: spatial conv ──────────────────────────────────────────────
        # kernel (1, k, k) — touches only H and W, never T.
        # Symmetric padding (k-1)//2 keeps H and W unchanged.
        pad_s = (kernel_size - 1) // 2
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, pad_s, pad_s),
            bias=bias,
        )

        # ── Pass 2: temporal conv ─────────────────────────────────────────────
        # kernel (k, 1, 1) — touches only T, never H or W.
        # NO built-in temporal padding; we apply left-only padding manually in
        # forward() to enforce causality.
        self.time_pad = kernel_size - 1   # how many frames to prepend on the left
        self.temporal_conv = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(0, 0, 0),            # manual causal padding in forward()
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T, H, W]

        # Pass 1 — spatial: mix neighbourhood pixels within each frame.
        # T is untouched because kernel_t = 1.
        x = self.spatial_conv(x)   # [B, out_channels, T, H, W]

        # Pass 2 — temporal: THIS is where causality is enforced.
        #
        # Same mechanism as CausalConv3d: prepend (kernel_size-1) copies of the
        # first frame so the conv window at every output position t only covers
        # frames ≤ t.  temporal_conv has padding=(0,0,0) so Conv3d itself never
        # adds anything on the right side of the time axis.
        #
        # Example with kernel_size=3, time_pad=2:
        #
        #   After spatial pass: [ f0,  f1,  f2,  f3,  f4 ]
        #   After prepend:      [ f0*, f0*, f0,  f1,  f2,  f3,  f4 ]
        #
        #   kernel window slides right with stride 1, no right-pad:
        #     output t=0  ←  [ f0*, f0*, f0 ]   only past ✓
        #     output t=1  ←  [ f0*, f0,  f1 ]   only past ✓
        #     output t=4  ←  [ f3,  f4,  ——]    wait — there is no f5, and we
        #                                         added no right pad, so the
        #                                         padded length is exactly T+2
        #                                         and the last window is [f3,f4]
        #                                         → still no future leakage ✓
        if self.time_pad > 0:
            pad = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([pad, x], dim=2)   # [B, C, T + time_pad, H, W]

        x = self.temporal_conv(x)  # [B, out_channels, T, H, W]
        return x


def make_causal_conv(
    in_channels:  int,
    out_channels: int,
    kernel_size:  int,
    conv_type:    str = "factorized",
    **kwargs,
) -> nn.Module:
    """
    Factory that returns either a full CausalConv3d or a FactorizedCausalConv3d.

    conv_type options:
      "factorized" — spatial pass (1,k,k) then temporal pass (k,1,1).
                     Fewer params; explicit space/time separation (Day 2 default).
      "full"       — joint 3D kernel (k,k,k).
                     More expressive; models space-time jointly (Day 1 style).

    kernel_size=1 always uses CausalConv3d regardless of conv_type because a
    1×1×1 kernel is identical in both variants.
    """
    if kernel_size == 1 or conv_type == "full":
        return CausalConv3d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
    return FactorizedCausalConv3d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)


if __name__ == "__main__":
    B, C_in, C_out, T, H, W = 2, 12, 32, 8, 32, 32
    x = torch.randn(B, C_in, T, H, W)

    conv = FactorizedCausalConv3d(C_in, C_out, kernel_size=3)
    y = conv(x)

    print(f"input : {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    assert y.shape == (B, C_out, T, H, W), f"unexpected shape {y.shape}"

    # Causality check: corrupting frames after split_t must not change output up to split_t
    split_t = T // 2
    x_corrupt = x.clone()
    x_corrupt[:, :, split_t:] = torch.randn_like(x_corrupt[:, :, split_t:])

    y_clean   = conv(x)
    y_corrupt = conv(x_corrupt)

    max_diff = (y_clean[:, :, :split_t] - y_corrupt[:, :, :split_t]).abs().max().item()
    print(f"causality max diff (past frames): {max_diff:.2e}")
    assert max_diff < 1e-5, f"causality violated: max_diff={max_diff:.2e}"

    print("all assertions passed")
