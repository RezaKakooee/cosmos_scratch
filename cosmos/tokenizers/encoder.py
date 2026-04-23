import torch
import torch.nn as nn

from cosmos.tokenizers.wavelet import Wavelet3D
from cosmos.tokenizers.causal_conv import make_causal_conv
from cosmos.tokenizers.blocks import ResBlock3D, DownsampleSpatial, DownsampleTemporal
from cosmos.tokenizers.attention import CausalSpatiotemporalAttention


class Encoder(nn.Module):
    """
    Causal video encoder: wavelet front-end → spatial compression → temporal
    compression → latent projection.

    Shape flow  (example: T=8, H=W=64, base_channels=64, temporal_compression=4):

      Input                [B,  3, T,    H,    W  ]
      Wavelet3D            [B, 12, T,    H/2,  W/2]   spatial ÷2, channels ×4
      stem  12→C           [B,  C, T,    H/2,  W/2]
      ResBlock(C)          [B,  C, T,    H/2,  W/2]
      DownsampleSpatial    [B, 2C, T,    H/4,  W/4]   spatial ÷2  (total ÷4 from input)
      ResBlock(2C)         [B, 2C, T,    H/4,  W/4]
      ── if temporal_compression >= 2 ──────────────────────────────────────
      DownsampleTemporal   [B, 2C, T/2,  H/4,  W/4]   temporal ÷2
      ResBlock(2C)         [B, 2C, T/2,  H/4,  W/4]
      CausalAttention(2C)  [B, 2C, T/2,  H/4,  W/4]
      ── if temporal_compression == 4 ──────────────────────────────────────
      DownsampleTemporal   [B, 4C, T/4,  H/4,  W/4]   temporal ÷2, channels ×2
      ResBlock(4C)         [B, 4C, T/4,  H/4,  W/4]
      CausalAttention(4C)  [B, 4C, T/4,  H/4,  W/4]
      ──────────────────────────────────────────────────────────────────────
      proj  deep_ch→L      [B,  L, T/tc, H/4,  W/4]

    temporal_compression controls the temporal bottleneck:
      1 → no temporal downsampling  → latent T unchanged
      2 → one  DownsampleTemporal   → latent T/2
      4 → two  DownsampleTemporal   → latent T/4

    deep_ch = 4C for tc=4, else 2C.
    Attention is placed after each temporal downsampling, so sequence length
    is small.  Disable with use_attention=False for fast overfit checks.
    """

    def __init__(
        self,
        latent_channels:      int  = 16,
        base_channels:        int  = 64,
        temporal_compression: int  = 4,
        use_attention:        bool = False,
        conv_type:            str  = "factorized",
    ):
        super().__init__()
        assert temporal_compression in (1, 2, 4), \
            f"temporal_compression must be 1, 2, or 4, got {temporal_compression}"

        self.temporal_compression = temporal_compression
        C, L = base_channels, latent_channels
        rb = dict(conv_type=conv_type)   # shorthand for ResBlock kwargs

        self.wavelet = Wavelet3D()   # fixed, no parameters
        self.stem    = make_causal_conv(12, C, kernel_size=3, conv_type=conv_type)

        # spatial stage: H/2 → H/4
        self.res1   = ResBlock3D(C, **rb)
        self.down_s = DownsampleSpatial(C, 2 * C)   # always CausalConv3d (strided)
        self.res2   = ResBlock3D(2 * C, **rb)

        # temporal stage 1 (tc >= 2): T → T/2, channels stay at 2C
        if temporal_compression >= 2:
            self.down_t1 = DownsampleTemporal(2 * C, 2 * C)
            self.res3    = ResBlock3D(2 * C, **rb)
            self.attn1   = (CausalSpatiotemporalAttention(2 * C, num_heads=8)
                            if use_attention else nn.Identity())

        # temporal stage 2 (tc == 4): T/2 → T/4, channels 2C → 4C
        if temporal_compression == 4:
            self.down_t2 = DownsampleTemporal(2 * C, 4 * C)
            self.res4    = ResBlock3D(4 * C, **rb)
            self.attn2   = (CausalSpatiotemporalAttention(4 * C, num_heads=8)
                            if use_attention else nn.Identity())

        deep_ch   = 4 * C if temporal_compression == 4 else 2 * C
        self.proj = make_causal_conv(deep_ch, L, kernel_size=1, conv_type=conv_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        x = self.wavelet(x)   # [B, 12, T, H/2, W/2]
        x = self.stem(x)      # [B,  C, T, H/2, W/2]

        x = self.res1(x)
        x = self.down_s(x)    # [B, 2C, T,   H/4, W/4]
        x = self.res2(x)

        if self.temporal_compression >= 2:
            x = self.down_t1(x)   # [B, 2C, T/2, H/4, W/4]
            x = self.res3(x)
            x = self.attn1(x)

        if self.temporal_compression == 4:
            x = self.down_t2(x)   # [B, 4C, T/4, H/4, W/4]
            x = self.res4(x)
            x = self.attn2(x)

        return self.proj(x)   # [B, L, T/tc, H/4, W/4]
