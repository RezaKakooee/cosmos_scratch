import torch
import torch.nn as nn

from cosmos.tokenizers.wavelet import InverseWavelet3D
from cosmos.tokenizers.causal_conv import make_causal_conv
from cosmos.tokenizers.blocks import ResBlock3D, UpsampleSpatial, UpsampleTemporal
from cosmos.tokenizers.attention import CausalSpatiotemporalAttention


class Decoder(nn.Module):
    """
    Causal video decoder — exact mirror of Encoder.

    Shape flow  (example: latent T/4, H/4=16, base_channels=64, temporal_compression=4):

      z                    [B,  L, T/tc, H/4,  W/4]
      proj  L→deep_ch      [B, 4C, T/4,  H/4,  W/4]
      ── if temporal_compression == 4 ──────────────────────────────────────
      ResBlock(4C)         [B, 4C, T/4,  H/4,  W/4]
      CausalAttention(4C)  [B, 4C, T/4,  H/4,  W/4]
      UpsampleTemporal     [B, 2C, T/2,  H/4,  W/4]   temporal ×2
      ── if temporal_compression >= 2 ──────────────────────────────────────
      ResBlock(2C)         [B, 2C, T/2,  H/4,  W/4]
      CausalAttention(2C)  [B, 2C, T/2,  H/4,  W/4]
      UpsampleTemporal     [B, 2C, T,    H/4,  W/4]   temporal ×2  (total ×tc)
      ──────────────────────────────────────────────────────────────────────
      ResBlock(2C)         [B, 2C, T,    H/4,  W/4]
      UpsampleSpatial      [B,  C, T,    H/2,  W/2]   spatial ×2
      ResBlock(C)          [B,  C, T,    H/2,  W/2]
      out conv  C→12       [B, 12, T,    H/2,  W/2]
      InverseWavelet3D     [B,  3, T,    H,    W  ]   spatial ×2  (total ×4)
      tanh                 [B,  3, T,    H,    W  ]   bound to [-1, 1]

    deep_ch = 4C for tc=4, else 2C.
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
        deep_ch = 4 * C if temporal_compression == 4 else 2 * C
        rb = dict(conv_type=conv_type)

        self.proj = make_causal_conv(L, deep_ch, kernel_size=1, conv_type=conv_type)

        # temporal stage 2 (tc == 4): T/4 → T/2, 4C → 2C
        if temporal_compression == 4:
            self.res1  = ResBlock3D(4 * C, **rb)
            self.attn1 = (CausalSpatiotemporalAttention(4 * C, num_heads=8)
                          if use_attention else nn.Identity())
            self.up_t1 = UpsampleTemporal(4 * C, 2 * C)

        # temporal stage 1 (tc >= 2): T/2 → T, 2C → 2C
        if temporal_compression >= 2:
            self.res2  = ResBlock3D(2 * C, **rb)
            self.attn2 = (CausalSpatiotemporalAttention(2 * C, num_heads=8)
                          if use_attention else nn.Identity())
            self.up_t2 = UpsampleTemporal(2 * C, 2 * C)

        # spatial stage: H/4 → H/2, 2C → C
        self.res3        = ResBlock3D(2 * C, **rb)
        self.up_s        = UpsampleSpatial(2 * C, C, conv_type=conv_type)
        self.res4        = ResBlock3D(C, **rb)
        self.out         = make_causal_conv(C, 12, kernel_size=3, conv_type=conv_type)
        self.inv_wavelet = InverseWavelet3D()   # fixed, no parameters

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, T/tc, H/4, W/4]
        x = self.proj(z)      # [B, deep_ch, T/tc, H/4, W/4]

        if self.temporal_compression == 4:
            x = self.res1(x)
            x = self.attn1(x)
            x = self.up_t1(x)   # [B, 2C, T/2, H/4, W/4]

        if self.temporal_compression >= 2:
            x = self.res2(x)
            x = self.attn2(x)
            x = self.up_t2(x)   # [B, 2C, T,   H/4, W/4]

        # after both temporal stages (or neither), x is [B, 2C, T, H/4, W/4]
        x = self.res3(x)
        x = self.up_s(x)      # [B, C, T, H/2, W/2]
        x = self.res4(x)
        x = self.out(x)       # [B, 12, T, H/2, W/2]
        return torch.tanh(self.inv_wavelet(x))
