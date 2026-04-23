import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn

from cosmos.tokenizers.encoder import Encoder
from cosmos.tokenizers.decoder import Decoder


class ContinuousTokenizer(nn.Module):
    """
    Continuous causal video tokenizer.

    Encodes a video clip to a compact latent and decodes it back.

    Compression (default settings: base_channels=64, temporal_compression=4):
      Input   [B, 3,  T,    H,    W  ]   e.g. [B, 3, 8, 64, 64]
      Latent  [B, L,  T/tc, H/4,  W/4]   e.g. [B, 16, 2, 16, 16]
      Output  [B, 3,  T,    H,    W  ]   in [-1, 1]

    Key parameters:
      latent_channels      — channel depth of the latent (default 16)
      base_channels        — channel width at the first conv stage; widths
                             scale as C, 2C, 4C through the network (default 64)
      temporal_compression — 1 (spatial only), 2 (T÷2), or 4 (T÷4)  (default 4)
      use_attention        — enable causal spatiotemporal attention blocks
      conv_type            — "factorized" (Day 2) or "full" (Day 1 style)
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
        kwargs = dict(
            latent_channels=latent_channels,
            base_channels=base_channels,
            temporal_compression=temporal_compression,
            use_attention=use_attention,
            conv_type=conv_type,
        )
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, T, H, W] → z: [B, L, T/tc, H/4, W/4]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, T/tc, H/4, W/4] → x_hat: [B, 3, T, H, W] in [-1, 1]"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x → (x_hat, z)"""
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


if __name__ == "__main__":
    B, C, T, H, W = 2, 3, 8, 64, 64
    x = torch.randn(B, C, T, H, W)

    for tc in (1, 2, 4):
        for ct in ("factorized", "full"):
            model    = ContinuousTokenizer(temporal_compression=tc, conv_type=ct)
            x_hat, z = model(x)
            n        = sum(p.numel() for p in model.parameters())
            print(f"tc={tc}  conv={ct:11s}  z={tuple(z.shape)}  params={n:,}")
            assert z.shape     == (B, 16, T // tc, H // 4, W // 4), f"bad z: {z.shape}"
            assert x_hat.shape == (B, C, T, H, W),                   f"bad x_hat: {x_hat.shape}"
            assert x_hat.min() >= -1 - 1e-5 and x_hat.max() <= 1 + 1e-5

    print("all assertions passed")
