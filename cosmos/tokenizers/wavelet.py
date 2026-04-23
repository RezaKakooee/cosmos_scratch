import torch
import torch.nn as nn


class Wavelet3D(nn.Module):
    """
    One level of 2D Haar wavelet decomposition applied to the spatial dimensions
    of a video tensor, independently at each time step.

    Why wavelets at all?
      Neighbouring pixels are highly correlated — most of the signal energy in
      a natural image sits in low spatial frequencies.  The wavelet separates
      each 2×2 pixel block into one low-frequency "average" subband (LL) that
      carries coarse appearance, and three high-frequency "detail" subbands
      (LH, HL, HH) that carry edges and texture.  The encoder that follows can
      then focus on compressing semantics rather than doing basic decorrelation.

    Haar is the simplest orthonormal wavelet: its filters are just [1, 1]/√2
    (low-pass) and [1, −1]/√2 (high-pass) applied along each axis.

    Subbands produced per input channel (for a 2×2 block with corners a,b,c,d):
      LL  low-low   = (a+b+c+d)/2  — spatial average   (most energy, coarse content)
      LH  low-high  = (a−b+c−d)/2  — horizontal edges
      HL  high-low  = (a+b−c−d)/2  — vertical edges
      HH  high-high = (a−b−c+d)/2  — diagonal details  (least energy)

    Shape:
      Input  [B, C,  T, H,   W  ]
      Output [B, 4C, T, H/2, W/2]   — channels ×4, spatial /2 each axis

    No learnable parameters.  H and W must be even.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for Haar wavelet"

        # Reshape so each 2×2 spatial block is explicit:
        # [B, C, T, H, W] → [B, C, T, H/2, 2, W/2, 2]
        # dim -3 = row index within block (0=top, 1=bottom)
        # dim -1 = col index within block (0=left, 1=right)
        x = x.view(B, C, T, H // 2, 2, W // 2, 2)

        # Four corners of each 2×2 block, each [B, C, T, H/2, W/2]
        a = x[:, :, :, :, 0, :, 0]   # top-left
        b = x[:, :, :, :, 0, :, 1]   # top-right
        c = x[:, :, :, :, 1, :, 0]   # bottom-left
        d = x[:, :, :, :, 1, :, 1]   # bottom-right

        # Apply 2D Haar filters over each 2×2 block.
        # The ×0.5 factor is the joint normalisation for both spatial axes
        # (one ÷√2 per axis = ÷2 total), keeping the transform orthonormal.
        LL = (a + b + c + d) * 0.5   # 2D low-pass  — coarse content
        LH = (a - b + c - d) * 0.5   # col high-pass — horizontal edges
        HL = (a + b - c - d) * 0.5   # row high-pass — vertical edges
        HH = (a - b - c + d) * 0.5   # both high-pass — diagonal details

        # Concatenate subbands along the channel axis → [B, 4C, T, H/2, W/2]
        # Order: LL | LH | HL | HH  (InverseWavelet3D relies on this order)
        return torch.cat([LL, LH, HL, HH], dim=1)


class InverseWavelet3D(nn.Module):
    """
    Exact inverse of Wavelet3D: reconstructs the full spatial resolution from
    the four Haar subbands.

    If the subbands are unmodified, reconstruction is lossless (up to float
    rounding); after the encoder-decoder pass the subbands are compressed, so
    the reconstruction is approximate.

    Inversion:
      Solving the four forward equations for the original corners gives:
        a (top-left)     = (LL + LH + HL + HH) / 2
        b (top-right)    = (LL − LH + HL − HH) / 2
        c (bottom-left)  = (LL + LH − HL − HH) / 2
        d (bottom-right) = (LL − LH − HL + HH) / 2

      These four values are then interleaved back into the full-resolution grid:
        even rows, even cols ← a
        even rows, odd  cols ← b
        odd  rows, even cols ← c
        odd  rows, odd  cols ← d

    Shape:
      Input  [B, 4C, T, H/2, W/2]
      Output [B, C,  T, H,   W  ]

    No learnable parameters.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, T, H2, W2 = x.shape
        assert C4 % 4 == 0, "Channel dim must be divisible by 4 (four Haar subbands)"
        C = C4 // 4

        # Split the four subbands — each [B, C, T, H/2, W/2]
        LL = x[:, 0*C : 1*C]
        LH = x[:, 1*C : 2*C]
        HL = x[:, 2*C : 3*C]
        HH = x[:, 3*C : 4*C]

        # Reconstruct the four 2×2 block corners using the inverse Haar equations
        a = (LL + LH + HL + HH) * 0.5   # top-left
        b = (LL - LH + HL - HH) * 0.5   # top-right
        c = (LL + LH - HL - HH) * 0.5   # bottom-left
        d = (LL - LH - HL + HH) * 0.5   # bottom-right

        # Interleave the four corners back into the full spatial grid.
        # Using index strides 0::2 / 1::2 to assign even/odd rows and cols.
        out = torch.empty(B, C, T, H2 * 2, W2 * 2, device=x.device, dtype=x.dtype)
        out[:, :, :, 0::2, 0::2] = a   # even row, even col
        out[:, :, :, 0::2, 1::2] = b   # even row, odd  col
        out[:, :, :, 1::2, 0::2] = c   # odd  row, even col
        out[:, :, :, 1::2, 1::2] = d   # odd  row, odd  col
        return out


if __name__ == "__main__":
    B, C, T, H, W = 1, 1, 2, 4, 4
    x = torch.randn(B, C, T, H, W)

    fwd = Wavelet3D()
    inv = InverseWavelet3D()

    y     = fwd(x)
    x_rec = inv(y)

    print(f"input  : {tuple(x.shape)}")
    print(f"wavelet: {tuple(y.shape)}")    # [1, 4, 2, 2, 2]
    print(f"recon  : {tuple(x_rec.shape)}")

    assert y.shape     == (B, 4 * C, T, H // 2, W // 2), f"unexpected wavelet shape {y.shape}"
    assert x_rec.shape == x.shape,                        f"unexpected recon shape {x_rec.shape}"

    # Perfect reconstruction: inverse(forward(x)) == x
    err = (x_rec - x).abs().max().item()
    print(f"max reconstruction error: {err:.2e}")
    assert err < 1e-5, f"InverseWavelet3D should be lossless, got err={err:.2e}"

    print("all assertions passed")
