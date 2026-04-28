import math
import torch
import torch.nn as nn


def _sincos_1d(length: int, dim: int, device=None) -> torch.Tensor:
    """
    Build 1-D sinusoidal position encodings.

    For positions 0 … length-1, produce a (length, dim) matrix where
    each row is the classic Transformer sinusoidal encoding:

      pos_enc[pos, 2i]   = sin(pos / 10000^(2i/dim))
      pos_enc[pos, 2i+1] = cos(pos / 10000^(2i/dim))

    Returns  [length, dim]
    """
    assert dim % 2 == 0
    half   = dim // 2
    pos    = torch.arange(length, dtype=torch.float32, device=device)   # [L]
    freqs  = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=device)
        / (half - 1)
    )                                                                    # [half]
    angles = pos[:, None] * freqs[None, :]                              # [L, half]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)              # [L, dim]


class RoPE3D(nn.Module):
    """
    Factorized 3-D sinusoidal positional encoding for patch token sequences.

    Each token covers one (t, h, w) position in the patch grid.  We assign
    a positional embedding by concatenating three independent 1-D sinusoidal
    encodings — one per axis — along the channel dimension:

      pos_enc = [sin/cos(t), sin/cos(h), sin/cos(w)]   length = d_model

    Why factorized?
      A joint 3-D encoding would need d_model frequencies for a 3-D grid,
      which is hard to scale and harder to generalise to unseen grid sizes.
      Factorizing means each axis gets d_model/3 independent frequencies,
      which is cheaper and transfers better to different T/H/W values.

    Why sinusoidal (not learned)?
      Sinusoidal encodings generalise to sequence lengths not seen at training
      time.  For a video model that may be applied to clips of varying length
      this is a useful property.

    ── Shape flow ───────────────────────────────────────────────────────────
    Input tokens   [B, N, d_model]    N = nT × nH × nW
    Output tokens  [B, N, d_model]    same shape, positions added

    The encoding is built once and cached.  If the grid changes (different
    T/H/W), it is rebuilt automatically.
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        assert d_model % 3 == 0, \
            f"d_model must be divisible by 3 for factorized 3-D RoPE, got {d_model}"
        self.d_model  = d_model
        self.d_axis   = d_model // 3   # dim allocated per axis
        self._cache: dict[tuple, torch.Tensor] = {}

    def _build(self, nT: int, nH: int, nW: int, device) -> torch.Tensor:
        """
        Build the [N, d_model] positional encoding for a (nT, nH, nW) grid.
        Cached so it is only computed once per grid shape.
        """
        key = (nT, nH, nW, str(device))
        if key not in self._cache:
            enc_t = _sincos_1d(nT, self.d_axis, device)   # [nT, d/3]
            enc_h = _sincos_1d(nH, self.d_axis, device)   # [nH, d/3]
            enc_w = _sincos_1d(nW, self.d_axis, device)   # [nW, d/3]

            # Broadcast each 1-D encoding across the other two axes,
            # then flatten (nT, nH, nW) → N.
            # enc_t:  [nT, 1,   1,   d/3]
            # enc_h:  [1,  nH,  1,   d/3]
            # enc_w:  [1,  1,   nW,  d/3]
            enc_t = enc_t[:, None, None, :]                 # [nT, 1,  1,  d/3]
            enc_h = enc_h[None, :, None, :]                 # [1, nH,  1,  d/3]
            enc_w = enc_w[None, None, :, :]                 # [1,  1, nW,  d/3]

            # Expand to [nT, nH, nW, d/3] each, then concat on last dim
            enc_t = enc_t.expand(nT, nH, nW, -1)
            enc_h = enc_h.expand(nT, nH, nW, -1)
            enc_w = enc_w.expand(nT, nH, nW, -1)

            # [nT, nH, nW, d_model] → [N, d_model]
            enc = torch.cat([enc_t, enc_h, enc_w], dim=-1)
            self._cache[key] = enc.reshape(nT * nH * nW, self.d_model)

        return self._cache[key]

    def forward(
        self,
        tokens: torch.Tensor,
        grid:   tuple[int, int, int],
    ) -> torch.Tensor:
        """
        tokens : [B, N, d_model]
        grid   : (nT, nH, nW)   — patch grid dimensions
        returns: [B, N, d_model]  tokens + positional encoding
        """
        nT, nH, nW = grid
        enc = self._build(nT, nH, nW, tokens.device)   # [N, d_model]
        return tokens + enc.unsqueeze(0)               # broadcast over B


if __name__ == "__main__":
    B, d_model = 2, 768
    nT, nH, nW = 4, 8, 8          # patch grid for latent [4, 16, 16] with p=(1,2,2)
    N = nT * nH * nW               # 256 tokens

    rope = RoPE3D(d_model=d_model)
    tokens = torch.zeros(B, N, d_model)
    out    = rope(tokens, grid=(nT, nH, nW))

    print(f"tokens in  : {tuple(tokens.shape)}")
    print(f"tokens out : {tuple(out.shape)}")
    assert out.shape == (B, N, d_model)

    # Different grid positions must have different encodings
    assert not torch.allclose(out[0, 0], out[0, 1]), "position 0 and 1 are identical"
    assert not torch.allclose(out[0, 0], out[0, N // 2]), "position 0 and N/2 are identical"

    # Encoding is the same for both batch items (no learned params)
    assert torch.allclose(out[0], out[1]), "encoding differs across batch"

    # No learned parameters
    n_params = sum(p.numel() for p in rope.parameters())
    print(f"params     : {n_params}  (expected 0 — fully fixed)")
    assert n_params == 0

    print("all assertions passed")
