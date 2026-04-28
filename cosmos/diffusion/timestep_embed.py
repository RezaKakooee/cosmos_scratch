import math
import torch
import torch.nn as nn


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Map a 1-D tensor of scalar values to sinusoidal Fourier features.

    This is the same idea as positional encoding in the original Transformer,
    but applied to a scalar (noise level or timestep) instead of a position.

    Construction:
      - Use `dim//2` frequencies spaced log-uniformly between 1 and 10 000.
      - For each frequency f_i, compute  sin(t * f_i)  and  cos(t * f_i).
      - Concatenate → a `dim`-dimensional vector per sample.

    Why sinusoidal?
      - Deterministic (no learned parameters here).
      - Smooth: nearby noise levels map to nearby vectors.
      - Injective: different noise levels map to different vectors.

    Input:  t  [B]        — one scalar per sample (e.g. log σ)
    Output:    [B, dim]
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2

    # Log-uniform frequencies: f_0 = 1, f_{half-1} = 10000
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device)
        / (half - 1)
    )                                       # [half]

    args = t[:, None] * freqs[None, :]      # [B, half] # args stands for "angles" or "arguments" to the sin/cos functions
    return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]


class TimestepEmbed(nn.Module):
    """
    Noise-level (σ) → conditioning vector used by every DiT block.

    Pipeline:
      σ  [B]
      → log(σ) / 4          normalise to a reasonable numeric range
      → sinusoidal_embed     [B, freq_dim]   fixed Fourier features
      → Linear → SiLU        [B, d_model]
      → Linear               [B, d_model]

    The final vector is added to (or used to modulate) the hidden state in
    each DiT block so the denoiser knows how noisy its input is.

    For AdaLN-style conditioning (scale + shift), set out_dim = 2 * d_model
    and split the output into (scale, shift) inside the DiT block.  The
    default out_dim = d_model is fine for simple additive conditioning.

    Why log(σ) / 4?
      EDM noise levels span a wide range (σ ∈ [0.002, 80]).  Taking log
      maps that to roughly [-6, 4.4].  Dividing by 4 puts most of the mass
      in [-1.5, 1.1], which is a well-behaved range for sinusoidal encoding.
    """

    def __init__(
        self,
        d_model:  int = 768,
        freq_dim: int = 256,
        out_dim:  int | None = None,
    ):
        super().__init__()
        self.freq_dim = freq_dim
        out_dim = out_dim or d_model

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        sigma: [B]  — per-sample noise level (standard deviation)
        returns [B, out_dim]
        """
        # Normalise noise level to a compact numeric range
        t   = sigma.float().log() / 4           # [B]
        emb = sinusoidal_embed(t, self.freq_dim) # [B, freq_dim]
        return self.mlp(emb)                    # [B, out_dim]


if __name__ == "__main__":
    B, d_model = 4, 768

    embed = TimestepEmbed(d_model=d_model, freq_dim=256)

    # Test with a range of σ values spanning the EDM range [0.002, 80]
    sigmas = torch.tensor([0.002, 0.5, 5.0, 80.0])
    out    = embed(sigmas)

    print(f"sigma  : {sigmas.tolist()}")
    print(f"emb    : {tuple(out.shape)}")
    assert out.shape == (4, d_model)

    # Different σ values must produce different embeddings
    assert not torch.allclose(out[0], out[-1]), "all embeddings are the same!"

    # AdaLN mode: out_dim = 2 * d_model → split into scale and shift
    embed_ada = TimestepEmbed(d_model=d_model, freq_dim=256, out_dim=2 * d_model)
    ada_out   = embed_ada(sigmas)
    scale, shift = ada_out.chunk(2, dim=-1)
    assert scale.shape == (4, d_model)
    assert shift.shape == (4, d_model)
    print(f"AdaLN  scale: {tuple(scale.shape)}  shift: {tuple(shift.shape)}")

    n_params = sum(p.numel() for p in embed.parameters())
    print(f"params : {n_params:,}")
    print("all assertions passed")
