import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Adaptive LayerNorm modulation.

    Standard LayerNorm normalises x then applies a learned affine transform.
    AdaLN *conditions* that affine transform on an external signal (here: the
    noise level embedding) instead of using fixed learned weights.

      AdaLN(x, shift, scale) = (1 + scale) * LayerNorm(x) + shift

    The (1 + scale) form means scale=0 → identity, which makes the gate at
    initialisation close to 1 (the block starts as a near-identity transform).

    x     : [B, N, d_model]
    shift : [B, d_model]   — broadcast over N
    scale : [B, d_model]   — broadcast over N
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block (DiT, Peebles & Xie 2023).

    One block of the latent denoiser.  Takes a sequence of patch tokens,
    a noise-level conditioning vector, and an optional context sequence
    (text embeddings for cross-attention).

    ── Layout ───────────────────────────────────────────────────────────────

      sigma_emb [B, d_model]
        └─ AdaLN projector → 6 vectors of size d_model each:
           (shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp)

      x [B, N, d_model]
        ├─ AdaLN(shift_sa, scale_sa) → norm1
        ├─ SelfAttention(norm1)
        ├─ x + gate_sa * attn_out          ← gated residual
        │
        ├─ LayerNorm → norm2
        ├─ CrossAttention(norm2, context)   ← tokens attend to text/null
        ├─ x + cross_out                   ← simple residual (no gate)
        │
        ├─ AdaLN(shift_mlp, scale_mlp) → norm3
        ├─ MLP(norm3)
        └─ x + gate_mlp * mlp_out          ← gated residual

    ── AdaLN-Zero initialisation ────────────────────────────────────────────
    The AdaLN projector's final linear layer is zero-initialised.  This means
    at the start of training:
      shift = 0, scale = 0, gate = 0
    So every block starts as a pure identity (output = input) and the network
    learns to deviate from that gradually.  This stabilises early training.

    ── Null conditioning ────────────────────────────────────────────────────
    When no text is available, pass context = zeros([B, 1, d_model]).
    Cross-attention then attends to a single all-zero token, adding nothing
    meaningful — the block degrades gracefully to unconditional generation.
    """

    def __init__(
        self,
        d_model:   int   = 768,
        num_heads: int   = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # ── AdaLN modulation projector ────────────────────────────────────────
        # Projects sigma_emb → 6 × d_model (shift/scale/gate for SA and MLP).
        # Zero-init on the Linear so all modulation starts at 0 (identity blocks).
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        # ── Self-attention ────────────────────────────────────────────────────
        # No built-in affine in LayerNorm — AdaLN provides the affine transform.
        self.norm1   = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.0
        )

        # ── Cross-attention to context (text / null) ──────────────────────────
        self.norm2      = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ctx   = nn.LayerNorm(d_model)  # context has its own norm
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.0
        )

        # ── MLP ───────────────────────────────────────────────────────────────
        self.norm3  = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_hidden  = int(d_model * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(
        self,
        x:         torch.Tensor,   # [B, N, d_model]  patch tokens
        sigma_emb: torch.Tensor,   # [B, d_model]      noise-level conditioning
        context:   torch.Tensor,   # [B, L, d_model]   text tokens or null zeros
    ) -> torch.Tensor:

        # ── AdaLN parameters from noise level ────────────────────────────────
        mod = self.adaLN(sigma_emb)                          # [B, 6*d_model]
        shift_sa, scale_sa, gate_sa, \
        shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)  # each [B, d_model]

        # ── Self-attention ────────────────────────────────────────────────────
        x_norm   = modulate(self.norm1(x), shift_sa, scale_sa)  # [B, N, d_model]
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * attn_out                 # gated residual

        # ── Cross-attention ───────────────────────────────────────────────────
        # Tokens (queries) attend to context (keys/values).
        # With null context = zeros, this adds ~0 and the block is unconditional.
        x_norm2  = self.norm2(x)
        ctx_norm = self.norm_ctx(context)
        cross_out, _ = self.cross_attn(x_norm2, ctx_norm, ctx_norm)
        x = x + cross_out                                        # simple residual

        # ── MLP ───────────────────────────────────────────────────────────────
        x_norm3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm3)       # gated residual

        return x


if __name__ == "__main__":
    B, N, d_model = 2, 256, 768
    num_heads      = 12
    L_ctx          = 1   # null context: single zero token

    block = DiTBlock(d_model=d_model, num_heads=num_heads)

    x         = torch.randn(B, N, d_model)
    sigma_emb = torch.randn(B, d_model)
    context   = torch.zeros(B, L_ctx, d_model)   # null conditioning

    out = block(x, sigma_emb, context)

    print(f"x in    : {tuple(x.shape)}")
    print(f"sigma   : {tuple(sigma_emb.shape)}")
    print(f"context : {tuple(context.shape)}")
    print(f"x out   : {tuple(out.shape)}")
    assert out.shape == (B, N, d_model), f"bad output shape: {out.shape}"

    # At init (zero-init AdaLN), gate=0 → attn/mlp contribute 0 → out ≈ x + cross_attn(x, 0)
    # Cross-attn to all-zero context ≈ 0 → out ≈ x
    diff = (out - x).abs().max().item()
    print(f"max |out - x| at init (should be ~0): {diff:.2e}")

    n_params = sum(p.numel() for p in block.parameters())
    print(f"params  : {n_params:,}")
    print("all assertions passed")
