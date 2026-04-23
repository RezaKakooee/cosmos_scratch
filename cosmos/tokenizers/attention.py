import torch
import torch.nn as nn


class CausalSpatiotemporalAttention(nn.Module):
    """
    Causal self-attention over the full (T, H, W) volume of a feature map.

    ── Why attention at all? ────────────────────────────────────────────────────
    Convolutions (even causal 3D ones) have a fixed receptive field.  A frame
    near the end of the clip cannot directly attend to a frame at the start
    unless the network is deep enough that the receptive field reaches it.
    Attention solves this by giving every spatial token direct access to every
    earlier token in one layer, regardless of distance.

    ── What does "spatiotemporal" mean here? ────────────────────────────────────
    Each (t, h, w) position in the feature map becomes one token.  So for a
    feature map of shape [B, C, T, H, W] we have N = T×H×W tokens per batch
    item, each with C-dimensional features.  A single attention layer can then
    relate any pixel at any time to any other pixel at any earlier time.

    ── Causal mask ──────────────────────────────────────────────────────────────
    We allow a token at time t to attend to ALL spatial positions at times ≤ t,
    but NEVER to any token at time > t (future leakage).

    The mask is *block-causal*: the N×N attention matrix is divided into T×T
    blocks of size (H×W)×(H×W).  Blocks on or below the diagonal are unmasked;
    blocks strictly above the diagonal are masked out.

    Visualised for T=3, each cell = one H×W block:

        attend to →   t=0      t=1      t=2
        t=0        [  open  |  MASK  |  MASK  ]
        t=1        [  open  |  open  |  MASK  ]
        t=2        [  open  |  open  |  open  ]

    Tokens at the same time step can freely attend to each other spatially
    (intra-frame attention), because that introduces no temporal leakage.

    ── Design choices ───────────────────────────────────────────────────────────
    - Pre-norm (GroupNorm before attention) + residual add: more stable than
      post-norm for deep networks.
    - The causal mask is computed once per (T, H, W) combination and cached
      so it is not rebuilt on every forward call.
    - Memory note: the attention matrix is N×N = (T·H·W)² floats.  At
      T=8, H=W=16 (a typical deep-stage feature map) that is 2048×2048 ≈ 16 MB
      per head — manageable on a modern GPU.  Avoid using this block on early
      (high-resolution) feature maps.

    Input / output shape: [B, C, T, H, W]
    """

    def __init__(
        self,
        channels:   int,
        num_heads:  int = 8,
        num_groups: int = 8,   # for GroupNorm
    ):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        # Pre-norm: normalise before attention so gradients are well-scaled
        self.norm = nn.GroupNorm(num_groups, channels)

        # Standard multi-head self-attention.
        # batch_first=True means input/output shape is [B, N, C]
        # (rather than PyTorch's older default [N, B, C]).
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

        # Cache for the causal mask — reused across forward calls with the
        # same (T, H, W).  Key: (T, H, W), Value: bool tensor [T*H*W, T*H*W]
        self._mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def _causal_mask(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Build (or retrieve from cache) the N×N boolean causal mask.

        mask[i, j] = True  →  token i is NOT allowed to attend to token j.

        Token i lives at time step  t_i = i // (H * W).
        Token j should be masked if  t_j > t_i  (j is in the future).
        """
        key = (T, H, W)
        if key not in self._mask_cache or self._mask_cache[key].device != device:
            N = T * H * W
            # time step of each token: repeats each value H*W times
            # e.g. T=2, H=W=2: [0, 0, 0, 0, 1, 1, 1, 1]
            time_idx = torch.arange(N, device=device) // (H * W)   # [N]

            # mask[i, j] = True when the time of j is strictly after the time of i
            # time_idx[:, None] broadcasts to [N, 1], time_idx[None, :] to [1, N]
            mask = time_idx[None, :] > time_idx[:, None]   # [N, N]  bool
            self._mask_cache[key] = mask

        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # ── Pre-norm (per-frame, causal) ──────────────────────────────────────
        residual = x                     # save input for residual add

        # IMPORTANT: GroupNorm applied directly to [B, C, T, H, W] computes
        # statistics over all T frames together — future frames would shift
        # the normalisation of past frames, breaking causality.
        #
        # Fix: merge batch and time into one axis before normalising.
        # GroupNorm then treats each (batch, time) frame as an independent
        # sample, so frame t is normalised using only its own pixels.
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T, C, H, W]
        x = self.norm(x)                                          # per-frame stats
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)     # [B, C, T, H, W]

        # ── Flatten (T, H, W) → sequence of N tokens ─────────────────────────
        # Rearrange from [B, C, T, H, W]
        #             to [B, T, H, W, C]   (move C to last)
        #             to [B, T*H*W,   C]   (flatten spatial+temporal into one axis)
        # Each of the N = T*H*W positions becomes one token of dimension C.
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)

        # ── Causal self-attention ─────────────────────────────────────────────
        # attn_mask is bool: True = masked out (future positions).
        # MultiheadAttention adds -inf to masked positions before softmax so
        # their contribution after softmax is ≈ 0.
        mask = self._causal_mask(T, H, W, x.device)
        x, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        # x: [B, T*H*W, C]

        # ── Unflatten back to video shape ─────────────────────────────────────
        # [B, T*H*W, C] → [B, T, H, W, C] → [B, C, T, H, W]
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        # ── Residual add ──────────────────────────────────────────────────────
        # Adding the original input back stabilises training and lets the
        # block learn a residual correction rather than the full mapping.
        return residual + x


if __name__ == "__main__":
    B, C, T, H, W = 2, 256, 4, 8, 8
    x = torch.randn(B, C, T, H, W)

    attn = CausalSpatiotemporalAttention(channels=256, num_heads=8)
    y = attn(x)

    print(f"input : {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    assert y.shape == x.shape, f"shape mismatch: {y.shape}"

    # ── Causality check ───────────────────────────────────────────────────────
    # Corrupt all frames after split_t and verify past outputs are unchanged.
    split_t = T // 2
    x_corrupt = x.clone()
    x_corrupt[:, :, split_t:] = torch.randn_like(x_corrupt[:, :, split_t:])

    with torch.no_grad():
        y_clean   = attn(x)
        y_corrupt = attn(x_corrupt)

    max_diff = (y_clean[:, :, :split_t] - y_corrupt[:, :, :split_t]).abs().max().item()
    print(f"causality max diff (past frames): {max_diff:.2e}")
    assert max_diff < 1e-5, f"causality violated: max_diff={max_diff:.2e}"

    # ── Mask shape check ─────────────────────────────────────────────────────
    mask = attn._causal_mask(T, H, W, torch.device("cpu"))
    N = T * H * W
    assert mask.shape == (N, N)
    # first H*W tokens (t=0) should never mask each other (same time step)
    assert not mask[:H*W, :H*W].any(), "tokens at t=0 should attend to each other"
    # first token should mask all tokens at t>0
    assert mask[0, H*W:].all(), "t=0 token must not attend to t=1,2,..."

    print("all assertions passed")
