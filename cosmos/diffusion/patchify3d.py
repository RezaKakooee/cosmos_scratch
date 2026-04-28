import torch
import torch.nn as nn


class Patchify3D(nn.Module):
    """
    Split a latent video volume into a flat sequence of patch tokens.

    The tokenizer latent has shape  [B, C, T, H, W].
    We divide the spatial and temporal axes into non-overlapping patches of
    size (p_t, p_h, p_w) and flatten each patch into a single vector.

    ── Why patch instead of using pixels / latent cells directly? ──────────────
    The transformer attention cost is O(N²) in sequence length N.  Operating
    on every latent cell of [B, 16, 4, 16, 16] would give N = 4×16×16 = 1024
    tokens — expensive.  Grouping into (1,2,2) patches gives N = 4×8×8 = 256,
    which is 4× cheaper while still preserving spatial structure.

    ── Shape flow (default patch sizes p_t=1, p_h=2, p_w=2) ───────────────────

      Input latent    [B, C=16, T=4,  H=16, W=16]
      After patching  [B, N=256, D=64] # N is number of tokens, D is token dimension

      where:
        N = (T/p_t) × (H/p_h) × (W/p_w)
          = (4/1)   × (16/2)  × (16/2)
          = 4       × 8       × 8
          = 256

        D = p_t × p_h × p_w × C
          = 1   × 2   × 2   × 16
          = 64

    Each of the 256 tokens is a D=64 dimensional vector representing one
    spatiotemporal patch.  The token order is (T, H, W) — time-major, then
    row-major in space — which matches how we build positional encodings later.

    unpatchify() is the exact inverse: [B, N, D] → [B, C, T, H, W].
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        latent_channels: int = 16,
    ):
        super().__init__()
        self.p_t, self.p_h, self.p_w = patch_size
        self.C = latent_channels

        # D = number of values in one patch
        self.D = self.p_t * self.p_h * self.p_w * self.C

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check(self, z: torch.Tensor) -> tuple[int, int, int, int, int]:
        B, C, T, H, W = z.shape
        assert C == self.C,        f"expected C={self.C}, got {C}"
        assert T % self.p_t == 0,  f"T={T} not divisible by p_t={self.p_t}"
        assert H % self.p_h == 0,  f"H={H} not divisible by p_h={self.p_h}"
        assert W % self.p_w == 0,  f"W={W} not divisible by p_w={self.p_w}"
        return B, C, T, H, W

    def grid_shape(self, T: int, H: int, W: int) -> tuple[int, int, int]:
        """Number of patches along each axis: (nT, nH, nW)."""
        return T // self.p_t, H // self.p_h, W // self.p_w

    # ── main ops ──────────────────────────────────────────────────────────────

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """
        [B, C, T, H, W] → [B, N, D]

        Step-by-step:
          1. reshape T → (nT, p_t), H → (nH, p_h), W → (nW, p_w) # example: [B, C, 4, 16, 16] → [B, C, 4//1=4, 1, 16//2=8, 2, 16//2=8, 2]
          2. move all patch-interior dims (p_t, p_h, p_w, C) together
          3. flatten into a single vector D = p_t*p_h*p_w*C per token
          4. flatten the (nT, nH, nW) grid into N tokens
        """
        B, C, T, H, W = self._check(z)
        nT, nH, nW = self.grid_shape(T, H, W)
        pt, ph, pw = self.p_t, self.p_h, self.p_w

        # [B, C, nT, pt, nH, ph, nW, pw]
        x = z.reshape(B, C, nT, pt, nH, ph, nW, pw)
        # [B, nT, nH, nW, pt, ph, pw, C]  — group grid dims first, patch dims last
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        # [B, nT*nH*nW, pt*ph*pw*C]  =  [B, N, D]
        x = x.reshape(B, nT * nH * nW, pt * ph * pw * C)
        return x

    def unpatchify(
        self,
        tokens: torch.Tensor,
        T: int, H: int, W: int,
    ) -> torch.Tensor:
        """
        [B, N, D] → [B, C, T, H, W]

        Exact inverse of patchify().  T, H, W are the original latent dims
        (before patching), needed to reconstruct the correct grid shape.
        """
        B, N, D = tokens.shape
        nT, nH, nW = self.grid_shape(T, H, W)
        pt, ph, pw, C = self.p_t, self.p_h, self.p_w, self.C
        assert D == pt * ph * pw * C, f"D mismatch: got {D}, expected {pt*ph*pw*C}"
        assert N == nT * nH * nW,     f"N mismatch: got {N}, expected {nT*nH*nW}"

        # [B, nT, nH, nW, pt, ph, pw, C]
        x = tokens.reshape(B, nT, nH, nW, pt, ph, pw, C)
        # [B, C, nT, pt, nH, ph, nW, pw]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # [B, C, T, H, W]
        x = x.reshape(B, C, T, H, W)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience: same as patchify()."""
        return self.patchify(z)


if __name__ == "__main__":
    B, C, T, H, W = 2, 16, 4, 16, 16

    p3d = Patchify3D(patch_size=(1, 2, 2), latent_channels=C)

    z      = torch.randn(B, C, T, H, W)
    tokens = p3d.patchify(z)
    z_back = p3d.unpatchify(tokens, T, H, W)

    nT, nH, nW = p3d.grid_shape(T, H, W)
    N = nT * nH * nW
    D = p3d.D

    print(f"latent  : {tuple(z.shape)}")
    print(f"tokens  : {tuple(tokens.shape)}   (N={N}, D={D})")
    print(f"restored: {tuple(z_back.shape)}")

    assert tokens.shape == (B, N, D),          f"bad token shape: {tokens.shape}"
    assert z_back.shape == z.shape,            f"bad restored shape: {z_back.shape}"
    assert torch.allclose(z, z_back, atol=1e-6), "unpatchify is not exact inverse!"

    print("all assertions passed")
