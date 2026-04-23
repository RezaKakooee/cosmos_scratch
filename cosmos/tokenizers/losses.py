import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared VGG feature extractor ─────────────────────────────────────────────

class VGGFeatures(nn.Module):
    """
    Frozen VGG16 feature extractor shared by PerceptualLoss and GramLoss.

    We slice the VGG feature tower into three segments and capture the
    activation after each ReLU group.  These intermediate feature maps
    encode progressively more abstract visual content:
      slice1 → relu1_2 : edges / colours  (low-level)
      slice2 → relu2_2 : textures          (mid-level)
      slice3 → relu3_3 : object parts      (high-level)

    Using multiple levels gives a richer perceptual signal than a single layer.

    All VGG weights are frozen — we never want gradients flowing into VGG,
    only back into the reconstructed frames that are fed into it.

    Input: [B, 3, H, W] in [-1, 1].
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models

        # Load pretrained VGG16 and take only the convolutional feature layers
        # (drop the final classifier head — we only need feature maps).
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        # Split the sequential tower into three named slices so we can
        # read off feature maps at each depth.
        self.slice1 = nn.Sequential(*list(vgg)[:4])    # up to relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9])   # up to relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:16])  # up to relu3_3

        # Freeze all parameters — VGG is a fixed perceptual judge, not trainable.
        for p in self.parameters():
            p.requires_grad_(False)

        # VGG was trained on ImageNet with this specific normalisation.
        # We store it as a buffer so it moves to the right device automatically.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Our model produces outputs in [-1, 1]; VGG expects ImageNet-normalised input.
        # Step 1: [-1, 1] → [0, 1]
        # Step 2: [0, 1]  → ImageNet z-score
        x = (x * 0.5 + 0.5 - self.mean) / self.std   # [B, 3, H, W]

        f1 = self.slice1(x)   # [B, 64,  H/1, W/1]  — relu1_2
        f2 = self.slice2(f1)  # [B, 128, H/2, W/2]  — relu2_2
        f3 = self.slice3(f2)  # [B, 256, H/4, W/4]  — relu3_3
        return [f1, f2, f3]


# ── individual losses ─────────────────────────────────────────────────────────

class ReconLoss(nn.Module):
    """
    Pixel-level L1 reconstruction loss.

    Mean absolute error between every pixel of x_hat and x.
    This is the primary training signal — it pushes the decoder output
    toward the correct pixel values frame by frame.

    L1 = mean |x_hat - x|

    We prefer L1 over MSE here because L1 is less sensitive to outlier
    pixels and tends to produce sharper reconstructions.

    Input: [B, C, T, H, W] in [-1, 1].
    """

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(x_hat, x)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss averaged over all T frames.

    Why perceptual loss?
      Pixel-level L1 penalises each pixel independently.  A slightly shifted
      reconstruction can have high L1 loss even if it looks correct to a human.
      Perceptual loss instead compares *feature maps* — if two images activate
      the same VGG neurons, they look similar to a perceptual system.

    How it works:
      For each frame t, extract VGG features from x_hat_t and x_t at three
      depths (relu1_2, relu2_2, relu3_3), then take the mean L1 distance
      across all feature map entries.  Average over all frames and layers.

    L_perc = (1 / T·L) ∑_t ∑_l ‖VGG_l(x̂_t) - VGG_l(x_t)‖₁

    Input: [B, 3, T, H, W] in [-1, 1].
    """

    def __init__(self, vgg: VGGFeatures):
        super().__init__()
        # Receive the shared VGG extractor — avoids loading VGG twice when
        # both PerceptualLoss and GramLoss are active.
        self.vgg = vgg

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        frame_losses = []
        for t in range(T):
            # Extract features from one frame at a time: [B, 3, H, W]
            feats_hat = self.vgg(x_hat[:, :, t])
            feats_x   = self.vgg(x[:, :, t])
            # L1 distance at each VGG depth, then average across depths
            frame_losses.append(
                torch.stack([F.l1_loss(fh, fr) for fh, fr in zip(feats_hat, feats_x)]).mean()
            )
        # Average the per-frame losses into one scalar
        return torch.stack(frame_losses).mean()


def _gram(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalised Gram matrix of a feature map.

    The Gram matrix G_{ij} = dot(channel_i, channel_j) captures which pairs
    of feature channels co-activate — i.e., the *texture statistics* of the
    image rather than the spatial layout of objects.

    feat : [B, C, H, W]
    return: [B, C, C]  — one C×C matrix per batch item
    """
    B, C, H, W = feat.shape
    # Flatten spatial dims so each channel becomes a vector of length H*W
    f = feat.view(B, C, -1)                            # [B, C, H*W]
    # Batch matrix multiply (torch.bmm): G = F · Fᵀ, then normalise so scale is independent
    # of feature map resolution and channel count
    # f : [B, C, H*W] -> f.transpose(1, 2) : [B, H*W, C] =  swaps the last two dimensions
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)   # [B, C, C] 


class GramLoss(nn.Module):
    """
    Gram matrix (style) loss averaged over all T frames.

    Why Gram loss?
      PerceptualLoss compares *where* features activate (spatial layout).
      Gram loss compares *which* features co-activate (texture / style).
      Together they push the reconstruction to match both content and texture.

    How it works:
      For each frame t and each VGG depth l, compute the Gram matrix of both
      the real and reconstructed feature maps, then take their L1 distance.

    L_gram = (1 / T·L) ∑_t ∑_l ‖GM_l(x̂_t) - GM_l(x_t)‖₁

    Input: [B, 3, T, H, W] in [-1, 1].
    """

    def __init__(self, vgg: VGGFeatures):
        super().__init__()
        self.vgg = vgg

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        frame_losses = []
        for t in range(T):
            feats_hat = self.vgg(x_hat[:, :, t])
            feats_x   = self.vgg(x[:, :, t])
            # Compare Gram matrices (not raw features) at each depth
            frame_losses.append(
                torch.stack([F.l1_loss(_gram(fh), _gram(fr)) for fh, fr in zip(feats_hat, feats_x)]).mean()
            )
        return torch.stack(frame_losses).mean()


class OpticalFlowLoss(nn.Module):
    """
    Temporal motion-consistency loss.

    Why a flow loss?
      L1 and perceptual losses are applied frame-by-frame independently.
      They do not penalise inconsistent *motion* between frames — a model
      could reconstruct each frame well individually but produce flickering
      or wrong motion trajectories.  The flow loss explicitly penalises
      motion mismatches.

    Implementation:
      We use frame differences as a differentiable proxy for optical flow.
      True optical flow (e.g. RAFT) would be more accurate but is expensive
      and has minimum-size requirements.  Frame differences capture the same
      first-order motion signal and gradients flow cleanly back to x_hat.

      Forward diff  Δfwd_t  = x_t   - x_{t-1}   (where is each pixel going?)
      Backward diff Δbwd_t  = x_t   - x_{t+1}   (where did each pixel come from?)

      We match both directions and then average them:
      L_flow = 0.5 · (L1(Δfwd_hat, Δfwd_x) + L1(Δbwd_hat, Δbwd_x))

    Input: [B, 3, T, H, W] in [-1, 1].
    """

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Forward temporal difference: motion from t-1 → t
        fwd_hat = x_hat[:, :, 1:]  - x_hat[:, :, :-1]   # [B, 3, T-1, H, W]
        fwd_x   = x[:, :, 1:]     - x[:, :, :-1]

        # Backward temporal difference: motion from t+1 → t
        bwd_hat = x_hat[:, :, :-1] - x_hat[:, :, 1:]
        bwd_x   = x[:, :, :-1]    - x[:, :, 1:]

        return 0.5 * (F.l1_loss(fwd_hat, fwd_x) + F.l1_loss(bwd_hat, bwd_x))


# ── combined loss ─────────────────────────────────────────────────────────────

class TokenizerLoss(nn.Module):
    """
    Combined tokenizer reconstruction loss for Day 2.

    Assembles all active loss terms into one weighted sum:

      L = λ_l1  · L1
        + λ_perc · L_perc   (if use_perceptual)
        + λ_flow · L_flow   (if use_flow)
        + λ_gram · L_gram   (if use_gram)

    Design notes:
    - VGG is instantiated once and shared between PerceptualLoss and GramLoss
      to avoid loading the weights twice.
    - Each term can be toggled independently so you can start with L1 only
      and layer in complexity one loss at a time.
    - Default weights come from the Day 2 spec:
        λ_l1=1.0, λ_perc=0.1, λ_flow=0.05, λ_gram=0.02

    Returns:
        (total_loss, loss_dict)
        loss_dict keys: "l1", "perceptual"*, "flow"*, "gram"*, "total"
        (* present only when the corresponding term is active)
    """

    def __init__(
        self,
        use_perceptual: bool  = True,
        use_flow:       bool  = True,
        use_gram:       bool  = True,
        lambda_l1:      float = 1.0,
        lambda_perc:    float = 0.1,
        lambda_flow:    float = 0.05,
        lambda_gram:    float = 0.02,
    ):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_flow = lambda_flow
        self.lambda_gram = lambda_gram

        self.recon = ReconLoss()
        self.flow  = OpticalFlowLoss() if use_flow else None

        # Instantiate VGG once and share it if both perceptual and Gram are active
        if use_perceptual or use_gram:
            vgg = VGGFeatures()
            self.perceptual = PerceptualLoss(vgg) if use_perceptual else None
            self.gram       = GramLoss(vgg)       if use_gram       else None
        else:
            self.perceptual = None
            self.gram       = None

    def forward(
        self, x_hat: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # L1 is always active — it is the primary pixel-level signal
        l1   = self.recon(x_hat, x)
        loss = self.lambda_l1 * l1
        loss_dict: dict[str, float] = {"l1": l1.item()}

        if self.perceptual is not None:
            lp   = self.perceptual(x_hat, x)
            loss = loss + self.lambda_perc * lp
            loss_dict["perceptual"] = lp.item()

        if self.flow is not None:
            lf   = self.flow(x_hat, x)
            loss = loss + self.lambda_flow * lf
            loss_dict["flow"] = lf.item()

        if self.gram is not None:
            lg   = self.gram(x_hat, x)
            loss = loss + self.lambda_gram * lg
            loss_dict["gram"] = lg.item()

        loss_dict["total"] = loss.item()
        return loss, loss_dict


if __name__ == "__main__":
    B, C, T, H, W = 2, 3, 8, 64, 64
    x     = torch.randn(B, C, T, H, W).clamp(-1, 1)
    x_hat = torch.randn(B, C, T, H, W).clamp(-1, 1)

    # L1 only — fastest, used for early overfit checks
    crit = TokenizerLoss(use_perceptual=False, use_flow=False, use_gram=False)
    loss, d = crit(x_hat, x)
    print(f"L1 only  → total={d['total']:.4f}  keys={list(d)}")

    # L1 + flow — adds motion consistency without loading VGG
    crit = TokenizerLoss(use_perceptual=False, use_flow=True, use_gram=False)
    loss, d = crit(x_hat, x)
    print(f"L1+flow  → total={d['total']:.4f}  keys={list(d)}")

    # All losses active — full Day 2 training signal
    crit = TokenizerLoss(use_perceptual=True, use_flow=True, use_gram=True)
    loss, d = crit(x_hat, x)
    print(f"all      → total={d['total']:.4f}  keys={list(d)}")
    assert set(d) == {"l1", "perceptual", "flow", "gram", "total"}
    print("all assertions passed")
