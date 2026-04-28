# TASK_DAY5_DIFFUSION_WFM.md

## Goal

Start the **diffusion World Foundation Model** on top of the continuous tokenizer.

Assume tokenizer is now strong enough:
- wavelet front-end
- causal conv blocks
- causal attention
- spatial + temporal compression
- continuous latent bottleneck

Day 5 should implement the **first usable latent diffusion backbone**.

Do **not** do autoregressive yet.

---

## Scope

### In scope
- freeze or mostly freeze the continuous tokenizer first
- encode video into continuous latents
- add Gaussian noise in latent space
- build the diffusion denoiser
- 3D patchify latent video
- time-step conditioning
- text conditioning via cross-attention
- first target = **Text2World**
- second target if time allows = **1-frame Video2World**

### Out of scope
- discrete tokenizer
- FSQ
- autoregressive GPT video model
- diffusion decoder
- prompt upsampler
- post-training for robotics / driving

---

## Day 5 target

By the end of Day 5, I want:

- tokenizer latents are usable as diffusion inputs
- a DiT-style latent denoiser runs end-to-end
- EDM-style loss works
- text-conditioned sampling works at least weakly
- saved samples and checkpoints exist

If time allows:
- add **1-frame Video2World** conditioning

---

## Recommended tensor setup

### Input video
```text
x: [B, 3, 16, 64, 64]
```

### Tokenizer latent
```text
z: [B, 16, 4, 16, 16]
```

### Patchified latent sequence
Choose patch sizes:
- `p_t = 1`
- `p_h = 2`
- `p_w = 2`

Then:
```text
z_patches: [B, N, D]
```

where:
- `N = T' * H' * W' / (p_t * p_h * p_w)`
- `D = patch_volume * latent_channels`

With:
- `T' = 4`
- `H' = W' = 16`
- patch `(1,2,2)`

you get:
- `N = 4 * 8 * 8 = 256`
- `D = 1 * 2 * 2 * 16 = 64`

So a good Day 5 sequence target is:
```text
[B, 256, 64] -> project -> [B, 256, d_model]
```

---

## Main architecture

```text
Input video
-> continuous tokenizer encoder
-> latent video z
-> add Gaussian noise
-> 3D patchify
-> DiT-style transformer blocks
-> unpatchify
-> tokenizer decoder
-> reconstructed / generated video
```

---

## Core equations

### Tokenizer encode / decode
\[
z = E(x), \qquad \hat{x} = D(z)
\]

### Diffusion corruption
\[
z_\sigma = z + n, \qquad n \sim \mathcal{N}(0, \sigma^2 I)
\]

### EDM denoising loss
\[
L(D_\theta, \sigma) =
\mathbb{E}_{z,n}
\left[
||D_\theta(z + n; \sigma) - z||_2^2
\right]
\]

### Full EDM-style objective
\[
L(D_\theta) =
\mathbb{E}_{\sigma}
\left[
\frac{\lambda(\sigma)}{e^{u(\sigma)}} L(D_\theta, \sigma) + u(\sigma)
\right]
\]

with

\[
\lambda(\sigma)=\frac{\sigma^2 + \sigma_{data}^2}{(\sigma \cdot \sigma_{data})^2}
\]

You can implement the simple version first:
\[
L = ||D_\theta(z+n;\sigma) - z||_2^2
\]

and add uncertainty weighting later.

---

## Conditioning plan

## Text conditioning
Use cross-attention from latent tokens to text embeddings.

Keep it simple:
- frozen text encoder first
- CLIP text or T5 encoder if available
- project text embeddings to model dimension
- cross-attention in each block or every few blocks

### First target
**Text2World**
- input: text prompt
- output: generated latent video -> decoded video

### If time allows
**Video2World**
- input: 1 conditioning frame + prompt
- output: future video

Do not jump to long conditioning sequences yet.

---

## Recommended module layout

```text
cosmos/diffusion/
  patchify3d.py
  rope3d.py
  timestep_embed.py
  text_encoder.py
  dit_block.py
  model.py
  edm_loss.py
  scheduler.py
  sample.py
```

---

## Core modules

## 1. Patchify3D
Convert latent video into transformer tokens.

Need:
- `patchify(z) -> tokens`
- `unpatchify(tokens) -> latent video`

Use patch sizes:
- `p_t = 1`
- `p_h = 2`
- `p_w = 2`

---

## 2. 3D positional encoding
Implement 3D positional encoding over:
- time
- height
- width

Use a simple version first:
- factorized sinusoidal or RoPE-style encoding

Need one embedding per token position in latent patch space.

---

## 3. Time-step embedding
Need a timestep / noise-level embedding.

Simple version:
- sinusoidal noise embedding
- small MLP
- inject into each transformer block

This tells the denoiser how noisy the latent is.

---

## 4. Text encoder wrapper
Need a small wrapper that returns:
- text embeddings
- padding mask if needed

Keep the API simple.

---

## 5. DiT block
Each block should look like:

```text
LayerNorm / AdaLN
-> self-attention
-> cross-attention to text
-> MLP
```

Minimum acceptable version:
- self-attention
- cross-attention
- feed-forward
- residual connections
- timestep conditioning

If AdaLN is too much initially, use:
- standard LayerNorm + additive timestep conditioning

But keep code ready to upgrade.

---

## 6. Diffusion model
Overall model input:
- noisy latent video
- noise level / timestep
- text embedding

Overall model output:
- predicted clean latent video

---

## Training plan

## Stage A — freeze tokenizer
First, **freeze tokenizer encoder and decoder**.
Do not jointly train tokenizer and diffusion on Day 5.

Reason:
- isolate diffusion debugging
- keep latent target stable

## Stage B — latent overfit
Overfit diffusion on:
- one batch
- very small text set

Goal:
- denoiser can reconstruct clean latents from noisy latents

## Stage C — full small run
Train on the small dataset with:
- fixed video size
- fixed clip length
- short prompts

---

## Recommended exact work order

1. freeze tokenizer and verify latent shapes
2. implement `Patchify3D`
3. implement timestep embedding
4. implement text encoder wrapper
5. implement one DiT block
6. build full diffusion model
7. implement simple EDM loss
8. overfit one batch
9. save latent-space denoising outputs
10. decode samples to RGB video
11. run short training
12. if stable, add 1-frame Video2World mask/conditioning

---

## Suggested training defaults

- batch size: as small as needed
- learning rate: `1e-4` or `2e-4`
- optimizer: `AdamW`
- loss: simple denoising MSE first
- train short and inspect samples often
- save:
  - `latest`
  - `best`
  - fixed prompt samples

---

## Logging

Track:
- `denoise_loss`
- `sigma_mean`
- `grad_norm` if easy
- sample saves every epoch or every N steps

Save:
- noisy latent recon
- clean latent recon
- final decoded sample video

---

## Acceptance checklist

- [ ] tokenizer latents feed into diffusion cleanly
- [ ] patchify / unpatchify works
- [ ] timestep conditioning implemented
- [ ] text conditioning implemented
- [ ] DiT-style denoiser runs
- [ ] simple EDM loss works
- [ ] one-batch overfit works
- [ ] decoded RGB samples saved
- [ ] checkpoint saved

---

## Definition of done

Day 5 is done when there is a **working latent diffusion WFM backbone** that:
- takes tokenizer latents
- adds noise
- denoises with a transformer
- conditions on text
- decodes back to video

Then the next step can be:
- proper **Video2World conditioning**
- better EDM weighting
- stronger positional encoding
- improved sampling
