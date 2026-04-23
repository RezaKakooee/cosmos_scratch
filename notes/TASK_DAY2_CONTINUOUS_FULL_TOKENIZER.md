# TASK_DAY2_CONTINUOUS_FULL_TOKENIZER.md

## Goal

Implement a **much more faithful Cosmos continuous tokenizer**, not an educational toy:

- **3D wavelet front-end**
- **causal factorized spatiotemporal conv blocks**
- **causal spatiotemporal attention**
- **encoder-decoder hierarchy**
- **spatial + temporal compression**
- **continuous latent bottleneck only**
- **stage-1 + stage-2 tokenizer losses**

Do **not** start the WFM yet.  
Do **not** implement the discrete / FSQ branch yet.

---

## What to match from the paper

Continuous tokenizer should have:

- **encoder-decoder**
- **temporally causal**
- **3D wavelet preprocessing**
- **factorized spatial then temporal conv**
- **causal spatiotemporal attention**
- **continuous latent dim = 16**
- training with:
  - **L1**
  - **perceptual**
  - **optical flow**
  - **Gram loss**
- optional adversarial loss can wait

This should now feel like a **real backbone for diffusion WFM training**.

---

## Scope

### In scope
- full **continuous tokenizer**
- wavelet + inverse wavelet
- encoder / decoder
- causal residual blocks
- causal downsample / upsample
- causal attention blocks
- temporal compression
- full reconstruction loss stack except GAN
- good logging, recon saves, causality tests

### Out of scope
- discrete tokenizer
- FSQ
- diffusion WFM
- autoregressive WFM
- prompt conditioning
- distributed scaling tricks

---

## Recommended tensor setup

### Safer version
Input:
```text
x: [B, 3, 16, 64, 64]
```

Target latent:
```text
z: [B, 16, 4, 16, 16]
```

Meaning:
- spatial compression = **4x**
- temporal compression = **4x**
- latent channels = **16**

### If stable and GPU allows
Input:
```text
x: [B, 3, 32, 128, 128]
```

Target latent:
```text
z: [B, 16, 8, 32, 32]
```

Start with the safer version.

---

## Main architecture

```text
Input video
-> Wavelet3D
-> Encoder stage 1
-> Encoder stage 2
-> Encoder stage 3
-> Latent projection
-> Decoder stage 1
-> Decoder stage 2
-> Decoder stage 3
-> InverseWavelet3D
-> Reconstructed video
```

---

## Recommended module layout

```text
cosmos/tokenizers/
  wavelet.py
  causal_conv.py
  attention.py
  blocks.py
  encoder.py
  decoder.py
  continuous_tokenizer.py
  losses.py
```

---

## Core modules

## 1. Wavelet front-end

Implement:

- `Wavelet3D`
- `InverseWavelet3D`

Goal:
- remove low-level pixel redundancy early
- let later layers focus on semantic compression
- operate on a more compact representation before heavy learned blocks

If exact Haar wavelet is hard, use a fixed Haar-style implementation, not a loose placeholder.

---

## 2. Factorized causal conv

Implement a real factorized block:

- spatial conv: kernel `(1, k, k)`
- temporal conv: kernel `(k, 1, 1)` with **left-only temporal padding**

This is the main conv primitive.

Suggested module:
- `FactorizedCausalConv3d`

---

## 3. Residual block

Use:

```text
Norm
-> SiLU
-> factorized causal conv
-> Norm
-> SiLU
-> factorized causal conv
-> residual add
```

Keep channels constant inside the block.

Use `GroupNorm` first if easiest.  
LayerNorm-style can come later if needed.

---

## 4. Attention block

Implement **causal spatiotemporal attention**.

Minimum acceptable version:
- flatten `(T, H, W)` into a sequence
- apply causal mask on time
- attention should not let tokens attend to future timesteps
- use it only in deeper stages / bottleneck if memory is a concern

Suggested module:
- `CausalSpatiotemporalAttention`

Use a config flag:
```yaml
use_attention: true
```

---

## 5. Downsampling / upsampling

Need both spatial and temporal compression now.

Implement:

- `DownsampleSpatial`
- `DownsampleTemporal`
- `UpsampleSpatial`
- `UpsampleTemporal`

Or combined modules if cleaner.

Recommendation:
- compress space early
- compress time deeper in encoder
- mirror this in decoder

---

## Suggested encoder

```text
Wavelet3D
-> ResBlock
-> DownsampleSpatial
-> ResBlock
-> CausalAttention
-> DownsampleSpatial + DownsampleTemporal
-> ResBlock
-> CausalAttention
-> latent projection to C = 16
```

---

## Suggested decoder

```text
latent projection
-> ResBlock
-> CausalAttention
-> UpsampleSpatial + UpsampleTemporal
-> ResBlock
-> CausalAttention
-> UpsampleSpatial
-> ResBlock
-> InverseWavelet3D
-> RGB output
```

---

## Core equations

### Encode / decode
\[
z = E(x), \qquad \hat{x} = D(z)
\]

### Reconstruction
\[
\hat{x}_{0:T} = D(E(x_{0:T}))
\]

### L1 loss
\[
L_1 = ||\hat{x} - x||_1
\]

### Perceptual loss
For each frame:
\[
L_{perc} = \frac{1}{L} \sum_l \alpha_l ||VGG_l(\hat{x}_t) - VGG_l(x_t)||_1
\]

Then average over time.

### Optical flow loss
Match motion between neighboring reconstructed and real frames:
\[
L_{flow}
=
\frac{1}{T}\sum_{t=1}^{T} ||OF(\hat{x}_t, \hat{x}_{t-1}) - OF(x_t, x_{t-1})||_1
+
\frac{1}{T}\sum_{t=0}^{T-1} ||OF(\hat{x}_t, \hat{x}_{t+1}) - OF(x_t, x_{t+1})||_1
\]

### Gram loss
Use feature Gram matrices for sharper texture:
\[
L_{gram} = \frac{1}{L} \sum_l \alpha_l ||GM_l(\hat{x}_t) - GM_l(x_t)||_1
\]

### Combined tokenizer loss
For now:
\[
L = \lambda_1 L_1 + \lambda_{perc} L_{perc} + \lambda_{flow} L_{flow} + \lambda_{gram} L_{gram}
\]

Suggested starting weights:
- `lambda_1 = 1.0`
- `lambda_perc = 0.1`
- `lambda_flow = 0.05`
- `lambda_gram = 0.02`

Tune if needed.

---

## Training requirements

- keep clips fixed length
- save train / val losses
- save reconstructions every epoch
- keep one fixed validation batch for visual comparison
- overfit one batch after every major refactor
- keep checkpoints:
  - `latest`
  - `best_val`

Return a clear `loss_dict`:
- `l1`
- `perceptual`
- `flow`
- `gram`
- `total`

---

## Causality requirements

This must still hold:

For any time `t`, encoded outputs up to `t` must not change if frames after `t` are modified.

Run this test after:
- wavelet integration
- attention integration
- temporal downsampling integration

---

## Exact implementation order

1. refactor Day 1 code into modules
2. implement `Wavelet3D` and `InverseWavelet3D`
3. implement `FactorizedCausalConv3d`
4. upgrade `ResBlock3D`
5. add spatial + temporal down/up modules
6. build new encoder / decoder
7. reconnect training with L1 only first
8. restore perceptual loss
9. add flow loss
10. add Gram loss
11. add causal attention
12. run short full training
13. run causality regression test
14. compare Day 1 vs Day 2 outputs

---

## Acceptance checklist

- [ ] wavelet modules implemented
- [ ] factorized causal conv implemented
- [ ] causal attention implemented
- [ ] spatial + temporal compression implemented
- [ ] continuous latent dim = 16
- [ ] encoder-decoder trains end-to-end
- [ ] L1 + perceptual + flow + Gram losses work
- [ ] reconstructions saved
- [ ] causality test passes
- [ ] checkpoint saved

---

## Practical guidance

- prefer a **real implementation** over placeholders now
- keep tensor shapes commented everywhere
- prioritize **causality correctness** over cleverness
- add one feature at a time and restore one-batch overfit each time
- if attention is memory-heavy, keep it only in deep stages first

---

## Definition of done

Day 2 is done when the tokenizer is a **full continuous Cosmos-style tokenizer backbone**:
- wavelet-based
- causal
- factorized spatiotemporal
- attention-equipped
- compressed in space and time
- trained with the main tokenizer losses

Then Day 3 can start the **diffusion WFM** on top of this tokenizer.
