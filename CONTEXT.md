# Cosmos Scratch Reimplementation Context

## Goal

I am **not** trying to reproduce full NVIDIA Cosmos at production scale.

I am building a **small educational reimplementation** to understand the paper from first principles. The learning order is:

1. dataset prep
2. **tokenizer v1**
3. diffusion world model
4. Video2World conditioning
5. autoregressive world model
6. optional extras later

The real paper is a **platform** with video curation, tokenizers, two pre-trained WFM families, post-training, and guardrails. It uses **two parallel WFM families**: diffusion and autoregressive. The world model interface is: past visual observations plus a perturbation/control signal produce a future observation. In the paper, observations are RGB video, and perturbations can be actions, prompts, trajectories, or instructions.  

## What we are implementing right now

### Current scope: `tokenizer_v1`

I only want a **minimal continuous video tokenizer** that trains and reconstructs short clips.

This is **not** the full Cosmos tokenizer yet.

### Tokenizer v1 requirements

* input: short video clips
* output: reconstructed clips
* latent: **continuous**
* architecture: **encoder-decoder**
* temporal behavior: **causal**
* loss: start with **L1**, optionally add perceptual loss
* no discrete quantization yet
* no wavelet yet
* no causal attention yet
* no diffusion model yet
* no autoregressive model yet

This is still faithful to the paper’s core tokenizer idea, because Cosmos tokenization is an encoder-decoder; continuous tokenization uses a vanilla autoencoder latent space; and the tokenizer is designed to be **temporally causal**. The paper also trains tokenizers by supervising the **final decoder output**, with L1 and perceptual loss in the first stage.  

## Why tokenizer first

Cosmos does **not** train the world model directly on raw RGB video. It first learns a tokenizer that maps raw visual data into **compact tokens**, because video is too large and redundant to model directly. The tokenizer is one of the core platform components, and the paper explicitly says both diffusion and autoregressive WFMs depend on video tokenization. Continuous tokenizers are used for diffusion models, and discrete tokenizers are used for GPT-style autoregressive models.  

## What the full paper does, for reference

### Data pipeline

The full paper uses a curated video pipeline:

* split into shot-consistent clips
* filter bad or low-value clips
* caption clips with a VLM
* deduplicate semantically
* shard into training-ready data 

### Tokenizer

The full paper’s tokenizer:

* is **temporally causal**
* uses a **2-level 3D wavelet transform**
* uses factorized spatiotemporal conv blocks
* uses causal self-attention
* has continuous and discrete variants
* uses latent dim **16** for continuous tokenizers
* uses **FSQ** for discrete tokenizers, with latent dim **6** and levels `(8,8,8,5,5,5)`  

### Two WFM families

The paper keeps two parallel pre-trained branches:

* **diffusion**: `Text2World -> Video2World`, using continuous tokens
* **autoregressive**: `next-video-token model -> Video2World`, using discrete tokens
  It also adds:
* a **prompt upsampler** for diffusion
* a **diffusion decoder** to sharpen autoregressive outputs  

## Current dataset status

I already downloaded the **BridgeData V2 scripted video subset** and will use that for now.

Important:

* For **Day 1 / tokenizer v1**, I only want to use the **video subset**
* I am **ignoring actions and text for now**
* I am also **ignoring the image subset for now**
* The image subset had metadata/loading issues, so do not rely on it for tokenizer v1

### Dataset preparation assumptions

Use a small, simple setup:

* fixed clip length: `T = 8` first
* fixed resolution: `64x64` first
* RGB only
* normalize to `[0,1]` or `[-1,1]`
* split into `train` and `val`
* dataloader should return shape:

```text
[B, T, 3, H, W]
```

Before any training:

* inspect samples visually
* verify consistent shapes
* verify clip ordering is correct
* verify clips are not corrupted

## Tokenizer v1 architecture

### Input

```text
x: [B, T, 3, H, W]
```

Start with:

* `T = 8`
* `H = W = 64`

### Output

```text
x_hat: [B, T, 3, H, W]
```

### Latent

Use a continuous latent with:

* spatial compression only at first
* latent channel dimension `C = 16`

Target latent shape:

```text
z: [B, T, 16, 16, 16]
```

That means:

* spatial compression = `4x`
* temporal compression = `1x` for v1

This matches the paper’s continuous tokenizer spirit, especially the use of **continuous latent dimension 16**, but keeps implementation simple. 

### Minimal encoder

Use a small causal 3D encoder:

* causal 3D conv
* residual block
* spatial downsample
* residual block
* spatial downsample
* residual block
* projection to latent channels

### Minimal decoder

Mirror the encoder:

* projection from latent channels
* residual block
* spatial upsample
* residual block
* spatial upsample
* residual block
* RGB output projection

### Causality rule

The temporal path must be **causal**:

* frame `t` may depend on frames `<= t`
* frame `t` must **not** depend on future frames

The paper enforces this with factorized temporal conv and **left padding** in time, so earlier states never see future frames. For tokenizer v1, implement the same causal principle even if the architecture is simpler than the paper’s final tokenizer. 

## Tokenizer v1 equations

### Encode / decode

[
z = \mathcal{E}(x), \qquad \hat{x} = \mathcal{D}(z)
]

### Reconstruction

[
\hat{x}*{0:T} = \mathcal{D}(\mathcal{E}(x*{0:T}))
]

This is the same high-level tokenizer equation used in the paper. 

### L1 loss

[
\mathcal{L}*{1} = |\hat{x}*{0:T} - x_{0:T}|_1
]

This is the paper’s first tokenizer loss. 

### Optional perceptual loss

[
\mathcal{L}*{\text{perc}} =
\frac{1}{L}\sum*{l=1}^{L}\sum_t \alpha_l
|VGG_l(\hat{x}_t)-VGG_l(x_t)|_1
]

This is also part of the paper’s stage-1 tokenizer training. 

### Final loss for v1

Start with:
[
\mathcal{L} = \mathcal{L}_1
]

Optionally later:
[
\mathcal{L} = \mathcal{L}*1 + \lambda \mathcal{L}*{\text{perc}}
]

Do **not** add flow loss, Gram loss, or adversarial loss yet. Those belong to later stages in the paper. 

## What tokenizer v1 is **not**

Tokenizer v1 is **not yet**:

* wavelet-based
* temporally compressed
* attention-based
* discrete
* FSQ-based
* diffusion-ready in full Cosmos form

Those belong to later steps.

## Coding priorities right now

### First deliverable

A working tokenizer that can:

* overfit **one batch**
* overfit **10–20 clips**
* reconstruct recognizable motion and object layout
* save input vs reconstruction visualizations
* save a checkpoint

### Second deliverable

A causality sanity check:

* modify only future frames in one clip
* confirm latent outputs for earlier timesteps do not change

### Third deliverable

A clean code layout with separate modules:

* dataset
* causal conv
* residual block
* tokenizer model
* training loop
* visualization utilities

## Preferred file structure

```text
cosmos_scratch/
  data/
  outputs/
  scripts/
    train_tokenizer_v1.py
    recon_tokenizer_v1.py
    overfit_one_batch.py
  cosmos/
    datasets/
    tokenizers/
      causal_conv.py
      blocks.py
      tokenizer_v1.py
      losses.py
    utils/
      viz.py
```

## Implementation constraints

When modifying code, follow these rules:

1. **Do not jump ahead to full Cosmos**

   * no diffusion model yet
   * no autoregressive model yet
   * no FSQ yet
   * no wavelet yet unless explicitly requested

2. **Prefer simplicity over faithfulness**

   * we want a working educational baseline first

3. **Keep tensor shapes explicit**

   * document every major tensor shape in code comments

4. **Keep causality correct**

   * if unsure, choose simpler causal temporal convs over fancy blocks

5. **Make debugging easy**

   * log losses
   * save reconstructions
   * print latent shapes
   * keep configs small

6. **Use PyTorch**

   * clean, readable modules
   * minimal abstractions
   * good defaults
   * no unnecessary framework complexity

7. **Do not introduce scale-only tricks yet**

   * no distributed training
   * no mixed-precision complexity unless needed
   * no FSDP / TP / SP / CP

## What success looks like today

Today is successful if we have:

* a dataloader returning `[B,T,3,H,W]`
* a working `TokenizerV1`
* L1 training that reduces reconstruction loss
* saved recon examples
* at least one checkpoint
* a passed causality sanity check

## Next steps after tokenizer v1

After tokenizer v1 works, the next roadmap is:

1. add perceptual loss if missing
2. improve causal blocks
3. optionally add wavelet front-end
4. then build a **small diffusion latent model**
5. then add **Video2World conditioning**
6. only later build the discrete tokenizer and autoregressive branch

This ordering matches the paper’s logic: tokenizer first, then world model in token space, then conditioning, then the second model family.  

## Short glossary

**WFM**
World Foundation Model: predicts future observations from past observations plus perturbation/control. 

**obs**
Past RGB video.

**perturbation / control**
Prompt, action, instruction, trajectory, etc. 

**continuous tokenizer**
Encodes video into continuous latent vectors; used by diffusion WFMs. 

**discrete tokenizer**
Encodes video into discrete indices; used by autoregressive WFMs. 

**causal**
No future information leakage.

**Video2World**
Predict future video from past video plus control/prompt. 
