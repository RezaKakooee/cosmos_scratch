# TASK_DAY1.md

## Objective

Build a **working tokenizer v1** for short robot-manipulation video clips.

By the end of Day 1, we want:
- a dataset loader that returns tensors shaped `[B, T, 3, H, W]`
- a **continuous causal video autoencoder**
- a training loop that can **overfit one batch**
- saved **input vs reconstruction** visualizations
- a saved **checkpoint**
- a basic **causality sanity check**

This is **not** full Cosmos yet.

We are intentionally skipping for Day 1:
- wavelet front-end
- causal attention
- temporal compression
- discrete tokenizer / FSQ
- diffusion model
- autoregressive model
- distributed training

---

## Scope

### Input
- source: BridgeData V2 scripted **video** subset only
- use only the visual stream for now
- ignore actions and language on Day 1

### Fixed setup
- clip length: `T = 8`
- resolution: `64 x 64`
- channels: RGB
- batch format: `[B, T, 3, 64, 64]`
- normalization: `[0, 1]` first, or `[-1, 1]` if easier for your codebase

### Target latent
- latent channels: `C = 16`
- spatial compression only: `4x`
- latent shape target:

```text
z: [B, T, 16, 16, 16]
```

---

## Deliverables

### Must-have by end of day
1. `VideoDataset` works and batch visualization looks correct.
2. `TokenizerV1` forward pass works.
3. `overfit_one_batch.py` shows reconstruction loss clearly dropping.
4. Reconstructions are visually recognizable.
5. One checkpoint saved.
6. One causality sanity test runs.

### Nice-to-have
- perceptual loss added
- train/val split working
- GIF or mp4 export for reconstructions

---

## Project structure

```text
cosmos_scratch/
├── data/
│   ├── processed/
│   ├── train.txt
│   └── val.txt
├── outputs/
│   ├── recon/
│   └── ckpts/
├── scripts/
│   ├── inspect_dataset.py
│   ├── overfit_one_batch.py
│   ├── train_tokenizer_v1.py
│   ├── recon_tokenizer_v1.py
│   └── causality_check.py
└── cosmos/
    ├── datasets/
    │   └── video_dataset.py
    ├── tokenizers/
    │   ├── causal_conv.py
    │   ├── blocks.py
    │   ├── tokenizer_v1.py
    │   └── losses.py
    └── utils/
        └── viz.py
```

---

## Model spec

### 1. `CausalConv3d`
Implement a 3D conv with:
- **left padding in time only**
- normal symmetric padding in height/width

Goal: output at time `t` depends only on frames `<= t`.

Suggested kernel:
- `kernel_size = (3, 3, 3)`
- `stride = 1`

Pseudo-rule: 
Left padding keeps past-context access; no right padding prevents future leakage
- pad temporal dimension with `k_t - 1` on the **left**
- do not right-pad time


### 2. `ResBlock3D`
Minimal residual block:
- norm
- SiLU
- causal conv
- norm
- SiLU
- causal conv
- residual add

Keep channels constant inside the block.

### 3. `DownsampleSpatial`
Only reduce height and width, not time.

Options:
- strided conv with stride `(1, 2, 2)`
- or average pool + conv

### 4. `UpsampleSpatial`
Only increase height and width, not time.

Options:
- nearest-neighbor upsample on spatial dims + causal conv
- or transposed conv with care

### 5. `TokenizerV1`
Methods:
- `encode(x) -> z`
- `decode(z) -> x_hat`
- `forward(x) -> x_hat, z`

Suggested encoder:
1. stem causal conv: `3 -> 64`
2. resblock
3. downsample spatial: `64 -> 128`
4. resblock
5. downsample spatial: `128 -> 256`
6. resblock
7. project to latent: `256 -> 16`

Suggested decoder:
1. project up: `16 -> 256`
2. resblock
3. upsample spatial: `256 -> 128`
4. resblock
5. upsample spatial: `128 -> 64`
6. resblock
7. output conv: `64 -> 3`

---

## Equations

### Encode / decode
\[
z = \mathcal{E}(x), \qquad \hat{x} = \mathcal{D}(z)
\]

### Reconstruction objective
\[
\mathcal{L}_{1} = \|\hat{x} - x\|_1
\]

### Optional Day 1 extension
\[
\mathcal{L} = \mathcal{L}_1 + \lambda_{perc} \mathcal{L}_{perc}
\]

Do **not** add flow loss, Gram loss, GAN loss, KL loss, or commitment loss on Day 1.

---

## Dataset tasks

### `inspect_dataset.py`
Tasks:
- load a few examples from the downloaded video subset
- decode frames
- verify temporal order
- print shape, dtype, min/max
- save a contact sheet or GIF

Acceptance:
- you can visually confirm clips are correct
- there are no obvious corrupted samples in the first small subset

### Preprocess to clips
Prepare a tiny subset first:
- take `100 to 500` clips max
- resize frames to `64 x 64`
- sample exactly `8` consecutive frames per clip
- save in a simple format

Good simple storage options:
- `.pt` tensors containing `[T, 3, H, W]`
- or `.npz`

For Day 1, prefer `.pt` if you want speed and simplicity.

### `VideoDataset`
Should:
- read processed clips
- return `torch.FloatTensor`
- shape: `[T, 3, H, W]`
- optionally apply normalization

Then `DataLoader` should produce:

```text
[B, T, 3, H, W]
```

---

## Training tasks

### Step 1: forward pass smoke test
Before training:
- instantiate model
- run one batch through it
- print:
  - input shape
  - latent shape
  - reconstruction shape

Acceptance:
- no shape errors
- latent is `[B, T, 16, 16, 16]`

### Step 2: overfit one batch
Script: `overfit_one_batch.py`

Use:
- optimizer: `AdamW`
- lr: `1e-4`
- batch size: `2` or `4`
- loss: `L1`
- steps: `500 to 2000`

Acceptance:
- loss drops clearly
- reconstructions become visibly close to the target

### Step 3: tiny training run
Script: `train_tokenizer_v1.py`

Use:
- small train split
- val split optional
- checkpoint every N steps
- save recon grids periodically

Acceptance:
- training runs without crashing
- recon quality is stable
- checkpoints saved correctly

---

## Visualization tasks

### `viz.py`
Implement helpers to save:
- frame grids
- side-by-side input / reconstruction
- optional GIF or mp4

Recommended layout:
- one row = input frames
- second row = reconstructed frames

### `recon_tokenizer_v1.py`
Inputs:
- checkpoint path
- one sample or one batch

Outputs:
- image grid
- optional GIF

Acceptance:
- easy visual inspection from saved files

---

## Causality sanity check

Script: `causality_check.py`

Procedure:
1. take one clip `x`
2. copy it to `x2`
3. change only frames after time `t`
4. compare:
   - `encode(x)[:, :t+1]`
   - `encode(x2)[:, :t+1]`

Expected:
- earlier latent states should be identical or nearly identical

Optional second check:
- compare decoder outputs for times `<= t`

Acceptance:
- future-frame edits do not affect earlier encoded states

---

## Suggested implementation order

### Hour 1
- write `inspect_dataset.py`
- confirm sample format
- preprocess a tiny set to fixed clips

### Hour 2
- implement `VideoDataset`
- confirm `[B, T, 3, H, W]`

### Hour 3
- implement `CausalConv3d`
- implement `ResBlock3D`

### Hour 4
- implement `TokenizerV1`
- smoke-test forward pass

### Hour 5
- implement `overfit_one_batch.py`
- train with L1 only

### Hour 6
- add visualization saving
- inspect reconstructions

### Hour 7
- save/load checkpoints
- run small train loop on more clips

### Hour 8
- run `causality_check.py`
- clean code and write notes

---

## Acceptance checklist

### Dataset
- [ ] Can load clips as `[T, 3, 64, 64]`
- [ ] Dataloader gives `[B, T, 3, 64, 64]`
- [ ] Sample visualization looks correct

### Model
- [ ] Forward pass works
- [ ] Latent shape is correct
- [ ] No future leakage in causal conv

### Training
- [ ] One-batch overfit works
- [ ] Loss decreases clearly
- [ ] Checkpoint saved

### Outputs
- [ ] Recon grid saved
- [ ] Optional GIF saved
- [ ] Causality sanity check passes

---

## Non-goals

Do not spend time today on:
- reproducing full Cosmos tokenizer exactly
- wavelet transform
- temporal compression
- attention blocks
- FSQ
- diffusion / autoregressive branches
- multi-GPU / distributed logic

---

## What comes tomorrow

If Day 1 works, Day 2 should be one of these:
1. add **perceptual loss**
2. improve block quality
3. add **wavelet front-end**
4. add **causal attention**

Only after tokenizer quality is decent should we move to the diffusion WFM.
