# Model Configuration Guide

All parameters live in the YAML files under `configs/`.
Run with:
```bash
python scripts/overfit_one_batch.py --config configs/overfit_continuous.yaml
python scripts/train_continuous_tokenizer.py --config configs/train_continuous.yaml
```

---

## Key model knobs

| YAML key | Values | Effect |
|---|---|---|
| `temporal_compression` | `1`, `2`, `4` | How many times T is halved |
| `conv_type` | `full`, `factorized` | Joint 3D kernel vs separate spatial+temporal passes |
| `base_channels` | e.g. `64`, `128` | Channel width; stages scale as C → 2C → 4C |
| `latent_channels` | e.g. `16` | Depth of the latent vector |
| `use_attention` | `true`, `false` | Causal spatiotemporal attention at deep stages |

---

## Day 1 vs Day 2 comparison

| Setting | Day 1 (original) | Day 2 (default) |
|---|---|---|
| `temporal_compression` | `1` (spatial only) | `4` (T÷4) |
| `conv_type` | `full` (joint 3D) | `factorized` (space+time separate) |
| `use_perceptual` | `false` | `true` |
| Latent shape (T=8) | `[B, 16, 8, 16, 16]` | `[B, 16, 2, 16, 16]` |
| Latent elements | 32,768 | 8,192 |
| Compression ratio | ~3× | ~12× |

**Why Day 1 looked better:** it compresses 4× less information into the latent.
The decoder has an easier job — no temporal frames need to be hallucinated.

---

## All tc × conv_type combinations

| `temporal_compression` | `conv_type` | Latent (T=8) | Params | Notes |
|---|---|---|---|---|
| `1` | `full` | `[B, 16, 8, 16, 16]` | 2.7M | Closest to Day 1 |
| `1` | `factorized` | `[B, 16, 8, 16, 16]` | 1.3M | Day 1 compression, Day 2 conv |
| `2` | `full` | `[B, 16, 4, 16, 16]` | 4.6M | Middle ground |
| `2` | `factorized` | `[B, 16, 4, 16, 16]` | 2.2M | Middle ground |
| `4` | `factorized` | `[B, 16, 2, 16, 16]` | 5.5M | **Day 2 default** |
| `4` | `full` | `[B, 16, 2, 16, 16]` | 11.8M | Day 2 compression, Day 1 conv |

---

## Increasing model capacity

Double `base_channels` to give the model more capacity for the same compression:

```yaml
model:
  base_channels: 128   # default 64 — stages become 128 → 256 → 512
```

| `base_channels` | `tc` | Params (factorized) |
|---|---|---|
| `64` | `4` | 5.5M |
| `128` | `4` | ~22M |
| `64` | `1` | 1.3M |
| `128` | `1` | ~5M |

---

## Loss knobs

| YAML key | Default | Notes |
|---|---|---|
| `use_perceptual` | `true` | VGG feature matching — adds sharpness |
| `use_flow` | `false` | Frame-diff optical flow proxy — adds temporal consistency |
| `use_gram` | `false` | Gram matrix style loss — adds texture matching |
| `lambda_l1` | `1.0` | Weight of pixel L1 term |
| `lambda_perc` | `0.1` | Weight of perceptual term |
| `lambda_flow` | `0.05` | Weight of flow term |
| `lambda_gram` | `0.02` | Weight of Gram term |

---

## Presets

### Day 1 replica
```yaml
model:
  temporal_compression: 1
  conv_type:            full
  use_attention:        false
loss:
  use_perceptual: false
```

### Day 2 default
```yaml
model:
  temporal_compression: 4
  conv_type:            factorized
  use_attention:        false
loss:
  use_perceptual: true
```

### Day 2 + full loss stack
```yaml
model:
  temporal_compression: 4
  conv_type:            factorized
  use_attention:        true
loss:
  use_perceptual: true
  use_flow:       true
  use_gram:       true
```
