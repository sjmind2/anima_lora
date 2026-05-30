# LoKR — Low-Rank Kronecker Product Adaptation

LoKR is a LyCORIS-family adapter that factorizes weight dimensions and composes them via the Kronecker product, producing a structured high-rank approximation whose parameter count depends on the factorization shape rather than a single rank value.

For the mathematical walkthrough (forward formulas, dimension analysis, weight key naming, scalar baking), see [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md). This guide covers usage and configuration.

## Quick Start

### GUI

Select **LoKR** from the method dropdown in the Anima GUI. The dedicated config `configs/gui-methods/lokr.toml` pre-fills all variant-specific parameters.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 8 | Rank for low-rank factors |
| `network_alpha` | 8 | LoRA alpha (scale = alpha / dim) |
| `decompose_both` | `true` | Decompose both W1 and W2 into low-rank pairs. Reduces parameter count significantly at small ranks. |
| `lokr_factor` | -1 | Target factor size for dimension factorization. `-1` = auto (recommended). |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |
| `weight_decompose` | `false` | Enable DoRA-style weight decomposition (LoKR only — not available for LoHA/LoCON) |
| `use_scalar` | `false` | Learnable scalar (zero-init) instead of fixed scalar=1 |
| `full_matrix` | `false` | Force full (non-decomposed) matrices |
| `conv_dim` | 4 | Rank for Conv2d layers (**inert on Anima** — DiT has no Conv2d) |
| `conv_alpha` | 4 | Alpha for Conv2d layers (**inert on Anima**) |
| `use_tucker` | `true` | Tucker core for Conv2d (**inert on Anima**) |

## Anima-Specific Notes

### QKV Fusion and `lokr_factor`

Anima-base-v1.0 DiT fuses Q/K/V into a single `qkv_proj` Linear layer with shape `[6144, 2048]`. When saving checkpoints for ComfyUI compatibility, this fused module must be split back into separate `q_proj`/`k_proj`/`v_proj`. The split requires `factorization(6144, factor)` to produce an `out_l` divisible by 3.

**Recommended `lokr_factor` values:**

| `lokr_factor` | Raw `out_l` | Adjusted `out_l` | Checkpoint size |
|---------------|------------|------------------|-----------------|
| **-1** (auto) | auto | auto | ~10 MB |
| **6** | 6 | 6 (no change) | ~13 MB |
| **12** | 12 | 12 (no change) | ~7 MB |
| **24** | 24 | 24 (no change) | ~4 MB |
| 4 | 4 | **3** (adjusted) | ~10 MB |
| 8 | 8 | **6** (adjusted) | ~13 MB |
| 16 | 16 | **12** (adjusted) | ~7 MB |

All values produce correctly-sized checkpoints. Using a factor that is a multiple of 3 (6, 12, 24) avoids the adjustment entirely. Use `lokr_factor = -1` (auto) for the most balanced factorization.

### No Conv2d in Anima DiT

All LoRA target layers in Anima-base-v1.0 DiT are `nn.Linear`. The `conv_dim`, `conv_alpha`, and `use_tucker` parameters are accepted without error but have no effect.

## Inference

### CLI — Static Merge

`inference.py` auto-detects LoKR by inspecting safetensors key prefixes (`lokr_*`). The weight delta is computed using the Kronecker product formula and merged into the base model weights before denoising.

LoKR checkpoints can coexist in the same `--lora_weight` list with regular LoRA files — each file is merged independently.

### ComfyUI

ComfyUI's LyCORIS loader node natively supports LoKR weight formats. The safetensors files produced by this trainer use the same key naming convention as the sd-scripts / LyCORIS ecosystem, so they are directly loadable without conversion.

## Compatibility

| Stacks with | Notes |
|-------------|-------|
| **T-LoRA** | Timestep rank masking applied after `make_weight` on the reconstructed weight |
| **Spectrum** | No interaction — cached steps skip blocks entirely |
| **Modulation guidance** | Orthogonal — touches AdaLN only |
| **ReFT** | Orthogonal side-channel |
| **P-GRAFT** | Cutoff step toggles `network.enabled` |
| **HydraLoRA** | **Not supported** — requires standard BA structure |
| **OrthoLoRA** | **Not supported** — Cayley re-parameterization defined for standard BA only |

## Recommended Configs

### Small datasets (≤ 20 images)

```toml
network_type = "lokr"
network_dim = 8
network_alpha = 8
decompose_both = true
lokr_factor = -1
scale_weight_norms = 1.0
learning_rate = 2e-5
max_train_epochs = 4
```

### Large datasets (100+ images)

```toml
network_type = "lokr"
network_dim = 16
network_alpha = 16
decompose_both = true
lokr_factor = 12
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 8
```
