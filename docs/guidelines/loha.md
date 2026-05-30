# LoHA — Low-Rank Hadamard Product Adaptation

LoHA is a LyCORIS-family adapter that uses the Hadamard (element-wise) product of two low-rank matrices to achieve effective rank r² with only ~2× the parameters of standard LoRA. Custom autograd functions provide exact gradients for the Hadamard product.

For the mathematical walkthrough (forward formulas, dimension analysis, weight key naming), see [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md). This guide covers usage and configuration.

## Quick Start

### GUI

Select **LoHA** from the method dropdown in the Anima GUI. The dedicated config `configs/gui-methods/loha.toml` pre-fills all variant-specific parameters.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 32 | Rank r (effective rank ≈ r²) |
| `network_alpha` | 16 | LoRA alpha. **Scale = alpha / dim** — recommended range 0.1–0.5. |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |
| `conv_dim` | 4 | Rank for Conv2d layers (**inert on Anima** — DiT has no Conv2d) |
| `conv_alpha` | 1 | Alpha for Conv2d layers (**inert on Anima**) |
| `use_tucker` | `true` | Tucker mode for Conv2d (**inert on Anima**) |

### Understanding Scale

The scale factor `s = alpha / dim` controls the magnitude of the weight update ΔW. For LoHA:

- `dim=32, alpha=16` → scale = 0.5 (good starting point)
- `dim=32, alpha=8` → scale = 0.25 (more conservative)
- `dim=16, alpha=8` → scale = 0.5 (same scale, lower effective rank)

Recommended scale range: **0.1–0.5**. Higher scale = stronger adaptation but risk of instability.

## Comparison with LoKR

| Feature | LoHA | LoKR |
|---------|------|------|
| Core operation | Hadamard (element-wise) product | Kronecker product |
| Effective rank | r² | rank(W1) × rank(W2) (adaptive) |
| Params (same r) | ~2× LoRA | Adaptive; can be < LoRA |
| DoRA support | No | Yes (`weight_decompose=true`) |
| Custom autograd | `HadaWeight` / `HadaWeightTucker` | `KronLinearFn` / `KronLinearTwoStageFn` |
| Best for | Higher effective rank from a given r | Flexible parameter count with factor tuning |

## Inference

### CLI — Static Merge

`inference.py` auto-detects LoHA by inspecting safetensors key prefixes (`hada_*`). The weight delta is computed using the Hadamard product formula and merged into the base model weights.

LoHA checkpoints can coexist in the same `--lora_weight` list with regular LoRA or LoKR files.

### ComfyUI

ComfyUI's LyCORIS loader node natively supports LoHA weight formats. Directly loadable without conversion.

## Compatibility

Same mutual exclusivity rules as LoKR:

| Stacks with | Notes |
|-------------|-------|
| **T-LoRA** | Timestep rank masking applied after `make_weight` |
| **Spectrum** | No interaction |
| **Modulation guidance** | Orthogonal |
| **ReFT** | Orthogonal side-channel |
| **P-GRAFT** | Cutoff step toggles `network.enabled` |
| **HydraLoRA** | **Not supported** |
| **OrthoLoRA** | **Not supported** |

## Recommended Configs

### Standard training

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 4
```

### With T-LoRA and timestep masking

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
use_timestep_mask = true
min_rank = 8
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 6
```
