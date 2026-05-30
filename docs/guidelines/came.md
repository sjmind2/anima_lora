# CAME Optimizer — Confidence-guided Adaptive Matrix Evaluation

CAME is a factorized optimizer that replaces the full-matrix second moment (`exp_avg_sq`) with row and column moments (`exp_avg_sq_row`, `exp_avg_sq_col`), dramatically reducing optimizer state memory for 2D+ parameters. It also adds a residual correction mechanism to compensate for the factorization approximation error.

CAME is the default optimizer in the [Workflow engine](workflow.md) and is particularly well-suited for LyCORIS variant training (LoKR, LoHA) where factorized modules have small but numerous parameter matrices.

## Usage

### CLI / TOML config

Set `optimizer_type` in your method or preset TOML:

```toml
optimizer_type = "CAME"
learning_rate = 1.5e-5
```

Pass optimizer-specific arguments via `optimizer_args`:

```toml
optimizer_type = "CAME"
optimizer_args = ["weight_decay=0.01", "betas=0.9,0.999,0.9999"]
learning_rate = 1.5e-5
```

### Workflow

CAME is the default optimizer in Workflow schemas (`workflow/schemas/train_common.yaml`). No configuration needed — it's selected automatically unless you change it.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betas` | (0.9, 0.999, 0.9999) | (beta0 for exp_avg, beta1 for second moments, beta2 for residual moments) |
| `eps` | (1e-30, 1e-16) | (eps0 for update stability, eps1 for residual stability) |
| `clip_threshold` | 1.0 | RMS clipping threshold — prevents update explosion |
| `weight_decay` | 0.0 | Weight decay coefficient |
| `lr` | — | Learning rate. **Recommended: 0.5–0.9× of your equivalent AdamW learning rate.** |

### Internal behavior

- **2D+ parameters**: Factorized update with row/column second moments + residual correction. Same-shape parameters are stacked into batch tensors for efficiency.
- **1D parameters** (biases, scalars): Unfactored Adam-style update with full `exp_avg_sq`.

## Comparison with Other Optimizers

| Optimizer | Memory (per param) | Auto LR | Best for | Source |
|-----------|-------------------|---------|----------|--------|
| **AdamW** | 2 × full state | No | General purpose, default | `torch.optim` |
| **CAME** | ~2 × (row+col) state | No | LyCORIS variants, memory-constrained | Built-in (`library/training/came_optimizer.py`) |
| **Adopt_Adv** | Configurable (factored option) | No | Small-batch stability, long training | `adv_optm` package |
| **Prodigy_Adv** | 2 × full state + D-Adaptation state | **Yes** (set `lr=1.0`) | Unknown optimal LR | `adv_optm` package |

For the full advanced optimizer guide (Adopt_Adv, Prodigy_Adv), see [docs/optimizations/adv_optm_guide.md](../optimizations/adv_optm_guide.md).

## Recommended Learning Rates

CAME's factorized second-moment estimate behaves differently from AdamW's full estimate. In practice:

| Scenario | AdamW lr | CAME lr |
|----------|----------|---------|
| LoRA rank 32 | 2e-4 | 1–1.5e-4 |
| LoKR dim 8 | 2e-5 | 1.5e-5 |
| LoHA dim 32 | 1e-4 | 7e-5 |

Rule of thumb: **start with 0.7× of your AdamW learning rate** and adjust.

## Known Limitations

- **1D parameters** use unfactored (Adam-style) updates — no memory savings for biases and scalars.
- **torch.compile group stacking** groups same-shape parameters into batch tensors. When many distinct shapes exist, the grouping overhead may reduce the benefit. This is rarely an issue in practice.
- **Not compatible with** gradient accumulation factors > 1 without care — the per-step state update is correct, but the effective lr scaling may differ from AdamW.
