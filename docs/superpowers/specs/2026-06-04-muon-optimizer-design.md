# Muon Optimizer Integration Design

**Date:** 2026-06-04
**Status:** Approved
**Scope:** Single-GPU only, torch.compile compatible

## Overview

Integrate the Muon optimizer (Momentum Orthogonalized by Newton-schulz) from `E:\Comfy\DiT-Muon\muon.py` into the project's optimizer catalog. Muon replaces AdamW's per-element adaptive learning rate with a spectral/orthogonalization post-processing step on the momentum update via Newton-Schulz quintic iteration.

## Source

`E:\Comfy\DiT-Muon\muon.py` (236 lines). Based on `SingleDeviceMuonWithAuxAdam` — single-GPU variant with built-in Adam fallback for 1D parameters.

## Why Muon

| Metric | AdamW | Muon |
|--------|-------|------|
| State per param | 2 buffers (exp_avg, exp_avg_sq) | 1 buffer (momentum only) |
| Memory overhead | 2x parameter size | 1x parameter size |
| Gradient processing | Per-element: m / (sqrt(v) + eps) | Per-matrix: orthogonalize via Newton-Schulz |
| Internal precision | FP32 or AMP | bfloat16 (NS iteration) |

## Mathematical Formulation (ported verbatim)

Given parameter matrix W with gradient G:

1. **Nesterov momentum:**
   ```
   momentum.lerp_(grad, 1 - beta)        # m = beta*m + (1-beta)*G
   update = grad.lerp_(momentum, beta)    # u = (1-beta)*G + beta*m  [Nesterov]
   ```

2. **Newton-Schulz quintic iteration** (5 steps, coefficients a=3.4445, b=-4.7750, c=2.0315):
   ```
   X = update.bfloat16()
   if rows > cols: X = X.mT              # work with smaller Gram matrix
   X = X / (||X||_F + 1e-7)              # spectral norm <= 1
   for _ in range(5):
       A = X @ X.mT
       B = b*A + c*(A @ A)
       X = a*X + B @ X
   if rows > cols: X = X.mT
   ```

3. **Aspect ratio compensation:**
   ```
   update = X * max(1, rows/cols)^0.5
   ```

4. **Weight decay + parameter update:**
   ```
   param *= (1 - lr * weight_decay)
   param -= lr * update
   ```

## Architecture

### New file: `library/training/muon_optimizer.py`

```
MuonOptimizer (torch.optim.Optimizer)
├── __init__(params, lr, momentum=0.95, ns_steps=5, weight_decay=0)
├── _init_state(p)           -> pre-initialize momentum buffer
├── step()                   -> eager loop over param_groups
│   ├── _muon_param_step()   <- torch.compile'd single-param step (2D+ params)
│   │   ├── momentum.lerp_(grad)
│   │   ├── Nesterov lookahead
│   │   ├── NS quintic iteration (inlined)
│   │   ├── aspect ratio scaling
│   │   ├── weight decay
│   │   └── parameter update
│   └── _adam_param_step()   <- torch.compile'd single-param step (1D params)
│       ├── exp_avg / exp_avg_sq EMA
│       ├── bias correction
│       └── weight decay + update
└── supports_fused = False
```

### Modification: `library/training/optimizers.py`

Add one `elif` branch:

```python
elif optimizer_type == "Muon".lower():
    from library.training.muon_optimizer import MuonOptimizer
    logger.info(f"use Muon optimizer | {optimizer_kwargs}")
    logger.info("Muon: recommended lr is ~10x your usual AdamW lr")
    optimizer_class = MuonOptimizer
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
```

No other files modified.

## torch.compile Strategy

### Compiled function signature

```python
def _muon_param_step(
    param_data: Tensor,    # (out, in) or (rank, dim)
    grad: Tensor,          # same shape
    momentum: Tensor,      # same shape
    lr: float, beta: float, weight_decay: float, ns_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:  # (param_data, momentum, dummy_grad)
```

### Why this won't trigger recompile

1. **LoRA parameter shapes are fixed** throughout training — `(rank, dim)` and `(out, rank)` never change.
2. **`dynamic=False`** (default) → compile generates one graph per unique shape, then reuses.
3. **`if G.size(-2) > G.size(-1)`** becomes a compile-time constant under `dynamic=False` — no runtime branch.
4. **No conv 4D branch** — removed (not needed in this project).
5. **State pre-initialized in `__init__`** — no first-step `len(state) == 0` branch.

### Why this won't interfere with DiT block compilation

- Optimizer step runs after backward, in a separate call.
- No shared operators or autograd context with `block._forward`.
- Momentum buffers managed inside the optimizer, never enter autograd graph.

### Performance rationale

Compiling the entire single-param step (not just NS iteration) enables:
- **Kernel fusion**: lerp + NS matmuls + scaling + decay → fewer kernel launches
- **Launch overhead reduction**: ~8 launches/param → ~1 launch/param in compiled graph
- The outer Python loop is cheap (just iteration overhead); the compiled inner function does all the work.

## Parameter Group Routing

The project's `prepare_optimizer_params` produces groups like `{lora, plus, router}` with different LRs. MuonOptimizer routes internally:

```python
for p in group["params"]:
    if p.ndim >= 2:
        # Muon path (Newton-Schulz orthogonalization)
    else:
        # Built-in Adam path (bias, LayerNorm, 1D router params)
```

No changes to `prepare_optimizer_params` output format. No `use_muon` flag required.

## Constraints

- **Single-GPU only** — no `dist.all_gather`, no distributed sharding.
- **2D+ parameters only** for Muon path — 1D params fall back to built-in Adam.
- **bfloat16 internal precision** — NS iteration forces `.bfloat16()` regardless of training dtype.
- **No new dependencies** — pure PyTorch.

## Usage

```bash
# CLI
python train.py --optimizer_type Muon --learning_rate 2e-4 ...

# Config (method TOML)
optimizer_type = "Muon"
learning_rate = 2e-4
```

## LR Guidance

Muon's LR semantics differ from AdamW (spectral norm per update vs. per-element magnitude). The original author recommends ~10x the typical AdamW LR. The optimizer logs a reminder at creation.

## Out of Scope

- Distributed Muon variants (`Muon`, `MuonWithAuxAdam` with `all_gather`).
- 4D conv parameter support (not needed).
- Modifications to `prepare_optimizer_params` output format.
- Modifications to DiT block compilation logic.
- Stacked/batched NS iteration (CAME-style) — ROI too low for LoRA's small matrices.
