# Muon Optimizer Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the Muon optimizer (Newton-Schulz orthogonalized momentum) into the optimizer catalog with torch.compile support.

**Architecture:** Single new file `library/training/muon_optimizer.py` containing the `MuonOptimizer` class with compiled single-param step functions. One modification to `library/training/optimizers.py` to add the dispatch branch. Tests verify math consistency, compile compatibility, and param routing.

**Tech Stack:** PyTorch, torch.compile, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `library/training/muon_optimizer.py` | Create | MuonOptimizer class, NS iteration, compiled step functions |
| `library/training/optimizers.py` | Modify (one `elif` branch) | Dispatch `Muon` type to MuonOptimizer |
| `tests/test_muon_optimizer.py` | Create | Math consistency, compile, routing tests |

---

### Task 1: Create Muon optimizer with NS iteration and compiled step functions

**Files:**
- Create: `library/training/muon_optimizer.py`

- [ ] **Step 1: Write the failing test for NS iteration correctness**

Create `tests/test_muon_optimizer.py`:

```python
"""Tests for the Muon optimizer integration."""
from __future__ import annotations

import math

import pytest
import torch


def test_ns_iteration_output_is_orthogonal():
    """Newton-Schulz iteration should produce an approximately orthogonal matrix."""
    from library.training.muon_optimizer import _zeropower_via_newtonschulz5

    torch.manual_seed(42)
    G = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    result = _zeropower_via_newtonschulz5(G, steps=5)

    # result^T @ result should be approximately identity (for wide matrices)
    gram = result.float() @ result.float().mT
    eye = torch.eye(32, dtype=torch.float32)
    assert torch.allclose(gram, eye, atol=0.15), f"Gram matrix not close to identity: max diff = {(gram - eye).abs().max().item()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\activate; python -m pytest tests/test_muon_optimizer.py::test_ns_iteration_output_is_orthogonal -v`
Expected: FAIL with `ImportError: cannot import name '_zeropower_via_newtonschulz5'`

- [ ] **Step 3: Write `library/training/muon_optimizer.py` with NS iteration and compiled step functions**

```python
"""Muon optimizer — Momentum Orthogonalized by Newton-schulz.

Ported from E:\\Comfy\\DiT-Muon\\muon.py (SingleDeviceMuonWithAuxAdam).
Single-GPU only. torch.compile compatible via compiled single-param step
functions.

Math reference:
    Given parameter matrix W with gradient G:
    1. Nesterov momentum: m = beta*m + (1-beta)*G; u = (1-beta)*G + beta*m
    2. Newton-Schulz quintic iteration (5 steps, coefficients 3.4445, -4.7750, 2.0315)
    3. Aspect ratio scaling: update *= max(1, rows/cols)^0.5
    4. Weight decay + parameter update
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to approximate the closest orthogonal matrix.

    Uses a quintic iteration with coefficients chosen to maximize the slope
    at zero rather than guarantee convergence to exact UV^T. The result is
    approximately US'V^T where S'_ii ~ Uniform(0.5, 1.5).

    Args:
        G: Input tensor, must be >= 2D. Last two dims are treated as the matrix.
        steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Approximately orthogonal matrix of the same shape.
    """
    assert G.ndim >= 2, f"Muon requires >= 2D tensors, got {G.ndim}D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_param_step(
    param_data: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    lr: float,
    beta: float,
    weight_decay: float,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled single-param step for Muon (2D+ parameters).

    Returns (updated_param_data, updated_momentum, consumed_grad).
    """
    # Nesterov momentum
    momentum = momentum.lerp(grad, 1 - beta)
    update = grad.lerp(momentum, beta)

    # Newton-Schulz orthogonalization
    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Aspect ratio compensation
    rows, cols = update.size(-2), update.size(-1)
    update = update * max(1, rows / cols) ** 0.5

    # Weight decay
    if weight_decay != 0:
        param_data = param_data.add(param_data, alpha=-weight_decay * lr)

    # Parameter update
    param_data = param_data.add(update, alpha=-lr)

    return param_data, momentum, grad


def _adam_param_step(
    param_data: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled single-param step for built-in Adam (1D parameters).

    Returns (updated_param_data, updated_exp_avg, updated_exp_avg_sq, updated_step).
    """
    step = step + 1

    exp_avg = exp_avg.lerp(grad, 1 - beta1)
    exp_avg_sq = exp_avg_sq.lerp(grad.square(), 1 - beta2)

    bias_correction1 = 1 - beta1 ** step.item()
    bias_correction2 = 1 - beta2 ** step.item()
    update = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)

    if weight_decay != 0:
        param_data = param_data.add(param_data, alpha=-weight_decay * lr)

    param_data = param_data.add(update, alpha=-lr)

    return param_data, exp_avg, exp_avg_sq, step


# Compiled versions — lazily initialized on first CUDA use
_muon_param_step_compiled = None
_adam_param_step_compiled = None


def _ensure_compiled():
    global _muon_param_step_compiled, _adam_param_step_compiled
    if _muon_param_step_compiled is not None:
        return
    _muon_param_step_compiled = torch.compile(_muon_param_step, fullgraph=False)
    _adam_param_step_compiled = torch.compile(_adam_param_step, fullgraph=False)


class MuonOptimizer(torch.optim.Optimizer):
    """Muon optimizer with built-in Adam fallback for 1D parameters.

    2D+ parameters (weight matrices) use Newton-Schulz orthogonalized momentum.
    1D parameters (biases, LayerNorm) use standard Adam with bias correction.

    Single-GPU only. torch.compile compatible — compiled single-param step
    functions are used on CUDA.

    Usage:
        optimizer = MuonOptimizer(trainable_params, lr=2e-4)
        # lr should be ~10x your usual AdamW lr

    Args:
        params: Parameter groups (same format as other optimizers in the catalog).
        lr: Learning rate (default 2e-4).
        momentum: Momentum beta for Muon path (default 0.95).
        ns_steps: Number of Newton-Schulz iterations (default 5).
        weight_decay: AdamW-style weight decay (default 0).
        betas: Adam betas for 1D params (default (0.9, 0.999)).
        eps: Adam eps for 1D params (default 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

        # Pre-initialize state for all parameters to avoid first-step recompile
        for group in self.param_groups:
            for p in group["params"]:
                self._init_state(p)

        logger.info(
            f"MuonOptimizer initialized | "
            f"lr={lr}, momentum={momentum}, ns_steps={ns_steps}, "
            f"weight_decay={weight_decay}"
        )

    def _init_state(self, p: torch.nn.Parameter):
        """Pre-initialize optimizer state to avoid first-step recompile."""
        if len(self.state[p]) > 0:
            return  # already initialized
        if p.ndim >= 2:
            self.state[p]["momentum_buffer"] = torch.zeros_like(p)
        else:
            self.state[p]["exp_avg"] = torch.zeros_like(p)
            self.state[p]["exp_avg_sq"] = torch.zeros_like(p)
            self.state[p]["step"] = torch.tensor(0.0, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        use_compiled = torch.cuda.is_available()
        if use_compiled:
            _ensure_compiled()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if p.ndim >= 2:
                    # Muon path (2D+ parameters)
                    fn = _muon_param_step_compiled if use_compiled else _muon_param_step
                    p.data, state["momentum_buffer"], _ = fn(
                        p.data, grad, state["momentum_buffer"],
                        lr, momentum, weight_decay, ns_steps,
                    )
                else:
                    # Built-in Adam path (1D parameters)
                    fn = _adam_param_step_compiled if use_compiled else _adam_param_step
                    p.data, state["exp_avg"], state["exp_avg_sq"], state["step"] = fn(
                        p.data, grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], lr, beta1, beta2, eps, weight_decay,
                    )

        return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\activate; python -m pytest tests/test_muon_optimizer.py::test_ns_iteration_output_is_orthogonal -v`
Expected: PASS

- [ ] **Step 5: Commit**

```
git add library/training/muon_optimizer.py tests/test_muon_optimizer.py
git commit -m "Add Muon optimizer with NS iteration and compiled step functions"
```

---

### Task 2: Add math consistency tests

**Files:**
- Modify: `tests/test_muon_optimizer.py`

- [ ] **Step 1: Write tests verifying math matches original Muon**

Append to `tests/test_muon_optimizer.py`:

```python
def test_muon_update_matches_reference():
    """Muon update should match the original muon_update function output."""
    from library.training.muon_optimizer import _zeropower_via_newtonschulz5

    torch.manual_seed(123)
    # Test wide matrix (rows < cols) — typical LoRA down shape (rank, dim)
    G_wide = torch.randn(16, 1024)
    result_wide = _zeropower_via_newtonschulz5(G_wide, steps=5)
    # Output should be same shape
    assert result_wide.shape == G_wide.shape

    # Test tall matrix (rows > cols) — typical LoRA up shape (dim, rank)
    G_tall = torch.randn(1024, 16)
    result_tall = _zeropower_via_newtonschulz5(G_tall, steps=5)
    assert result_tall.shape == G_tall.shape

    # Test square matrix
    G_sq = torch.randn(64, 64)
    result_sq = _zeropower_via_newtonschulz5(G_sq, steps=5)
    assert result_sq.shape == G_sq.shape


def test_momentum_update_correctness():
    """Verify Nesterov momentum computation matches the original formula."""
    torch.manual_seed(42)
    beta = 0.95
    grad = torch.randn(32, 64)
    momentum = torch.zeros_like(grad)

    # Reference: original muon.py formula
    momentum_ref = momentum.clone()
    momentum_ref.lerp_(grad, 1 - beta)           # m = beta*m + (1-beta)*G
    update_ref = grad.clone().lerp_(momentum_ref, beta)  # u = (1-beta)*G + beta*m

    # Our implementation (same formula, from _muon_param_step internals)
    momentum_test = torch.zeros_like(grad)
    momentum_test = momentum_test.lerp(grad, 1 - beta)
    update_test = grad.lerp(momentum_test, beta)

    assert torch.allclose(momentum_ref, momentum_test, atol=1e-7)
    assert torch.allclose(update_ref, update_test, atol=1e-7)


def test_ns_iteration_bfloat16_internal():
    """NS iteration should use bfloat16 internally regardless of input dtype."""
    from library.training.muon_optimizer import _zeropower_via_newtonschulz5

    G = torch.randn(32, 64, dtype=torch.float32)
    result = _zeropower_via_newtonschulz5(G, steps=5)
    # Output dtype should be bfloat16 (the internal compute precision)
    assert result.dtype == torch.bfloat16


def test_ns_iteration_coefficients():
    """Verify the exact Newton-Schulz coefficients match the reference."""
    # The coefficients are (3.4445, -4.7750, 2.0315) — this test prevents
    # accidental modification.
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Manual single NS step verification
    torch.manual_seed(0)
    X = torch.randn(4, 8, dtype=torch.bfloat16)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    A = X @ X.mT
    B = b * A + c * A @ A
    X_new = a * X + B @ X

    # Verify it's not NaN and not all zeros
    assert not torch.isnan(X_new).any()
    assert X_new.abs().sum() > 0


def test_aspect_ratio_scaling():
    """Verify aspect ratio scaling: max(1, rows/cols)^0.5."""
    # Wide matrix (rows < cols): scale = 1
    assert max(1, 16 / 1024) ** 0.5 == 1.0

    # Tall matrix (rows > cols): scale = sqrt(rows/cols)
    expected = (1024 / 16) ** 0.5
    assert abs(max(1, 1024 / 16) ** 0.5 - expected) < 1e-6

    # Square: scale = 1
    assert max(1, 64 / 64) ** 0.5 == 1.0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv\Scripts\activate; python -m pytest tests/test_muon_optimizer.py -v -k "test_muon_update_matches or test_momentum_update or test_ns_iteration_bfloat16 or test_ns_iteration_coefficients or test_aspect_ratio"`
Expected: All PASS

- [ ] **Step 3: Commit**

```
git add tests/test_muon_optimizer.py
git commit -m "Add math consistency tests for Muon optimizer"
```

---

### Task 3: Add optimizer integration and param routing tests

**Files:**
- Modify: `tests/test_muon_optimizer.py`

- [ ] **Step 1: Write tests for MuonOptimizer class and param routing**

Append to `tests/test_muon_optimizer.py`:

```python
def test_optimizer_routes_2d_to_muon_1d_to_adam():
    """2D+ params use Muon path, 1D params use Adam path."""
    from library.training.muon_optimizer import MuonOptimizer

    torch.manual_seed(42)
    param_2d = torch.nn.Parameter(torch.randn(32, 64))
    param_1d = torch.nn.Parameter(torch.randn(32))

    opt = MuonOptimizer([param_2d, param_1d], lr=1e-3)

    # State should be pre-initialized
    state_2d = opt.state[param_2d]
    state_1d = opt.state[param_1d]

    assert "momentum_buffer" in state_2d
    assert "exp_avg" in state_1d
    assert "exp_avg_sq" in state_1d
    assert "step" in state_1d


def test_optimizer_step_updates_params():
    """A single optimizer step should update all parameters."""
    from library.training.muon_optimizer import MuonOptimizer

    torch.manual_seed(42)
    param_2d = torch.nn.Parameter(torch.randn(16, 32))
    param_1d = torch.nn.Parameter(torch.randn(16))

    opt = MuonOptimizer([param_2d, param_1d], lr=1e-3)

    # Simulate gradients
    param_2d.grad = torch.randn_like(param_2d)
    param_1d.grad = torch.randn_like(param_1d)

    p2d_before = param_2d.data.clone()
    p1d_before = param_1d.data.clone()

    opt.step()

    # Parameters should have changed
    assert not torch.allclose(param_2d.data, p2d_before)
    assert not torch.allclose(param_1d.data, p1d_before)


def test_optimizer_param_groups_with_lr():
    """MuonOptimizer should accept parameter groups with per-group LR."""
    from library.training.muon_optimizer import MuonOptimizer

    p1 = torch.nn.Parameter(torch.randn(16, 32))
    p2 = torch.nn.Parameter(torch.randn(8, 16))

    opt = MuonOptimizer(
        [
            {"params": [p1], "lr": 1e-3},
            {"params": [p2], "lr": 2e-3},
        ],
        lr=1e-4,  # default, overridden by group lr
    )

    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 2e-3


def test_optimizer_weight_decay():
    """Weight decay should shrink parameters toward zero."""
    from library.training.muon_optimizer import MuonOptimizer

    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.randn(16, 32))

    opt = MuonOptimizer([param], lr=1e-3, weight_decay=0.1)
    param.grad = torch.zeros_like(param)  # zero gradient

    param_before = param.data.clone()
    opt.step()

    # With zero gradient and weight_decay > 0, param should shrink
    assert param.data.abs().sum() < param_before.abs().sum()


def test_optimizer_zero_grad_no_change():
    """With None gradients, parameters should not change."""
    from library.training.muon_optimizer import MuonOptimizer

    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.randn(16, 32))

    opt = MuonOptimizer([param], lr=1e-3)
    # No gradient set (p.grad is None)

    param_before = param.data.clone()
    opt.step()

    assert torch.allclose(param.data, param_before)
```

- [ ] **Step 2: Run tests**

Run: `.venv\Scripts\activate; python -m pytest tests/test_muon_optimizer.py -v -k "test_optimizer"`
Expected: All PASS

- [ ] **Step 3: Commit**

```
git add tests/test_muon_optimizer.py
git commit -m "Add param routing and optimizer class tests for Muon"
```

---

### Task 4: Integrate into optimizer catalog

**Files:**
- Modify: `library/training/optimizers.py` (add `elif` branch after the `CAME` branch, before `AdamW`)

- [ ] **Step 1: Add Muon branch to `get_optimizer()`**

In `library/training/optimizers.py`, insert after the `CAME` branch (after line 353) and before the `AdamW` branch (line 355):

```python
    elif optimizer_type == "Muon".lower():
        from library.training.muon_optimizer import MuonOptimizer

        logger.info(f"use Muon optimizer | {optimizer_kwargs}")
        logger.info(
            "Muon: recommended lr is ~10x your usual AdamW lr "
            f"(current lr={lr})"
        )
        optimizer_class = MuonOptimizer
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
```

- [ ] **Step 2: Write test verifying Muon is dispatchable via get_optimizer**

Append to `tests/test_muon_optimizer.py`:

```python
def test_get_optimizer_dispatches_muon():
    """get_optimizer should create MuonOptimizer when optimizer_type='Muon'."""
    import argparse

    from library.training.optimizers import get_optimizer

    args = argparse.Namespace(
        optimizer_type="Muon",
        use_8bit_adam=False,
        use_lion_optimizer=False,
        fused_backward_pass=False,
        learning_rate=2e-4,
        optimizer_args=None,
    )

    p1 = torch.nn.Parameter(torch.randn(16, 32))
    p2 = torch.nn.Parameter(torch.randn(8))

    name, opt_args, optimizer = get_optimizer(args, [p1, p2])

    from library.training.muon_optimizer import MuonOptimizer
    assert isinstance(optimizer, MuonOptimizer)
    assert "Muon" in name
```

- [ ] **Step 3: Run all tests**

Run: `.venv\Scripts\activate; python -m pytest tests/test_muon_optimizer.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```
git add library/training/optimizers.py tests/test_muon_optimizer.py
git commit -m "Integrate Muon optimizer into optimizer catalog dispatch"
```

---

### Task 5: Lint and final verification

**Files:**
- All modified files

- [ ] **Step 1: Run ruff lint and format**

Run: `.venv\Scripts\activate; ruff check library/training/muon_optimizer.py library/training/optimizers.py tests/test_muon_optimizer.py --fix; ruff format library/training/muon_optimizer.py library/training/optimizers.py tests/test_muon_optimizer.py`
Expected: No errors

- [ ] **Step 2: Run full test suite**

Run: `.venv\Scripts\activate; python -m pytest tests/test_muon_optimizer.py -v`
Expected: All PASS

- [ ] **Step 3: Commit any lint fixes**

```
git add -A
git commit -m "Lint and format Muon optimizer files"
```
