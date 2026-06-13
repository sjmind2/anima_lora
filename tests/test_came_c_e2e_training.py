"""
End-to-end training simulation tests for CAME_C vs CAME.

Simulates a real training process with a synthetic LoRA adapter model and
compares mathematical consistency, loss trajectories, optimizer states,
and wall-clock performance between the Python CAME and C++ CAME_C optimizers.

Key design decisions:
  - Uses fp32 parameters throughout to avoid the Python CAME dtype-promotion
    issue (Python promotes bf16 params to fp32 after step 1 via p.data =
    new_tensor, while CAME_C preserves original dtype via copy_).
  - Both optimizers receive identical initial conditions (same seed, same
    params, same gradients) at every step.
  - Timing uses wall-clock measurement after a warmup phase to amortize JIT
    compilation overhead (torch.compile for CAME, cpp_extension.load for CAME_C).

Run with:
    pytest tests/test_came_c_e2e_training.py -v
    python tests/test_came_c_e2e_training.py
"""

import copy
import time

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optimizer imports with availability guard
# ---------------------------------------------------------------------------
from library.training.came_optimizer import CAME  # Python reference

try:
    from library.training.came_cpp_extension import CAME_C  # C++ version
    from library.training.came_cpp_extension import (
        _EXTENSION_AVAILABLE as _CAME_C_EXT_OK,
    )

    _CAME_C_AVAILABLE = True
except Exception:
    _CAME_C_AVAILABLE = False
    _CAME_C_EXT_OK = False

_CAME_C_SKIP = pytest.mark.skipif(
    not (_CAME_C_AVAILABLE and _CAME_C_EXT_OK),
    reason="CAME_C extension not available",
)

_CUDA_SKIP = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Synthetic model
# ---------------------------------------------------------------------------


class SyntheticLoRA(nn.Module):
    """Minimal model that mimics a LoRA adapter.

    Real LoRA shapes: (dim, rank) and (rank, dim) where dim=4096, rank=32.
    We use smaller dims (dim=256, rank=16) for test speed.
    """

    def __init__(self, dim: int = 256, rank: int = 16):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False)  # weight shape: (rank, dim)
        self.lora_B = nn.Linear(
            rank, dim, bias=True
        )  # weight shape: (dim, rank), bias: (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared constants
DIM = 256
RANK = 16
BATCH_SIZE = 4
DTYPE = torch.float32
DEVICE = "cuda"
LR = 1e-4
BETAS = (0.9, 0.999, 0.9999)
SEED = 42
LOSS_FN = nn.MSELoss()


def _make_base_model(
    dim: int = DIM,
    rank: int = RANK,
    seed: int = SEED,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> nn.Module:
    """Create a fresh SyntheticLoRA model with deterministic weights."""
    torch.manual_seed(seed)
    return SyntheticLoRA(dim=dim, rank=rank).to(device=device, dtype=dtype)


def _make_training_pair():
    """Create a pair of cloned models + optimizers with identical weights.

    Returns (model_py, model_cpp, opt_py, opt_cpp).
    """
    base = _make_base_model()
    model_py = copy.deepcopy(base)
    model_cpp = copy.deepcopy(base)

    # Verify weights are byte-identical before training
    for (n1, p1), (n2, p2) in zip(
        model_py.named_parameters(), model_cpp.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1.data, p2.data), f"Initial weights differ for {n1}"

    opt_py = CAME(model_py.parameters(), lr=LR, betas=BETAS)
    opt_cpp = CAME_C(model_cpp.parameters(), lr=LR, betas=BETAS)
    return model_py, model_cpp, opt_py, opt_cpp


def _train_step(model, x, target, optimizer, loss_fn):
    """Forward + backward + optimizer.step(). Returns the scalar loss value."""
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def _train_n_steps(model, optimizer, loss_fn, n_steps, seed_offset=0):
    """Run *n_steps* training iterations with fresh random data each step.

    Returns a list of loss values.
    """
    losses = []
    for step in range(n_steps):
        torch.manual_seed(seed_offset + step + 1000)
        x = torch.randn(BATCH_SIZE, 256, DIM, device=DEVICE, dtype=DTYPE)
        target = torch.randn(BATCH_SIZE, 256, DIM, device=DEVICE, dtype=DTYPE)
        loss_val = _train_step(model, x, target, optimizer, loss_fn)
        losses.append(loss_val)
    return losses


def _collect_optimizer_states(optimizer, model):
    """Return a dict of {param_name: {state_key: tensor.cpu()}}."""
    result = {}
    for name, p in model.named_parameters():
        state = optimizer.state.get(p, {})
        result[name] = {}
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                result[name][key] = val.detach().cpu().clone()
            else:
                result[name][key] = val  # e.g. step counter
    return result


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestCAMECE2ETraining:
    """End-to-end training simulation: CAME vs CAME_C."""

    @_CAME_C_SKIP
    @_CUDA_SKIP
    def test_single_step_parity(self):
        """One step with identical params/grads -> identical results."""
        model_py, model_cpp, opt_py, opt_cpp = _make_training_pair()

        torch.manual_seed(SEED + 200)
        x = torch.randn(BATCH_SIZE, 256, DIM, device=DEVICE, dtype=DTYPE)
        target = torch.randn_like(x)

        # Identical inputs for both optimizers
        loss_py = _train_step(model_py, x, target, opt_py, LOSS_FN)
        loss_cpp = _train_step(model_cpp, x, target, opt_cpp, LOSS_FN)

        # Loss should be identical (same model, same data, same math)
        assert loss_py == pytest.approx(loss_cpp, rel=1e-5), (
            f"Loss mismatch after 1 step: Python={loss_py:.8f}, C++={loss_cpp:.8f}"
        )

        # Parameter values should match within tight fp32 tolerance
        for (n1, p1), (n2, p2) in zip(
            model_py.named_parameters(), model_cpp.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(
                p1.data,
                p2.data,
                atol=1e-6,
                rtol=1e-5,
                msg=f"Parameter {n1} diverged after 1 step",
            )

    @_CAME_C_SKIP
    @_CUDA_SKIP
    def test_multi_step_trajectory(self):
        """50 steps -> loss curves match within 1% relative error."""
        model_py, model_cpp, opt_py, opt_cpp = _make_training_pair()

        n_steps = 50

        # Run both optimizers on identical data streams (same seed_offset)
        losses_py = _train_n_steps(model_py, opt_py, LOSS_FN, n_steps, seed_offset=0)
        losses_cpp = _train_n_steps(model_cpp, opt_cpp, LOSS_FN, n_steps, seed_offset=0)

        # Every step's loss should match within 1% relative error
        max_rel_error = 0.0
        for i, (lp, lc) in enumerate(zip(losses_py, losses_cpp)):
            denom = max(abs(lp), abs(lc), 1e-8)
            rel_error = abs(lp - lc) / denom
            max_rel_error = max(max_rel_error, rel_error)
            assert rel_error < 0.01, (
                f"Loss diverged at step {i + 1}: Python={lp:.8f}, C++={lc:.8f}, "
                f"rel_error={rel_error:.6f} (max 1%)"
            )

        print(
            f"\n  [multi-step] {n_steps} steps, max relative loss error: {max_rel_error:.2e}"
        )

        # Also verify final parameter values match
        for (n1, p1), (n2, p2) in zip(
            model_py.named_parameters(), model_cpp.named_parameters()
        ):
            assert n1 == n2
            # Slightly looser tolerance after 50 steps (accumulated fp differences)
            torch.testing.assert_close(
                p1.data,
                p2.data,
                atol=1e-5,
                rtol=1e-3,
                msg=f"Parameter {n1} diverged after {n_steps} steps",
            )

    @_CAME_C_SKIP
    @_CUDA_SKIP
    def test_optimizer_state_consistency(self):
        """After 10 steps, all optimizer states (exp_avg, etc.) match."""
        model_py, model_cpp, opt_py, opt_cpp = _make_training_pair()

        n_steps = 10

        _train_n_steps(model_py, opt_py, LOSS_FN, n_steps, seed_offset=500)
        _train_n_steps(model_cpp, opt_cpp, LOSS_FN, n_steps, seed_offset=500)

        states_py = _collect_optimizer_states(opt_py, model_py)
        states_cpp = _collect_optimizer_states(opt_cpp, model_cpp)

        for param_name in states_py:
            assert param_name in states_cpp, f"Missing state for {param_name} in CAME_C"
            state_py = states_py[param_name]
            state_cpp = states_cpp[param_name]

            for key in state_py:
                assert key in state_cpp, (
                    f"Missing state key '{key}' for {param_name} in CAME_C"
                )
                val_py = state_py[key]
                val_cpp = state_cpp[key]

                if isinstance(val_py, torch.Tensor) and isinstance(
                    val_cpp, torch.Tensor
                ):
                    # exp_avg accumulates reassociation differences from
                    # torch.compile fusion in Python CAME; use a looser
                    # tolerance.  Factored second-moment states (sq_row/col,
                    # res_row/col) are computed per-step and stay tight.
                    if key == "exp_avg":
                        atol, rtol = 5e-5, 1e-2
                    else:
                        atol, rtol = 1e-5, 1e-3
                    torch.testing.assert_close(
                        val_py.float(),
                        val_cpp.float(),
                        atol=atol,
                        rtol=rtol,
                        msg=f"State '{key}' for {param_name} differs",
                    )
                else:
                    # Non-tensor state (e.g. step counter)
                    assert val_py == val_cpp, (
                        f"State '{key}' for {param_name}: Python={val_py}, C++={val_cpp}"
                    )

        print(
            f"\n  [state-check] All optimizer states consistent after {n_steps} steps"
        )

    @_CAME_C_SKIP
    @_CUDA_SKIP
    @pytest.mark.slow
    def test_performance_not_degraded(self):
        """CAME_C is not more than 2x slower than CAME (wall-clock, 50 steps).

        Both optimizers have JIT overhead on the first invocation (torch.compile
        for CAME, cpp_extension.load for CAME_C), so we run a warmup phase first.
        """
        n_warmup = 5
        n_benchmark = 50

        results = {}
        for label, OptClass in [("CAME (Python)", CAME), ("CAME_C (C++)", CAME_C)]:
            model = _make_base_model()
            optimizer = OptClass(model.parameters(), lr=LR, betas=BETAS)

            # Warmup: triggers JIT compilation (torch.compile for CAME)
            _train_n_steps(model, optimizer, LOSS_FN, n_warmup, seed_offset=1000)

            # Synchronize CUDA before timing
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            _train_n_steps(model, optimizer, LOSS_FN, n_benchmark, seed_offset=2000)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            results[label] = elapsed

        time_py = results["CAME (Python)"]
        time_cpp = results["CAME_C (C++)"]
        ratio = time_cpp / time_py

        print("\n  -- Performance comparison ----------------------------")
        print(f"  CAME  (Python) : {time_py * 1000:8.1f} ms  ({n_benchmark} steps)")
        print(f"  CAME_C (C++)   : {time_cpp * 1000:8.1f} ms  ({n_benchmark} steps)")
        print(f"  Ratio (C++/Python): {ratio:.3f}x")
        if ratio < 1.0:
            print(f"  CAME_C is {1 / ratio:.2f}x FASTER than CAME")
        else:
            print(f"  CAME_C is {ratio:.2f}x slower than CAME")
        print("  ------------------------------------------------------\n")

        # CAME_C should not be more than 2x slower
        assert ratio < 2.0, (
            f"CAME_C is {ratio:.2f}x slower than CAME (threshold: 2x). "
            f"Python: {time_py * 1000:.1f}ms, C++: {time_cpp * 1000:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
