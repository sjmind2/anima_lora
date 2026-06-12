"""
End-to-end training test for LoKR CUDA kernels.

Builds a small synthetic model with LokrModule instances mapped to real DiT
shapes, runs forward/backward/optimizer.step() for N steps, and verifies
that the CUDA kernel path and PyTorch fallback path produce equivalent
training trajectories.

Run with:
    pytest tests/test_lokr_cuda_e2e.py -v
"""

import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from networks.lora_modules.lokr_cpp_extension import _LOKR_KERNEL_AVAILABLE
if not _LOKR_KERNEL_AVAILABLE:
    pytest.skip("LoKR CUDA extension not available", allow_module_level=True)

from networks.lora_modules.lokr_cpp_extension import (
    KronLinearFnCUDA,
    KronLinearTwoStageFnCUDA,
)
from networks.lora_modules.lokr import (
    _KronLinearFn_PyTorch as KronLinearFnRef,
    _KronLinearTwoStageFn_PyTorch as KronLinearTwoStageFnRef,
)


class SyntheticLoKR(nn.Module):
    """Minimal model that mimics LoKR forward path for one Linear layer."""

    def __init__(self, fn_class, in_m, in_n, out_l, out_k, scalar=0.5):
        super().__init__()
        self.fn_class = fn_class
        self.w1 = nn.Parameter(torch.randn(out_l, in_m) * 0.01)
        self.w2 = nn.Parameter(torch.randn(out_k, in_n) * 0.01)
        self.register_buffer("scalar", torch.tensor(scalar))

    def forward(self, x):
        return self.fn_class.apply(x.contiguous(), self.w1, self.w2, self.scalar)


class SyntheticLoKRTwoStage(nn.Module):
    """Two-stage model with decompose_both=True."""

    def __init__(self, fn_class, in_m, in_n, out_l, out_k, r1, r2, scalar=0.5):
        super().__init__()
        self.fn_class = fn_class
        self.w1_a = nn.Parameter(torch.randn(out_l, r1) * 0.01)
        self.w1_b = nn.Parameter(torch.randn(r1, in_m) * 0.01)
        self.w2_a = nn.Parameter(torch.randn(out_k, r2) * 0.01)
        self.w2_b = nn.Parameter(torch.randn(r2, in_n) * 0.01)
        self.register_buffer("scalar", torch.tensor(scalar))

    def forward(self, x):
        return self.fn_class.apply(
            x.contiguous(), self.w1_a, self.w1_b, self.w2_a, self.w2_b, self.scalar
        )


def _train_steps(model, x_data, target_fn, n_steps=20, lr=1e-3):
    """Run N training steps and return loss history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        out = model(x_data)
        target = target_fn(x_data)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_e2e_single_stage():
    """Single-stage LoKR: CUDA vs PyTorch reference training trajectories."""
    B, in_m, in_n, out_l, out_k = 32, 8, 256, 8, 256
    dtype = torch.float32
    device = "cuda"

    torch.manual_seed(42)
    x_data = torch.randn(B, in_m * in_n, device=device, dtype=dtype)

    # Fixed target: same for both models
    torch.manual_seed(123)
    target_w1 = torch.randn(out_l, in_m, device=device, dtype=dtype)
    target_w2 = torch.randn(out_k, in_n, device=device, dtype=dtype)
    target_fn = lambda x: KronLinearFnRef.apply(x, target_w1, target_w2, torch.tensor(0.5, device=device))

    # Build identical models with different fn classes
    torch.manual_seed(0)
    model_ref = SyntheticLoKR(KronLinearFnRef, in_m, in_n, out_l, out_k).to(device)
    torch.manual_seed(0)
    model_cuda = SyntheticLoKR(KronLinearFnCUDA, in_m, in_n, out_l, out_k).to(device)

    # Verify initial parameters are identical
    torch.testing.assert_close(model_ref.w1, model_cuda.w1)
    torch.testing.assert_close(model_ref.w2, model_cuda.w2)

    losses_ref = _train_steps(model_ref, x_data, target_fn, n_steps=20)
    losses_cuda = _train_steps(model_cuda, x_data, target_fn, n_steps=20)

    # Loss curves should be very close
    for i, (lr_val, lc_val) in enumerate(zip(losses_ref, losses_cuda)):
        rel_err = abs(lr_val - lc_val) / (abs(lr_val) + 1e-8)
        assert rel_err < 0.01, f"Step {i}: loss ref={lr_val:.6f}, cuda={lc_val:.6f}, rel_err={rel_err:.6f}"

    # Final parameters should be close
    torch.testing.assert_close(model_ref.w1, model_cuda.w1, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(model_ref.w2, model_cuda.w2, atol=1e-4, rtol=1e-3)


def test_e2e_two_stage():
    """Two-stage LoKR: CUDA vs PyTorch reference training trajectories."""
    B, in_m, in_n, out_l, out_k, r1, r2 = 32, 8, 64, 8, 64, 8, 8
    dtype = torch.float32
    device = "cuda"

    torch.manual_seed(42)
    x_data = torch.randn(B, in_m * in_n, device=device, dtype=dtype)

    # Fixed target
    torch.manual_seed(123)
    t_w1a = torch.randn(out_l, r1, device=device, dtype=dtype)
    t_w1b = torch.randn(r1, in_m, device=device, dtype=dtype)
    t_w2a = torch.randn(out_k, r2, device=device, dtype=dtype)
    t_w2b = torch.randn(r2, in_n, device=device, dtype=dtype)
    target_fn = lambda x: KronLinearTwoStageFnRef.apply(
        x, t_w1a, t_w1b, t_w2a, t_w2b, torch.tensor(0.5, device=device)
    )

    torch.manual_seed(0)
    model_ref = SyntheticLoKRTwoStage(
        KronLinearTwoStageFnRef, in_m, in_n, out_l, out_k, r1, r2
    ).to(device)
    torch.manual_seed(0)
    model_cuda = SyntheticLoKRTwoStage(
        KronLinearTwoStageFnCUDA, in_m, in_n, out_l, out_k, r1, r2
    ).to(device)

    # Verify initial parameters are identical
    torch.testing.assert_close(model_ref.w1_a, model_cuda.w1_a)

    losses_ref = _train_steps(model_ref, x_data, target_fn, n_steps=20)
    losses_cuda = _train_steps(model_cuda, x_data, target_fn, n_steps=20)

    for i, (lr_val, lc_val) in enumerate(zip(losses_ref, losses_cuda)):
        rel_err = abs(lr_val - lc_val) / (abs(lr_val) + 1e-8)
        assert rel_err < 0.02, f"Step {i}: loss ref={lr_val:.6f}, cuda={lc_val:.6f}, rel_err={rel_err:.6f}"


def test_e2e_adamw_fused_compatible():
    """Verify CUDA kernels work correctly with AdamW fused optimizer."""
    B, in_m, in_n, out_l, out_k = 16, 8, 64, 8, 64
    device = "cuda"

    model = SyntheticLoKR(KronLinearFnCUDA, in_m, in_n, out_l, out_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    x = torch.randn(B, in_m * in_n, device=device)
    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"Loss not finite at step {step}"


def test_e2e_came_compatible():
    """Verify CUDA kernels work correctly with CAME optimizer."""
    B, in_m, in_n, out_l, out_k = 16, 8, 64, 8, 64
    device = "cuda"

    model = SyntheticLoKR(KronLinearFnCUDA, in_m, in_n, out_l, out_k).to(device)

    # Try CAME_C first, fall back to standard AdamW
    try:
        from library.training.came_cpp_extension import CAME_C
        optimizer = CAME_C(model.parameters(), lr=1e-3)
        opt_name = "CAME_C"
    except Exception:
        try:
            from library.training.came_optimizer import CAME
            optimizer = CAME(model.parameters(), lr=1e-3)
            opt_name = "CAME"
        except Exception:
            pytest.skip("CAME/CAME_C not available")

    x = torch.randn(B, in_m * in_n, device=device)
    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"Loss not finite at step {step} with {opt_name}"
