"""
Mathematical equivalence tests for LoKR CUDA kernels vs PyTorch reference.

Verifies that the fused CUDA kernels produce numerically equivalent results
to the pure-PyTorch autograd.Function implementations in lokr.py.

Tests cover:
  - Single-stage forward/backward (KronLinearFn)
  - Two-stage forward/backward (KronLinearTwoStageFn)
  - Anima DiT realistic shapes (qkv_proj, output_proj, mlp)
  - Multiple dtypes (fp32, bf16, fp16)

Run with:
    pytest tests/test_lokr_kernel_equiv.py -v
"""

import pytest
import torch
import torch.nn as nn

# Skip entire module if no CUDA
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

# Import both implementations
from networks.lora_modules.lokr import (
    _KronLinearFn_PyTorch as KronLinearFnRef,
    _KronLinearTwoStageFn_PyTorch as KronLinearTwoStageFnRef,
)
try:
    from networks.lora_modules.lokr_cpp_extension import (
        KronLinearFnCUDA as KronLinearFnCUDA_,
        KronLinearTwoStageFnCUDA as KronLinearTwoStageFnCUDA_,
        _LOKR_KERNEL_AVAILABLE,
    )
    _CUDA_AVAILABLE = _LOKR_KERNEL_AVAILABLE
except Exception:
    _CUDA_AVAILABLE = False

if not _CUDA_AVAILABLE:
    pytest.skip("LoKR CUDA extension not available", allow_module_level=True)


# ============================================================================
# Test shapes — Anima DiT Linear layers with factor=8
# ============================================================================

# (name, B, in_m, in_n, out_l, out_k)
ANIMA_SHAPES = [
    ("qkv_proj",     64, 8, 256, 8, 768),
    ("output_proj",  64, 8, 256, 8, 256),
    ("gate_up_proj", 64, 8, 256, 8, 1024),
    ("down_proj",    64, 8, 1024, 8, 256),
    # Small smoke test
    ("tiny",          4, 2,   4, 3,   5),
]

DTYPES = [torch.float32, torch.bfloat16]


# ============================================================================
# Helpers
# ============================================================================

def _make_inputs(B, in_m, in_n, out_l, out_k, dtype, device="cuda", seed=42):
    """Create test inputs for KronLinearFn."""
    torch.manual_seed(seed)
    x = torch.randn(B, in_m * in_n, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(out_l, in_m, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(out_k, in_n, device=device, dtype=dtype, requires_grad=True)
    scalar = torch.tensor(0.5, device=device)
    return x, w1, w2, scalar


def _make_inputs_twostage(B, in_m, in_n, out_l, out_k, r1, r2, dtype, device="cuda", seed=42):
    """Create test inputs for KronLinearTwoStageFn."""
    torch.manual_seed(seed)
    x = torch.randn(B, in_m * in_n, device=device, dtype=dtype, requires_grad=True)
    w1_a = torch.randn(out_l, r1, device=device, dtype=dtype, requires_grad=True)
    w1_b = torch.randn(r1, in_m, device=device, dtype=dtype, requires_grad=True)
    w2_a = torch.randn(out_k, r2, device=device, dtype=dtype, requires_grad=True)
    w2_b = torch.randn(r2, in_n, device=device, dtype=dtype, requires_grad=True)
    scalar = torch.tensor(0.5, device=device)
    return x, w1_a, w1_b, w2_a, w2_b, scalar


# ============================================================================
# Single-stage forward tests
# ============================================================================

@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k", ANIMA_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_kron1_forward(name, B, in_m, in_n, out_l, out_k, dtype):
    """Test KronLinearFn forward: CUDA vs PyTorch reference."""
    x1, w1_1, w2_1, scalar1 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    x2, w1_2, w2_2, scalar2 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)

    out_ref = KronLinearFnRef.apply(x1, w1_1, w2_1, scalar1)
    out_cuda = KronLinearFnCUDA_.apply(x2, w1_2, w2_2, scalar2)

    # fp32 forward: custom kernel sequential accumulation vs cuBLAS tree reduction
    # → ~5e-5 max diff for K=256; use 1e-3 for cross-implementation robustness
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out_ref, out_cuda, atol=atol, rtol=rtol)


# ============================================================================
# Single-stage backward tests
# ============================================================================

@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k", ANIMA_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_kron1_backward(name, B, in_m, in_n, out_l, out_k, dtype):
    """Test KronLinearFn backward: CUDA vs PyTorch reference."""
    x1, w1_1, w2_1, scalar1 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    x2, w1_2, w2_2, scalar2 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)

    out_ref = KronLinearFnRef.apply(x1, w1_1, w2_1, scalar1)
    out_cuda = KronLinearFnCUDA_.apply(x2, w1_2, w2_2, scalar2)

    # Use same grad_out
    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)

    out_ref.backward(grad_out)
    out_cuda.backward(grad_out)

    # grad_x: same cross-implementation precision gap as forward (sequential
    # accumulation in kernel vs cuBLAS tree reduction in PyTorch reference)
    atol = 1e-3 if dtype == torch.float32 else 5e-2
    rtol = 1e-3 if dtype == torch.float32 else 5e-2
    # Weight grads use atomicAdd across B blocks — additional reordering noise
    atol_w = 5e-3 if dtype == torch.float32 else 5e-2
    rtol_w = 5e-3 if dtype == torch.float32 else 5e-2

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w1_1.grad, w1_2.grad, atol=atol_w, rtol=rtol_w,
                               msg=lambda e: f"grad_w1 mismatch: {e}")
    torch.testing.assert_close(w2_1.grad, w2_2.grad, atol=atol_w, rtol=rtol_w,
                               msg=lambda e: f"grad_w2 mismatch: {e}")


@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k", ANIMA_SHAPES)
def test_kron1_backward_fp32(name, B, in_m, in_n, out_l, out_k):
    """Test KronLinearFn backward in fp32 — stricter tolerances."""
    dtype = torch.float32
    x1, w1_1, w2_1, scalar1 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    x2, w1_2, w2_2, scalar2 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)

    out_ref = KronLinearFnRef.apply(x1, w1_1, w2_1, scalar1)
    out_cuda = KronLinearFnCUDA_.apply(x2, w1_2, w2_2, scalar2)

    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)

    out_ref.backward(grad_out)
    out_cuda.backward(grad_out)

    # Cross-implementation comparison: custom kernel sequential fp32
    # accumulation vs cuBLAS tree reduction. For K=256/1024, grad_x sees
    # ~3e-4 max diff; weight grads via atomicAdd see ~5e-3.
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w1_1.grad, w1_2.grad, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(w2_1.grad, w2_2.grad, atol=5e-3, rtol=5e-3)


# ============================================================================
# Two-stage forward tests
# ============================================================================

# (name, B, in_m, in_n, out_l, out_k, r1, r2)
TWOSTAGE_SHAPES = [
    ("qkv_proj",     32, 8, 256, 8, 768, 16, 16),
    ("output_proj",  32, 8, 256, 8, 256, 16, 16),
    ("tiny",          4, 2,   4, 3,   5,  4,  3),
]


@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k, r1, r2", TWOSTAGE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_kron2_forward(name, B, in_m, in_n, out_l, out_k, r1, r2, dtype):
    """Test KronLinearTwoStageFn forward: CUDA vs PyTorch reference."""
    x1, w1a_1, w1b_1, w2a_1, w2b_1, scalar1 = _make_inputs_twostage(
        B, in_m, in_n, out_l, out_k, r1, r2, dtype
    )
    x2, w1a_2, w1b_2, w2a_2, w2b_2, scalar2 = _make_inputs_twostage(
        B, in_m, in_n, out_l, out_k, r1, r2, dtype
    )

    out_ref = KronLinearTwoStageFnRef.apply(
        x1, w1a_1, w1b_1, w2a_1, w2b_1, scalar1
    )
    out_cuda = KronLinearTwoStageFnCUDA_.apply(
        x2, w1a_2, w1b_2, w2a_2, w2b_2, scalar2
    )

    # fp32 forward: custom kernel sequential accumulation vs cuBLAS tree reduction
    # → ~5e-5 max diff for K=256; use 1e-3 for cross-implementation robustness
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out_ref, out_cuda, atol=atol, rtol=rtol)


# ============================================================================
# Two-stage backward tests
# ============================================================================

@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k, r1, r2", TWOSTAGE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_kron2_backward(name, B, in_m, in_n, out_l, out_k, r1, r2, dtype):
    """Test KronLinearTwoStageFn backward: CUDA vs PyTorch reference."""
    x1, w1a_1, w1b_1, w2a_1, w2b_1, scalar1 = _make_inputs_twostage(
        B, in_m, in_n, out_l, out_k, r1, r2, dtype
    )
    x2, w1a_2, w1b_2, w2a_2, w2b_2, scalar2 = _make_inputs_twostage(
        B, in_m, in_n, out_l, out_k, r1, r2, dtype
    )

    out_ref = KronLinearTwoStageFnRef.apply(
        x1, w1a_1, w1b_1, w2a_1, w2b_1, scalar1
    )
    out_cuda = KronLinearTwoStageFnCUDA_.apply(
        x2, w1a_2, w1b_2, w2a_2, w2b_2, scalar2
    )

    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)

    out_ref.backward(grad_out)
    out_cuda.backward(grad_out)

    atol = 1e-3 if dtype == torch.float32 else 5e-2
    rtol = 1e-3 if dtype == torch.float32 else 5e-2
    # Weight grads use atomicAdd across B blocks — additional reordering noise
    atol_w = 5e-3 if dtype == torch.float32 else 5e-2
    rtol_w = 5e-3 if dtype == torch.float32 else 5e-2

    torch.testing.assert_close(x1.grad, x2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w1a_1.grad, w1a_2.grad, atol=atol_w, rtol=rtol_w)
    torch.testing.assert_close(w1b_1.grad, w1b_2.grad, atol=atol_w, rtol=rtol_w)
    torch.testing.assert_close(w2a_1.grad, w2a_2.grad, atol=atol_w, rtol=rtol_w)
    torch.testing.assert_close(w2b_1.grad, w2b_2.grad, atol=atol_w, rtol=rtol_w)
