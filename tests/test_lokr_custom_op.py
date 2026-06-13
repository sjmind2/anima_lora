"""
Compile-compatibility tests for LoKr custom_op CUDA kernels.

Verifies that the torch.library.custom_op registration works correctly in:
  - Eager mode (forward + backward)
  - torch.compile mode (forward + backward, no graph breaks)
  - Gradient checkpointing recompute path (checkpoint + compile)

Reference: tests/test_lokr_kernel_equiv.py for the pure numerical equivalence
tests between CUDA kernels and the PyTorch reference.

Run with:
    pytest tests/test_lokr_custom_op.py -v
"""

import pytest
import torch
import torch.utils.checkpoint as cp

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from networks.lora_modules.lokr import (
    _KronLinearFn_PyTorch as KronLinearFnRef,
    _KronLinearTwoStageFn_PyTorch as KronLinearTwoStageFnRef,
    _kron1_delta,
    _kron2_delta,
)
try:
    from networks.lora_modules.lokr_cpp_extension import (
        KronLinearFnCUDA as KronLinearFnCUDA,
        KronLinearTwoStageFnCUDA as KronLinearTwoStageFnCUDA,
        _LOKR_KERNEL_AVAILABLE,
    )
    _CUDA_AVAILABLE = _LOKR_KERNEL_AVAILABLE
except Exception:
    _CUDA_AVAILABLE = False

if not _CUDA_AVAILABLE:
    pytest.skip("LoKR CUDA extension not available", allow_module_level=True)


# ============================================================================
# Test shapes (same as test_lokr_kernel_equiv.py)
# ============================================================================

# (name, B, in_m, in_n, out_l, out_k) — single-stage
ANIMA_SHAPES = [
    ("qkv_proj",     64, 8, 256, 8, 768),
    ("output_proj",  64, 8, 256, 8, 256),
    ("tiny",          4, 2,   4, 3,   5),
]

# (name, B, in_m, in_n, out_l, out_k, r1, r2) — two-stage
TWOSTAGE_SHAPES = [
    ("qkv_real",     64, 8, 256, 8, 768, 8, 8),
    ("down_real",    64, 8, 1024, 8, 256, 8, 8),
    ("tiny",          4, 2,   4, 3,   5, 2, 2),
]


def _make_inputs(B, in_m, in_n, out_l, out_k, dtype, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(B, in_m * in_n, device="cuda", dtype=dtype, requires_grad=True)
    w1 = torch.randn(out_l, in_m, device="cuda", dtype=dtype, requires_grad=True)
    w2 = torch.randn(out_k, in_n, device="cuda", dtype=dtype, requires_grad=True)
    scalar = torch.tensor(0.5, device="cuda")
    return x, w1, w2, scalar


def _make_inputs_twostage(B, in_m, in_n, out_l, out_k, r1, r2, dtype, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(B, in_m * in_n, device="cuda", dtype=dtype, requires_grad=True)
    w1_a = torch.randn(out_l, r1, device="cuda", dtype=dtype, requires_grad=True)
    w1_b = torch.randn(r1, in_m, device="cuda", dtype=dtype, requires_grad=True)
    w2_a = torch.randn(out_k, r2, device="cuda", dtype=dtype, requires_grad=True)
    w2_b = torch.randn(r2, in_n, device="cuda", dtype=dtype, requires_grad=True)
    scalar = torch.tensor(0.5, device="cuda")
    return x, w1_a, w1_b, w2_a, w2_b, scalar


# Tolerances matching test_lokr_kernel_equiv.py
def _atol(dtype, fwd=True):
    if fwd:
        return 1e-3 if dtype == torch.float32 else 1e-2
    else:
        return 1e-3 if dtype == torch.float32 else 5e-2


# ============================================================================
# Eager-mode tests: custom_op vs PyTorch reference
# ============================================================================

@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k", ANIMA_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kron1_fwd_eager(name, B, in_m, in_n, out_l, out_k, dtype):
    """custom_op forward (eager) matches PyTorch reference."""
    x1, w1_1, w2_1, s1 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    x2, w1_2, w2_2, s2 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    out_ref = KronLinearFnRef.apply(x1, w1_1, w2_1, s1)
    out_cuda = KronLinearFnCUDA.apply(x2, w1_2, w2_2, s2)
    torch.testing.assert_close(out_ref, out_cuda, atol=_atol(dtype), rtol=_atol(dtype))


@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k", ANIMA_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kron1_bwd_eager(name, B, in_m, in_n, out_l, out_k, dtype):
    """custom_op backward (eager) matches PyTorch reference."""
    x1, w1_1, w2_1, s1 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    x2, w1_2, w2_2, s2 = _make_inputs(B, in_m, in_n, out_l, out_k, dtype)
    out_ref = KronLinearFnRef.apply(x1, w1_1, w2_1, s1)
    out_cuda = KronLinearFnCUDA.apply(x2, w1_2, w2_2, s2)
    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_cuda.backward(grad_out)
    torch.testing.assert_close(x1.grad, x2.grad, atol=_atol(dtype, fwd=False), rtol=_atol(dtype, fwd=False))
    torch.testing.assert_close(w1_1.grad, w1_2.grad, atol=5e-3 if dtype == torch.float32 else 5e-2,
                               rtol=5e-3 if dtype == torch.float32 else 5e-2)


@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k, r1, r2", TWOSTAGE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kron2_fwd_eager(name, B, in_m, in_n, out_l, out_k, r1, r2, dtype):
    """Two-stage custom_op forward (eager) matches PyTorch reference."""
    x1, w1a_1, w1b_1, w2a_1, w2b_1, s1 = _make_inputs_twostage(B, in_m, in_n, out_l, out_k, r1, r2, dtype)
    x2, w1a_2, w1b_2, w2a_2, w2b_2, s2 = _make_inputs_twostage(B, in_m, in_n, out_l, out_k, r1, r2, dtype)
    out_ref = KronLinearTwoStageFnRef.apply(x1, w1a_1, w1b_1, w2a_1, w2b_1, s1)
    out_cuda = KronLinearTwoStageFnCUDA.apply(x2, w1a_2, w1b_2, w2a_2, w2b_2, s2)
    torch.testing.assert_close(out_ref, out_cuda, atol=_atol(dtype), rtol=_atol(dtype))


@pytest.mark.parametrize("name, B, in_m, in_n, out_l, out_k, r1, r2", TWOSTAGE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kron2_bwd_eager(name, B, in_m, in_n, out_l, out_k, r1, r2, dtype):
    """Two-stage custom_op backward (eager) matches PyTorch reference."""
    x1, w1a_1, w1b_1, w2a_1, w2b_1, s1 = _make_inputs_twostage(B, in_m, in_n, out_l, out_k, r1, r2, dtype)
    x2, w1a_2, w1b_2, w2a_2, w2b_2, s2 = _make_inputs_twostage(B, in_m, in_n, out_l, out_k, r1, r2, dtype)
    out_ref = KronLinearTwoStageFnRef.apply(x1, w1a_1, w1b_1, w2a_1, w2b_1, s1)
    out_cuda = KronLinearTwoStageFnCUDA.apply(x2, w1a_2, w1b_2, w2a_2, w2b_2, s2)
    torch.manual_seed(99)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_cuda.backward(grad_out)
    torch.testing.assert_close(x1.grad, x2.grad, atol=_atol(dtype, fwd=False), rtol=_atol(dtype, fwd=False))


# ============================================================================
# Compile-mode tests: custom_op inside torch.compile (CUDA + inductor)
# ============================================================================

def test_kron1_fwd_compiled():
    """custom_op forward works inside torch.compile on CUDA."""
    x, w1, w2, scalar = _make_inputs(64, 8, 256, 8, 768, torch.bfloat16)
    x_ref = x.detach().clone().requires_grad_(True)
    w1_ref = w1.detach().clone().requires_grad_(True)
    w2_ref = w2.detach().clone().requires_grad_(True)

    out_ref = KronLinearFnRef.apply(x_ref, w1_ref, w2_ref, scalar)

    fn = torch.compile(KronLinearFnCUDA.apply, backend="inductor", fullgraph=True)
    out_compiled = fn(x, w1, w2, scalar)

    torch.testing.assert_close(out_compiled, out_ref, atol=1e-2, rtol=1e-2)


def test_kron1_bwd_compiled():
    """custom_op forward + backward works inside torch.compile on CUDA."""
    x, w1, w2, scalar = _make_inputs(64, 8, 256, 8, 768, torch.bfloat16)

    fn = torch.compile(KronLinearFnCUDA.apply, backend="inductor", fullgraph=True)
    out = fn(x, w1, w2, scalar)
    out.sum().backward()
    assert x.grad is not None, "x.grad should be populated"
    assert w1.grad is not None, "w1.grad should be populated"


def test_kron2_fwd_compiled():
    """Two-stage custom_op forward works inside torch.compile on CUDA."""
    x, w1_a, w1_b, w2_a, w2_b, scalar = _make_inputs_twostage(64, 8, 256, 8, 768, 8, 8, torch.bfloat16)
    x_ref = x.detach().clone().requires_grad_(True)

    fn = torch.compile(KronLinearTwoStageFnCUDA.apply, backend="inductor", fullgraph=True)
    out = fn(x, w1_a, w1_b, w2_a, w2_b, scalar)
    assert out.shape == (64, 8 * 768), f"Unexpected output shape: {out.shape}"


def test_kron2_bwd_compiled():
    """Two-stage custom_op fwd+bwd works inside torch.compile on CUDA."""
    x, w1_a, w1_b, w2_a, w2_b, scalar = _make_inputs_twostage(64, 8, 256, 8, 768, 8, 8, torch.bfloat16)

    fn = torch.compile(KronLinearTwoStageFnCUDA.apply, backend="inductor", fullgraph=True)
    out = fn(x, w1_a, w1_b, w2_a, w2_b, scalar)
    out.sum().backward()
    assert x.grad is not None, "x.grad should be populated"
    assert w1_a.grad is not None, "w1_a.grad should be populated"
    assert w2_a.grad is not None, "w2_a.grad should be populated"


# ============================================================================
# Gradient checkpointing + compile: the critical GC recompute path
# ============================================================================

def test_gc_recompute_kron1():
    """custom_op works inside torch.utils.checkpoint + torch.compile."""
    x, w1, w2, scalar = _make_inputs(64, 8, 256, 8, 768, torch.bfloat16)

    compiled_fn = torch.compile(KronLinearFnCUDA.apply, backend="inductor", fullgraph=True)

    out = cp.checkpoint(compiled_fn, x, w1, w2, scalar, use_reentrant=False)
    out.sum().backward()
    assert x.grad is not None, "x.grad should be populated after GC recompute"
    assert w1.grad is not None, "w1.grad should be populated after GC recompute"


def test_gc_recompute_kron2():
    """Two-stage custom_op works inside checkpoint + compile."""
    x, w1_a, w1_b, w2_a, w2_b, scalar = _make_inputs_twostage(64, 8, 256, 8, 768, 8, 8, torch.bfloat16)

    compiled_fn = torch.compile(KronLinearTwoStageFnCUDA.apply, backend="inductor", fullgraph=True)

    out = cp.checkpoint(compiled_fn, x, w1_a, w1_b, w2_a, w2_b, scalar, use_reentrant=False)
    out.sum().backward()
    assert x.grad is not None
    assert w1_a.grad is not None
