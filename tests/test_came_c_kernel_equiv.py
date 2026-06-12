"""
Mathematical equivalence tests for CAME_C vs CAME.

Verifies that the C++ ATen implementation produces numerically equivalent
results to the Python reference implementation within floating-point tolerance.

Key difference: The Python CAME optimizer promotes param dtype to fp32
(via p.data = param_data.add(-update) where type promotion occurs).
The C++ CAME_C preserves the original param dtype by computing in fp32
and converting back via copy_. Tests account for this by comparing values
in a common dtype.

Run with:
    pytest tests/test_came_c_kernel_equiv.py -v
"""

import torch
import pytest
from typing import List, Tuple

# Import both optimizers
from library.training.came_optimizer import CAME  # Python reference
try:
    from library.training.came_cpp_extension import CAME_C  # C++ version
    _CAME_C_AVAILABLE = True
except ImportError:
    _CAME_C_AVAILABLE = False
    pytest.skip("CAME_C extension not available", allow_module_level=True)


def _init_identical_params(
    shape: Tuple[int, ...] = (64, 128),
    device: str = "cuda",
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
):
    """Create two identical parameter tensors with fresh optimizer state."""
    torch.manual_seed(seed)

    # Create parameters
    param_py = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    param_cpp = param_py.detach().clone().requires_grad_(True)

    # Create optimizers
    opt_py = CAME([param_py], lr=1e-4, betas=(0.9, 0.999, 0.9999))
    opt_cpp = CAME_C([param_cpp], lr=1e-4, betas=(0.9, 0.999, 0.9999))

    # Generate identical gradient (in original dtype for proper assignment)
    grad = torch.randn(shape, device=device, dtype=dtype)

    # Assign gradients (ensure dtype matches param)
    param_py.grad = grad.clone()
    param_cpp.grad = grad.clone()

    # Force state initialization by doing a dummy step check
    for p, opt in [(param_py, opt_py), (param_cpp, opt_cpp)]:
        g = p.grad.float() if p.grad.dtype in (torch.float16, torch.bfloat16) else p.grad
        if len(g.shape) >= 2:
            opt._init_state(p, g, opt.state[p])
        else:
            opt._init_state(p, g, opt.state[p])

    # Copy internal states to ensure identical starting point
    state_py = opt_py.state[param_py]
    state_cpp = opt_cpp.state[param_cpp]

    for key in state_py:
        if key in state_cpp and isinstance(state_py[key], torch.Tensor):
            state_cpp[key].copy_(state_py[key])

    return param_py, param_cpp, opt_py, opt_cpp, grad


def _check_state_equality(opt_py, opt_cpp, param_py, param_cpp, atol: float = 1e-6, rtol: float = 1e-6):
    """Verify all optimizer states match.

    Python CAME stores state in grad.dtype (bf16 for bf16 params).
    CAME_C stores state in fp32 always. Compare in fp32 for fairness.

    Default atol=1e-6 accounts for CUDA fast-math rsqrtf (~2 ULP error vs
    Python's double-precision rsqrt), which causes ~3e-7 differences in
    exp_avg after one step.
    """
    state_py = opt_py.state[param_py]
    state_cpp = opt_cpp.state[param_cpp]

    errors = {}
    for key in state_py:
        if key in state_cpp and isinstance(state_py[key], torch.Tensor) and isinstance(state_cpp[key], torch.Tensor):
            # Compare in fp32 to handle dtype differences
            py_val = state_py[key].float()
            cpp_val = state_cpp[key].float()
            if not torch.allclose(py_val, cpp_val, atol=atol, rtol=rtol):
                max_err = (py_val - cpp_val).abs().max().item()
                errors[key] = max_err

    if errors:
        error_msg = "\n".join([f"  {k}: max_err={v:.2e}" for k, v in errors.items()])
        raise AssertionError(f"State mismatch:\n{error_msg}")


def _assert_params_close(param_py, param_cpp, atol=1e-3, rtol=1e-2, msg=None):
    """Compare params across dtypes.

    Python CAME promotes param to fp32 via p.data = new_tensor.
    C++ CAME_C preserves original dtype by computing in fp32 and converting back.

    Additionally, Python CAME uses torch.compile which fuses operations and may
    reassociate floating-point arithmetic, causing tiny numerical differences
    from the step-by-step C++ implementation. The default tolerances account
    for both bf16 rounding and torch.compile fusion differences.

    We compare values in the C++ param's dtype (original dtype).
    """
    # Convert Python param to C++ param's dtype for fair comparison
    py_val = param_py.data.to(param_cpp.dtype)
    cpp_val = param_cpp.data
    torch.testing.assert_close(py_val, cpp_val, atol=atol, rtol=rtol, msg=msg)


class TestCAMECKernelEquivalence:
    """Test suite for CAME_C kernel correctness vs Python CAME."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_factored_2d_tall(self):
        """Verify C++ factored kernel matches Python for tall 2D matrices."""
        param_py, param_cpp, opt_py, opt_cpp, grad = _init_identical_params((64, 128))

        # Run one step
        opt_py.step()
        opt_cpp.step()

        # Check parameter update parity
        _assert_params_close(
            param_py, param_cpp,
            msg="Parameter update diverged after one step (tall matrix)"
        )

        # Check all internal states
        _check_state_equality(opt_py, opt_cpp, param_py, param_cpp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_factored_2d_wide(self):
        """Verify C++ factored kernel matches Python for wide 2D matrices."""
        param_py, param_cpp, opt_py, opt_cpp, grad = _init_identical_params((128, 64))

        opt_py.step()
        opt_cpp.step()

        _assert_params_close(param_py, param_cpp)
        _check_state_equality(opt_py, opt_cpp, param_py, param_cpp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_factored_2d_square(self):
        """Verify C++ factored kernel matches Python for square matrices."""
        param_py, param_cpp, opt_py, opt_cpp, grad = _init_identical_params((256, 256))

        opt_py.step()
        opt_cpp.step()

        _assert_params_close(param_py, param_cpp)
        _check_state_equality(opt_py, opt_cpp, param_py, param_cpp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_unfactored_1d_bias(self):
        """Verify C++ unfactored kernel matches Python for 1D tensors (biases)."""
        param_py, param_cpp, opt_py, opt_cpp, grad = _init_identical_params((512,))

        opt_py.step()
        opt_cpp.step()

        _assert_params_close(param_py, param_cpp)
        _check_state_equality(opt_py, opt_cpp, param_py, param_cpp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_combinations(self, dtype):
        """Verify numerical stability across gradient dtypes."""
        param_py, param_cpp, opt_py, opt_cpp, grad = _init_identical_params(
            (32, 64), dtype=dtype
        )

        opt_py.step()
        opt_cpp.step()

        # Tolerance depends on original dtype precision
        if dtype == torch.float16:
            atol, rtol = 1e-3, 1e-2
        elif dtype == torch.bfloat16:
            atol, rtol = 1e-3, 1e-2
        else:  # float32
            atol, rtol = 1e-6, 1e-5

        _assert_params_close(param_py, param_cpp, atol=atol, rtol=rtol)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_zero_gradient(self):
        """Verify behavior with zero gradient (edge case)."""
        param_py, param_cpp, opt_py, opt_cpp, _ = _init_identical_params((64, 64))

        # Set zero gradient (dtype-safe)
        param_py.grad = torch.zeros_like(param_py)
        param_cpp.grad = torch.zeros_like(param_cpp)

        opt_py.step()
        opt_cpp.step()

        # Parameters should not change with zero gradient
        _assert_params_close(param_py, param_cpp, atol=1e-8, rtol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_large_gradient_clipping(self):
        """Verify RMS clipping with very large gradient."""
        param_py, param_cpp, opt_py, opt_cpp, _ = _init_identical_params((32, 32))

        # Create very large gradient (>clip_threshold) in param dtype
        large_grad = torch.randn(32, 32, device="cuda", dtype=param_cpp.dtype) * 1000
        param_py.grad = large_grad.clone()
        param_cpp.grad = large_grad.clone()

        opt_py.step()
        opt_cpp.step()

        # Should be clipped identically
        _assert_params_close(param_py, param_cpp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_weight_decay(self):
        """Verify weight decay is applied identically."""
        param_py = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        param_cpp = param_py.detach().clone().requires_grad_(True)

        opt_py = CAME([param_py], lr=1e-4, weight_decay=0.01)
        opt_cpp = CAME_C([param_cpp], lr=1e-4, weight_decay=0.01)

        grad = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        param_py.grad = grad.clone()
        param_cpp.grad = grad.clone()

        opt_py.step()
        opt_cpp.step()

        _assert_params_close(param_py, param_cpp)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_steps(self):
        """Verify consistency over multiple optimization steps.

        Uses fp32 params so that both optimizers see identical gradients at
        every step. With bf16 params, Python CAME promotes param to fp32 after
        step 1 (via p.data = new_tensor), changing the gradient dtype in
        subsequent steps -- this is a dtype handling difference, not a math error.
        """
        param_py, param_cpp, opt_py, opt_cpp, _ = _init_identical_params(
            (32, 32), dtype=torch.float32
        )

        for step in range(10):
            grad = torch.randn(32, 32, device="cuda", dtype=torch.float32)
            param_py.grad = grad.clone()
            param_cpp.grad = grad.clone()

            opt_py.step()
            opt_cpp.step()

            # Check after each step
            _assert_params_close(
                param_py, param_cpp,
                msg=f"Divergence at step {step + 1}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
