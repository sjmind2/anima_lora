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
from typing import Tuple

# Import both optimizers
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

if not (_CAME_C_AVAILABLE and _CAME_C_EXT_OK):
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
        g = (
            p.grad.float()
            if p.grad.dtype in (torch.float16, torch.bfloat16)
            else p.grad
        )
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


def _check_state_equality(
    opt_py, opt_cpp, param_py, param_cpp, atol: float = 1e-6, rtol: float = 1e-6
):
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
        if (
            key in state_cpp
            and isinstance(state_py[key], torch.Tensor)
            and isinstance(state_cpp[key], torch.Tensor)
        ):
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
            param_py,
            param_cpp,
            msg="Parameter update diverged after one step (tall matrix)",
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
        param_py = torch.randn(
            (64, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
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
                param_py, param_cpp, msg=f"Divergence at step {step + 1}"
            )


class TestCAMECBatchedEquivalence:
    """Test batched (shape-grouped) path vs single-param path.

    Strategy: put each param in its own param_group for single-path,
    all params in one group for batched-path. The math is identical;
    only kernel launch strategy differs.
    """

    @staticmethod
    def _make_paired(n, shape, dtype=torch.bfloat16, seed=42):
        """Create (params_single, params_batched, opt_single, opt_batched)."""
        torch.manual_seed(seed)
        refs = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(n)]
        params_s = [r.clone().requires_grad_(True) for r in refs]
        params_b = [r.clone().requires_grad_(True) for r in refs]

        opt_s = CAME_C(
            [{"params": [p]} for p in params_s],
            lr=1e-4,
            betas=(0.9, 0.999, 0.9999),
        )
        opt_b = CAME_C(params_b, lr=1e-4, betas=(0.9, 0.999, 0.9999))
        return params_s, params_b, opt_s, opt_b

    @staticmethod
    def _assign_grads(params, seed=100):
        """Assign identical random gradients."""
        torch.manual_seed(seed)
        for p in params:
            p.grad = torch.randn_like(p)

    @staticmethod
    def _compare(ps_list, pb_list, opt_s, opt_b, atol=1e-6, rtol=1e-5):
        """Compare params and states between single and batched paths."""
        for i, (ps, pb) in enumerate(zip(ps_list, pb_list)):
            torch.testing.assert_close(
                ps.data.to(torch.float32),
                pb.data.to(torch.float32),
                atol=atol,
                rtol=rtol,
                msg=f"Param {i} mismatch",
            )
            # Compare states
            ss = opt_s.state[ps]
            sb = opt_b.state[pb]
            for key in (
                "exp_avg",
                "exp_avg_sq_row",
                "exp_avg_sq_col",
                "exp_avg_res_row",
                "exp_avg_res_col",
                "exp_avg_sq",
            ):
                if key in ss and key in sb:
                    torch.testing.assert_close(
                        ss[key],
                        sb[key],
                        atol=atol,
                        rtol=rtol,
                        msg=f"State[{key}] mismatch for param {i}",
                    )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_factored_wide(self):
        """8 wide matrices (32, 2048) — batched vs single."""
        ps, pb, os_, ob = self._make_paired(8, (32, 2048))
        self._assign_grads(ps, seed=100)
        self._assign_grads(pb, seed=100)
        os_.step()
        ob.step()
        self._compare(ps, pb, os_, ob)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_factored_tall(self):
        """8 tall matrices (2048, 32) — batched vs single."""
        ps, pb, os_, ob = self._make_paired(8, (2048, 32))
        self._assign_grads(ps, seed=200)
        self._assign_grads(pb, seed=200)
        os_.step()
        ob.step()
        self._compare(ps, pb, os_, ob)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_factored_small(self):
        """8 small matrices (64, 32) — typical LoKR shapes."""
        ps, pb, os_, ob = self._make_paired(8, (64, 32))
        self._assign_grads(ps, seed=300)
        self._assign_grads(pb, seed=300)
        os_.step()
        ob.step()
        self._compare(ps, pb, os_, ob)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_unfactored(self):
        """8 1D params (512,) — batched vs single."""
        ps, pb, os_, ob = self._make_paired(8, (512,))
        self._assign_grads(ps, seed=400)
        self._assign_grads(pb, seed=400)
        os_.step()
        ob.step()
        self._compare(ps, pb, os_, ob)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_fp32_params(self):
        """Batched path with fp32 params (no dtype conversion)."""
        ps, pb, os_, ob = self._make_paired(8, (64, 128), dtype=torch.float32)
        self._assign_grads(ps, seed=500)
        self._assign_grads(pb, seed=500)
        os_.step()
        ob.step()
        self._compare(ps, pb, os_, ob, atol=1e-7, rtol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_multi_step(self):
        """50-step trajectory — batched vs single should stay identical."""
        ps, pb, os_, ob = self._make_paired(8, (32, 256))
        for step in range(50):
            self._assign_grads(ps, seed=600 + step)
            self._assign_grads(pb, seed=600 + step)
            os_.step()
            ob.step()
        self._compare(ps, pb, os_, ob)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_mixed_shapes(self):
        """Multiple shape groups in one step — simulates real DiT."""
        torch.manual_seed(42)
        shapes = [(32, 2048), (32, 1024), (2048, 32), (6144, 32), (64, 32)]
        refs_s = {}
        refs_b = {}
        for shp in shapes:
            refs_s[shp] = [
                torch.randn(
                    *shp, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                for _ in range(4)
            ]
            refs_b[shp] = [p.detach().clone().requires_grad_(True) for p in refs_s[shp]]

        all_s = [p for shp in shapes for p in refs_s[shp]]
        all_b = [p for shp in shapes for p in refs_b[shp]]
        opt_s = CAME_C([{"params": [p]} for p in all_s], lr=1e-4)
        opt_b = CAME_C(all_b, lr=1e-4)

        # Assign identical gradients via separate seeded passes
        torch.manual_seed(700)
        for p in all_s:
            p.grad = torch.randn_like(p)
        torch.manual_seed(700)
        for p in all_b:
            p.grad = torch.randn_like(p)

        opt_s.step()
        opt_b.step()

        flat_s = [p for shp in shapes for p in refs_s[shp]]
        flat_b = [p for shp in shapes for p in refs_b[shp]]
        self._compare(flat_s, flat_b, opt_s, opt_b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_state_dict_roundtrip(self):
        """Checkpoint save/load — batched path must recover correctly.

        PyTorch's load_state_dict casts state tensors to the param dtype (bf16),
        causing precision loss. This is standard PyTorch behavior, not a bug.
        We use bf16-appropriate tolerances for the post-roundtrip comparison.
        """
        ps, pb, os_, ob = self._make_paired(6, (32, 128))
        self._assign_grads(ps, seed=800)
        self._assign_grads(pb, seed=800)
        os_.step()
        ob.step()

        # Save batched optimizer state
        sd = ob.state_dict()
        # Create fresh optimizer and load
        pb2 = [p.detach().clone().requires_grad_(True) for p in pb]
        # Copy param data from pb to pb2
        for src, dst in zip(pb, pb2):
            dst.data.copy_(src.data)
        ob2 = CAME_C(pb2, lr=1e-4, betas=(0.9, 0.999, 0.9999))
        ob2.load_state_dict(sd)

        # Step both single and loaded-batched
        self._assign_grads(ps, seed=801)
        self._assign_grads(pb2, seed=801)
        os_.step()
        ob2.step()

        # load_state_dict casts fp32 state → bf16 (param dtype), losing precision.
        # Use bf16-level tolerance for this comparison.
        self._compare(ps, pb2, os_, ob2, atol=1e-3, rtol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batched_single_param_fallback(self):
        """Singleton shape group falls back to single-param path."""
        torch.manual_seed(42)
        p_s = torch.randn(
            32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        p_b = p_s.detach().clone().requires_grad_(True)

        # Single group with 1 param → singleton → single-param path
        opt_s = CAME_C([p_s], lr=1e-4)
        # Group with 1 param → also singleton
        opt_b = CAME_C([p_b], lr=1e-4)

        p_s.grad = torch.randn_like(p_s)
        p_b.grad = p_s.grad.clone()

        opt_s.step()
        opt_b.step()

        torch.testing.assert_close(
            p_s.data.to(torch.float32),
            p_b.data.to(torch.float32),
            atol=1e-6,
            rtol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
