"""
LoKR CUDA Extension — Fused CUDA kernels for LoKR forward/backward.

This module provides CUDA-accelerated versions of KronLinearFn and
KronLinearTwoStageFn, reducing kernel launches from ~8-22 per layer
to 1-3 by fusing the GEMM chain, dtype casts, and scalar multiplication.

Usage:
    from networks.lora_modules.lokr_cpp_extension import (
        KronLinearFnCUDA as KronLinearFn,
        KronLinearTwoStageFnCUDA as KronLinearTwoStageFn,
    )

If the CUDA extension fails to load, these classes alias to the pure-PyTorch
fallback in networks.lora_modules.lokr, and _LOKR_KERNEL_AVAILABLE = False.
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# JIT extension loading (mirrors CAME_C pattern)
# ============================================================================

def _load_extension():
    from torch.utils.cpp_extension import _get_build_directory, load
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    # Stale lock cleanup: if a previous process crashed during JIT compile,
    # torch's FileBaton leaves a "lock" file in the build directory. The next
    # load() call then blocks forever in baton.wait() → time.sleep() — which
    # hangs the daemon worker and every downstream import (this module is
    # pulled in transitively via networks → weights → strategy → TE caching).
    # The .pyd already exists from a prior successful build, so removing the
    # stale lock lets load() find and reuse it without recompiling.
    try:
        build_dir = _get_build_directory("lokr_cpp", verbose=False)
        lock_file = os.path.join(build_dir, "lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except OSError:
        pass
    return load(
        name='lokr_cpp',
        sources=[
            os.path.join(ext_dir, 'lokr_cuda_kernel.cu'),
            os.path.join(ext_dir, 'lokr_op.cpp'),
        ],
        extra_cflags=['/O2', '/std:c++17', '/DNDEBUG', '/wd4819'] if os.name == 'nt'
                       else ['-O3', '-std=c++17', '-DNDEBUG', '-fPIC'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-Xptxas=--disable-warnings',
            '--expt-relaxed-constexpr',
            '-lineinfo',
        ],
        verbose=False,
    ), None


try:
    lokr_cpp, _IMPORT_ERROR = _load_extension()
    _LOKR_KERNEL_AVAILABLE = True
    logger.info("LoKR CUDA extension loaded successfully")
except Exception as e:
    lokr_cpp = None
    _IMPORT_ERROR = e
    _LOKR_KERNEL_AVAILABLE = False
    logger.warning(f"LoKR CUDA extension not available, using PyTorch fallback: {e}")


# ============================================================================
# Compile-compatible CUDA kernels via torch.library.custom_op
# ============================================================================
#
# The previous torch.autograd.Function approach forced lokr.py to check
# torch.compiler.is_compiling() and fall back to slow PyTorch einsum inside
# compiled graphs — because Dynamo cannot trace through the opaque C++ calls
# in a Function's forward/backward.
#
# custom_op + register_fake + register_autograd solves this: Dynamo sees the
# op as a single opaque node (shape-only via register_fake), and the CUDA
# kernel runs in both eager and compiled contexts — including inside
# torch.utils.checkpoint recomputation during gradient checkpointing.

if _LOKR_KERNEL_AVAILABLE:

    # ---- kron1: single-stage (w1 full, w2 full or factored) ---------------
    #
    # scalar is deliberately excluded from the custom op: the CUDA kernel
    # receives 1.0 and the scalar multiply is done outside the op
    # (`kron1_fwd(...) * scalar`) so autograd handles the scalar gradient.
    # This avoids float(tensor) during AOTAutograd tracing (which triggers
    # GuardOnDataDependentSymNode).

    @torch.library.custom_op("lokr::kron1_fwd", mutates_args=())
    def kron1_fwd(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return lokr_cpp.kron1_forward(x, w1, w2, 1.0)

    @kron1_fwd.register_fake
    def _kron1_fwd_fake(x, w1, w2):
        return torch.empty(
            (*x.shape[:-1], w1.shape[0] * w2.shape[0]),
            dtype=x.dtype, device=x.device,
        )

    def _kron1_bwd(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x, grad_w1, grad_w2, _ = lokr_cpp.kron1_backward(
            grad_out, x, w1, w2, 1.0
        )
        return grad_x, grad_w1, grad_w2

    def _kron1_setup(ctx, inputs, output):
        x, w1, w2 = inputs
        ctx.save_for_backward(x, w1, w2)

    torch.library.register_autograd(
        "lokr::kron1_fwd", _kron1_bwd, setup_context=_kron1_setup,
    )

    # ---- kron2: two-stage (decompose_both, w1 and w2 factored) ------------
    #
    # kron2's C++ backward uses raw data_ptr access (CUDA kernel launches),
    # which make_fx cannot trace. We register it as its own custom_op so
    # AOTAutograd treats it as an opaque node in the backward graph.

    @torch.library.custom_op("lokr::kron2_fwd", mutates_args=())
    def kron2_fwd(
        x: torch.Tensor, w1_a: torch.Tensor, w1_b: torch.Tensor,
        w2_a: torch.Tensor, w2_b: torch.Tensor,
    ) -> torch.Tensor:
        x = x.contiguous()
        out, _temp1 = lokr_cpp.kron2_forward(x, w1_a, w1_b, w2_a, w2_b, 1.0)
        return out

    @kron2_fwd.register_fake
    def _kron2_fwd_fake(x, w1_a, w1_b, w2_a, w2_b):
        return torch.empty(
            (*x.shape[:-1], w1_a.shape[0] * w2_a.shape[0]),
            dtype=x.dtype, device=x.device,
        )

    @torch.library.custom_op("lokr::kron2_bwd_op", mutates_args=())
    def kron2_bwd_op(
        grad_out: torch.Tensor, temp1: torch.Tensor,
        x: torch.Tensor, w1_a: torch.Tensor, w1_b: torch.Tensor,
        w2_a: torch.Tensor, w2_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_x, grad_w1a, grad_w1b, grad_w2a, grad_w2b, _ = (
            lokr_cpp.kron2_backward(
                grad_out, temp1.contiguous(), x, w1_a, w1_b, w2_a, w2_b, 1.0
            )
        )
        return grad_x, grad_w1a, grad_w1b, grad_w2a, grad_w2b

    @kron2_bwd_op.register_fake
    def _kron2_bwd_op_fake(grad_out, temp1, x, w1_a, w1_b, w2_a, w2_b):
        return (
            torch.empty(x.shape, dtype=x.dtype, device=x.device),
            torch.empty(w1_a.shape, dtype=w1_a.dtype, device=w1_a.device),
            torch.empty(w1_b.shape, dtype=w1_b.dtype, device=w1_b.device),
            torch.empty(w2_a.shape, dtype=w2_a.dtype, device=w2_a.device),
            torch.empty(w2_b.shape, dtype=w2_b.dtype, device=w2_b.device),
        )

    def _kron2_bwd(ctx, grad_out):
        x, w1_a, w1_b, w2_a, w2_b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        # Recompute temp1 (cheap: one small matmul on the backward path).
        # These are standard ATen ops that make_fx can trace.
        in_m = w1_b.shape[1]
        in_n = w2_b.shape[1]
        x_f = x.float().reshape(-1, in_m * in_n)
        X = x_f.reshape(x_f.shape[0], in_m, in_n)
        temp1 = X @ w2_b.float().t()
        # kron2_bwd_op is an opaque custom_op — make_fx won't trace into it.
        return kron2_bwd_op(grad_out, temp1, x, w1_a, w1_b, w2_a, w2_b)

    def _kron2_setup(ctx, inputs, output):
        x, w1_a, w1_b, w2_a, w2_b = inputs
        ctx.save_for_backward(x, w1_a, w1_b, w2_a, w2_b)

    torch.library.register_autograd(
        "lokr::kron2_fwd", _kron2_bwd, setup_context=_kron2_setup,
    )

    # ---- Backward-compatible wrapper classes (.apply API) -----------------
    # scalar multiply is outside the custom op so autograd handles its grad.

    class KronLinearFnCUDA:
        """Single-stage Kronecker via compile-compatible custom op."""

        @staticmethod
        def apply(x, w1, w2, scalar):
            return kron1_fwd(x, w1, w2) * scalar

    class KronLinearTwoStageFnCUDA:
        """Two-stage Kronecker via compile-compatible custom op."""

        @staticmethod
        def apply(x, w1_a, w1_b, w2_a, w2_b, scalar):
            return kron2_fwd(x, w1_a, w1_b, w2_a, w2_b) * scalar

else:
    # Fallback to pure-PyTorch implementations
    from networks.lora_modules.lokr import (
        KronLinearFn as KronLinearFnCUDA,
        KronLinearTwoStageFn as KronLinearTwoStageFnCUDA,
    )
