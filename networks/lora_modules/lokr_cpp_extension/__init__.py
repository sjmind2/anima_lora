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
# CUDA-accelerated autograd Functions
# ============================================================================

if _LOKR_KERNEL_AVAILABLE:

    class KronLinearFnCUDA(torch.autograd.Function):
        """Fused single-stage Kronecker forward/backward (CUDA kernel).

        Replaces ~8 forward + ~15 backward kernel launches with 1 + 2.
        Mathematically identical to the pure-PyTorch KronLinearFn.
        """

        @staticmethod
        def forward(ctx, x, w1, w2, scalar):
            assert x.is_contiguous(), "x must be contiguous"
            out = lokr_cpp.kron1_forward(x, w1, w2, float(scalar))
            ctx.save_for_backward(x, w1, w2)
            ctx.scalar = float(scalar)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x, w1, w2 = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_x, grad_w1, grad_w2, grad_scalar = lokr_cpp.kron1_backward(
                grad_out, x, w1, w2, ctx.scalar
            )
            return grad_x, grad_w1, grad_w2, grad_scalar

    class KronLinearTwoStageFnCUDA(torch.autograd.Function):
        """Fused two-stage Kronecker forward/backward (CUDA kernel).

        Replaces ~12 forward + ~22 backward kernel launches with 1 + 3.
        For the decompose_both=True path where both w1 and w2 are factored.
        """

        @staticmethod
        def forward(ctx, x, w1_a, w1_b, w2_a, w2_b, scalar):
            assert x.is_contiguous(), "x must be contiguous"
            out, temp1 = lokr_cpp.kron2_forward(
                x, w1_a, w1_b, w2_a, w2_b, float(scalar)
            )
            ctx.save_for_backward(x, w1_a, w1_b, w2_a, w2_b)
            ctx.temp1 = temp1
            ctx.scalar = float(scalar)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x, w1_a, w1_b, w2_a, w2_b = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_x, grad_w1a, grad_w1b, grad_w2a, grad_w2b, grad_scalar = (
                lokr_cpp.kron2_backward(
                    grad_out, ctx.temp1, x, w1_a, w1_b, w2_a, w2_b, ctx.scalar
                )
            )
            ctx.temp1 = None  # free workspace
            return grad_x, grad_w1a, grad_w1b, grad_w2a, grad_w2b, grad_scalar

else:
    # Fallback to pure-PyTorch implementations
    from networks.lora_modules.lokr import (
        KronLinearFn as KronLinearFnCUDA,
        KronLinearTwoStageFn as KronLinearTwoStageFnCUDA,
    )
