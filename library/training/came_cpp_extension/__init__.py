"""
CAME_C Optimizer — C++/CUDA accelerated variant of CAME.

This module provides the CAME_C optimizer class, which uses pre-compiled
CUDA kernels for maximum performance on RTX 5090 and similar GPUs.

Usage:
    from library.training.came_cpp_extension import CAME_C

    optimizer = CAME_C(model.parameters(), lr=1e-4)
    # ... training loop ...
    optimizer.step()

Note: This is a SEPARATE optimizer from CAME (Python reference). Users must
explicitly set optimizer_type = "CAME_C" in their config to use it.
"""

import os
import sys
import torch
from typing import Optional

# Load the C++ extension via JIT compilation (handles DLL paths on Windows).
# Returns (module, None) on success or (None, error) on failure.
def _load_extension():
    from torch.utils.cpp_extension import load
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name='came_cpp',
        sources=[
            os.path.join(ext_dir, 'came_cuda_kernel.cu'),
            os.path.join(ext_dir, 'came_op.cpp'),
        ],
        extra_cflags=['/O2', '/std:c++17', '/DNDEBUG', '/wd4819'] if os.name == 'nt' else ['-O3', '-std=c++17', '-DNDEBUG'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-Xptxas=--disable-warnings',
            '--expt-relaxed-constexpr',
        ],
        verbose=False,
    ), None

try:
    came_cpp, _IMPORT_ERROR = _load_extension()
    _EXTENSION_AVAILABLE = True
except Exception as e:
    came_cpp = None
    _IMPORT_ERROR = e
    _EXTENSION_AVAILABLE = False


class CAME_C(torch.optim.Optimizer):
    """
    CAME optimizer with pre-compiled CUDA kernels (C++ extension).

    This is a SEPARATE optimizer class from CAME to allow explicit opt-in.
    Users switch via optimizer_type = "CAME_C" in TOML config.

    CAME (Confidence-guided Adaptive Matrix Evaluation) is a factorized
    optimizer that replaces full-matrix second moments with row/column
    moments, reducing memory while maintaining convergence quality.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (recommended: 0.5-0.9x of AdamW lr)
        eps: Epsilon values for numerical stability (default: (1e-30, 1e-16))
        clip_threshold: RMS clipping threshold (default: 1.0)
        betas: Coefficients for momentum, second moment, residual moment
               (default: (0.9, 0.999, 0.9999))
        weight_decay: Weight decay coefficient (default: 0.0)

    Example:
        >>> from library.training.came_cpp_extension import CAME_C
        >>> optimizer = CAME_C(model.parameters(), lr=1e-4)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float,
        eps=(1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
    ):
        if not _EXTENSION_AVAILABLE:
            raise RuntimeError(
                f"CAME_C extension not available: {_IMPORT_ERROR}\n"
                "Build it first:\n"
                "  cd library/training/came_cpp_extension\n"
                "  python setup.py develop"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CAME_C requires CUDA")

        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def _init_state(self, p: torch.Tensor, grad: torch.Tensor, state: dict):
        """Initialize optimizer state for a parameter."""
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad, dtype=torch.float32)

        if len(grad.shape) >= 2:
            # Factored state for 2D+ parameters
            state["exp_avg_sq_row"] = torch.zeros(
                grad.shape[:-1], dtype=torch.float32, device=grad.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                grad.shape[:-2] + grad.shape[-1:],
                dtype=torch.float32,
                device=grad.device,
            )
            state["exp_avg_res_row"] = torch.zeros(
                grad.shape[:-1], dtype=torch.float32, device=grad.device
            )
            state["exp_avg_res_col"] = torch.zeros(
                grad.shape[:-2] + grad.shape[-1:],
                dtype=torch.float32,
                device=grad.device,
            )
        else:
            # Unfactored state for 1D parameters
            state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)

    def _cpp_step(self, p: torch.Tensor, state: dict, group: dict):
        """Dispatch to pre-compiled CUDA kernel."""
        grad = p.grad
        if grad is None:
            return

        # Convert gradient to fp32 if needed (internal state is always fp32)
        if grad.dtype in (torch.float16, torch.bfloat16):
            grad_fp32 = grad.float()
        else:
            grad_fp32 = grad

        lr = group["lr"]
        beta0, beta1, beta2 = group["betas"]
        eps0, eps1 = group["eps"]
        clip_threshold = group["clip_threshold"]
        weight_decay = group["weight_decay"]

        if len(grad.shape) >= 2:
            # Factored update for 2D+ parameters
            came_cpp.came_factored_step(
                p.data,
                grad_fp32,
                state["exp_avg"],
                state["exp_avg_sq_row"],
                state["exp_avg_sq_col"],
                state["exp_avg_res_row"],
                state["exp_avg_res_col"],
                lr,
                beta0,
                beta1,
                beta2,
                eps0,
                eps1,
                clip_threshold,
                weight_decay,
            )
        else:
            # Unfactored update for 1D parameters
            came_cpp.came_unfactored_step(
                p.data,
                grad_fp32,
                state["exp_avg"],
                state["exp_avg_sq"],
                lr,
                beta0,
                beta1,
                eps0,
                clip_threshold,
                weight_decay,
            )

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
                     (optional, for algorithms like LBFGS)

        Returns:
            The loss value if closure was provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, p.grad, state)

                state["step"] += 1

                # Always use C++ kernel (no Python fallback in this class)
                self._cpp_step(p, state, group)

        return loss

    def __repr__(self):
        return f"CAME_C(lr={self.defaults['lr']}, betas={self.defaults['betas']})"
