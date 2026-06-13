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
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, List

# Load the C++ extension via JIT compilation (handles DLL paths on Windows).
# Returns (module, None) on success or (None, error) on failure.
def _load_extension():
    from torch.utils.cpp_extension import _get_build_directory, load
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    # Stale lock cleanup — see lokr_cpp_extension for full rationale.
    # A crashed prior process leaves a "lock" file that makes load() block
    # forever in FileBaton.wait() → time.sleep().
    try:
        build_dir = _get_build_directory("came_cpp", verbose=False)
        lock_file = os.path.join(build_dir, "lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except OSError:
        pass
    return load(
        name='came_cpp',
        sources=[
            os.path.join(ext_dir, 'came_cuda_kernel.cu'),
            os.path.join(ext_dir, 'came_cuda_batched.cu'),
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


# ============================================================================
# Stacked group containers — hold persistent stacked tensors for batched kernels
# ============================================================================

@dataclass
class StackedFactoredGroup:
    """Persistent stacked tensors for B same-shape (R, C) factored params."""
    B: int
    R: int
    C: int
    param_stack: torch.Tensor     # (B, R, C) — param dtype, filled each step
    grad_stack: torch.Tensor      # (B, R, C) — fp32, filled each step
    exp_avg_stack: torch.Tensor   # (B, R, C) — fp32, persistent state
    sq_row_stack: torch.Tensor    # (B, R) — fp32, persistent state
    sq_col_stack: torch.Tensor    # (B, C) — fp32, persistent state
    res_row_stack: torch.Tensor   # (B, R) — fp32, persistent state
    res_col_stack: torch.Tensor   # (B, C) — fp32, persistent state
    scalar_buf: torch.Tensor      # (B, 3) — fp32, scratch
    approx_buf: torch.Tensor      # (B, R, C) — fp32, scratch
    param_fp32: torch.Tensor      # (B, R, C) — fp32, scratch


@dataclass
class StackedUnfactoredGroup:
    """Persistent stacked tensors for B same-shape (N,) unfactored params."""
    B: int
    N: int
    param_stack: torch.Tensor       # (B, N) — param dtype, filled each step
    grad_stack: torch.Tensor        # (B, N) — fp32, filled each step
    exp_avg_stack: torch.Tensor     # (B, N) — fp32, persistent state
    exp_avg_sq_stack: torch.Tensor  # (B, N) — fp32, persistent state
    scalar_buf: torch.Tensor        # (B,) — fp32, scratch
    buf: torch.Tensor               # (B, N) — fp32, scratch
    param_fp32: torch.Tensor        # (B, N) — fp32, scratch


class CAME_C(torch.optim.Optimizer):
    """
    CAME optimizer with pre-compiled CUDA kernels (C++ extension).

    This is a SEPARATE optimizer class from CAME to allow explicit opt-in.
    Users switch via optimizer_type = "CAME_C" in TOML config.

    CAME (Confidence-guided Adaptive Matrix Evaluation) is a factorized
    optimizer that replaces full-matrix second moments with row/column
    moments, reducing memory while maintaining convergence quality.

    Uses shape-grouped stacked batching: parameters with the same shape are
    grouped and processed in a single set of batched CUDA kernel launches,
    reducing kernel launch overhead from ~3,768/step to ~108/step.

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

        # In-place stack storage: {shape_key: group}
        self._stacked_groups: dict = {}        # {(R,C): StackedFactoredGroup}
        self._stacked_unfactored: dict = {}    # {(N,): StackedUnfactoredGroup}

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

    # ========================================================================
    # Single-param path (existing kernels, unchanged)
    # ========================================================================

    def _cpp_step(self, p: torch.Tensor, state: dict, group: dict):
        """Dispatch to pre-compiled single-param CUDA kernel."""
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

    # ========================================================================
    # Batched path — shape-grouped stacked kernels
    # ========================================================================

    def _ensure_stacked_factored(
        self, params: List[torch.Tensor], R: int, C: int
    ) -> StackedFactoredGroup:
        """Get or create stacked group for shape (R, C).

        Handles three cases:
        1. First creation: allocate stacks, copy initial state, install views
        2. Existing & views intact: direct reuse (zero-copy)
        3. Existing but views broken (e.g. after load_state_dict): copy state
           data into stacks and reinstall views
        """
        key = (R, C)
        B = len(params)
        device = params[0].device
        dtype = params[0].dtype

        if key in self._stacked_groups:
            group = self._stacked_groups[key]
            if group.B == B:
                # Check if state tensors are still views of the stack
                first_state = self.state[params[0]]
                if (
                    "exp_avg" in first_state
                    and first_state["exp_avg"].untyped_storage().data_ptr()
                    == group.exp_avg_stack.untyped_storage().data_ptr()
                ):
                    # Views intact — direct reuse
                    return group
                # Views broken (load_state_dict) — rebuild
                for i, p in enumerate(params):
                    st = self.state[p]
                    group.exp_avg_stack[i].copy_(st["exp_avg"])
                    group.sq_row_stack[i].copy_(st["exp_avg_sq_row"])
                    group.sq_col_stack[i].copy_(st["exp_avg_sq_col"])
                    group.res_row_stack[i].copy_(st["exp_avg_res_row"])
                    group.res_col_stack[i].copy_(st["exp_avg_res_col"])
                    # Reinstall views
                    st["exp_avg"] = group.exp_avg_stack[i]
                    st["exp_avg_sq_row"] = group.sq_row_stack[i]
                    st["exp_avg_sq_col"] = group.sq_col_stack[i]
                    st["exp_avg_res_row"] = group.res_row_stack[i]
                    st["exp_avg_res_col"] = group.res_col_stack[i]
                return group
            # B changed — fall through to create new group

        # Create new group
        group = StackedFactoredGroup(
            B=B, R=R, C=C,
            param_stack=torch.empty(B, R, C, device=device, dtype=dtype),
            grad_stack=torch.empty(B, R, C, dtype=torch.float32, device=device),
            exp_avg_stack=torch.zeros(B, R, C, dtype=torch.float32, device=device),
            sq_row_stack=torch.zeros(B, R, dtype=torch.float32, device=device),
            sq_col_stack=torch.zeros(B, C, dtype=torch.float32, device=device),
            res_row_stack=torch.zeros(B, R, dtype=torch.float32, device=device),
            res_col_stack=torch.zeros(B, C, dtype=torch.float32, device=device),
            scalar_buf=torch.empty(B, 3, dtype=torch.float32, device=device),
            approx_buf=torch.empty(B, R, C, dtype=torch.float32, device=device),
            param_fp32=torch.empty(B, R, C, dtype=torch.float32, device=device),
        )

        # Copy existing state data into stack and install views
        for i, p in enumerate(params):
            st = self.state[p]
            if "exp_avg" in st and isinstance(st["exp_avg"], torch.Tensor):
                group.exp_avg_stack[i].copy_(st["exp_avg"])
                group.sq_row_stack[i].copy_(st["exp_avg_sq_row"])
                group.sq_col_stack[i].copy_(st["exp_avg_sq_col"])
                group.res_row_stack[i].copy_(st["exp_avg_res_row"])
                group.res_col_stack[i].copy_(st["exp_avg_res_col"])
            # Install views (zero-copy link to persistent stack)
            st["exp_avg"] = group.exp_avg_stack[i]
            st["exp_avg_sq_row"] = group.sq_row_stack[i]
            st["exp_avg_sq_col"] = group.sq_col_stack[i]
            st["exp_avg_res_row"] = group.res_row_stack[i]
            st["exp_avg_res_col"] = group.res_col_stack[i]

        self._stacked_groups[key] = group
        return group

    def _ensure_stacked_unfactored(
        self, params: List[torch.Tensor], N: int
    ) -> StackedUnfactoredGroup:
        """Get or create stacked group for shape (N,)."""
        key = (N,)
        B = len(params)
        device = params[0].device
        dtype = params[0].dtype

        if key in self._stacked_unfactored:
            group = self._stacked_unfactored[key]
            if group.B == B:
                first_state = self.state[params[0]]
                if (
                    "exp_avg" in first_state
                    and first_state["exp_avg"].untyped_storage().data_ptr()
                    == group.exp_avg_stack.untyped_storage().data_ptr()
                ):
                    return group
                # Views broken — rebuild
                for i, p in enumerate(params):
                    st = self.state[p]
                    group.exp_avg_stack[i].copy_(st["exp_avg"])
                    group.exp_avg_sq_stack[i].copy_(st["exp_avg_sq"])
                    st["exp_avg"] = group.exp_avg_stack[i]
                    st["exp_avg_sq"] = group.exp_avg_sq_stack[i]
                return group
            # B changed — fall through

        group = StackedUnfactoredGroup(
            B=B, N=N,
            param_stack=torch.empty(B, N, device=device, dtype=dtype),
            grad_stack=torch.empty(B, N, dtype=torch.float32, device=device),
            exp_avg_stack=torch.zeros(B, N, dtype=torch.float32, device=device),
            exp_avg_sq_stack=torch.zeros(B, N, dtype=torch.float32, device=device),
            scalar_buf=torch.empty(B, dtype=torch.float32, device=device),
            buf=torch.empty(B, N, dtype=torch.float32, device=device),
            param_fp32=torch.empty(B, N, dtype=torch.float32, device=device),
        )

        for i, p in enumerate(params):
            st = self.state[p]
            if "exp_avg" in st and isinstance(st["exp_avg"], torch.Tensor):
                group.exp_avg_stack[i].copy_(st["exp_avg"])
                group.exp_avg_sq_stack[i].copy_(st["exp_avg_sq"])
            st["exp_avg"] = group.exp_avg_stack[i]
            st["exp_avg_sq"] = group.exp_avg_sq_stack[i]

        self._stacked_unfactored[key] = group
        return group

    def _cpp_step_batched_factored(
        self, group: StackedFactoredGroup, params: List[torch.Tensor], cfg: dict
    ):
        """Fill stacks → batched kernel → write back params.

        Uses torch._foreach_copy_ for batch GPU-to-GPU copy to minimize
        Python loop overhead.
        """
        B = group.B

        # 1. Fill param data into stack (batch copy)
        param_views = [p.data for p in params]
        torch._foreach_copy_(group.param_stack.unbind(0), param_views)

        # 2. Fill grad data — need fp32 conversion for bf16/fp16 grads
        grads_fp32 = []
        need_float = False
        for p in params:
            g = p.grad
            if g.dtype != torch.float32:
                need_float = True
            grads_fp32.append(g)
        if need_float:
            grads_fp32 = [g.float() if g.dtype != torch.float32 else g
                          for g in grads_fp32]
        torch._foreach_copy_(group.grad_stack.unbind(0), grads_fp32)

        # 3. Call batched kernel (modifies exp_avg_stack, sq_row/col, etc. in-place)
        came_cpp.came_factored_batched_step(
            group.param_stack,
            group.grad_stack,
            group.exp_avg_stack,
            group.sq_row_stack,
            group.sq_col_stack,
            group.res_row_stack,
            group.res_col_stack,
            group.scalar_buf,
            group.approx_buf,
            group.param_fp32,
            cfg["lr"],
            cfg["betas"][0],
            cfg["betas"][1],
            cfg["betas"][2],
            cfg["eps"][0],
            cfg["eps"][1],
            cfg["clip_threshold"],
            cfg["weight_decay"],
        )

        # 4. Write back params (batch copy)
        torch._foreach_copy_(param_views, group.param_stack.unbind(0))

    def _cpp_step_batched_unfactored(
        self, group: StackedUnfactoredGroup, params: List[torch.Tensor], cfg: dict
    ):
        """Fill stacks → batched kernel → write back params."""
        param_views = [p.data for p in params]
        torch._foreach_copy_(group.param_stack.unbind(0), param_views)

        grads_fp32 = []
        need_float = False
        for p in params:
            g = p.grad
            if g.dtype != torch.float32:
                need_float = True
            grads_fp32.append(g)
        if need_float:
            grads_fp32 = [g.float() if g.dtype != torch.float32 else g
                          for g in grads_fp32]
        torch._foreach_copy_(group.grad_stack.unbind(0), grads_fp32)

        came_cpp.came_unfactored_batched_step(
            group.param_stack,
            group.grad_stack,
            group.exp_avg_stack,
            group.exp_avg_sq_stack,
            group.scalar_buf,
            group.buf,
            group.param_fp32,
            cfg["lr"],
            cfg["betas"][0],
            cfg["betas"][1],
            cfg["eps"][0],
            cfg["clip_threshold"],
            cfg["weight_decay"],
        )

        torch._foreach_copy_(param_views, group.param_stack.unbind(0))

    def reset_stacks(self):
        """Discard all stacked groups. Call after param set changes."""
        self._stacked_groups.clear()
        self._stacked_unfactored.clear()

    # ========================================================================
    # Main step — dynamic shape grouping + dispatch
    # ========================================================================

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Groups parameters by shape and dispatches to batched CUDA kernels
        when multiple params share the same shape. Singleton shape groups
        fall back to the single-param kernel path.

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
            # 1. Group params by shape
            factored_groups: dict = defaultdict(list)
            unfactored_groups: dict = defaultdict(list)

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, p.grad, state)

                state["step"] += 1

                grad = p.grad
                if grad.dim() >= 2:
                    # Use first 2 dims for grouping key
                    R, C = grad.shape[0], grad.shape[1]
                    factored_groups[(R, C)].append(p)
                else:
                    N = grad.shape[0]
                    unfactored_groups[(N,)].append(p)

            # 2. Process factored groups
            for (R, C), params in factored_groups.items():
                if len(params) == 1:
                    # Singleton — use single-param path
                    self._cpp_step(params[0], self.state[params[0]], group)
                else:
                    # Batched path
                    stacked = self._ensure_stacked_factored(params, R, C)
                    self._cpp_step_batched_factored(stacked, params, group)

            # 3. Process unfactored groups
            for (N,), params in unfactored_groups.items():
                if len(params) == 1:
                    self._cpp_step(params[0], self.state[params[0]], group)
                else:
                    stacked = self._ensure_stacked_unfactored(params, N)
                    self._cpp_step_batched_unfactored(stacked, params, group)

        return loss

    def __repr__(self):
        return f"CAME_C(lr={self.defaults['lr']}, betas={self.defaults['betas']})"
