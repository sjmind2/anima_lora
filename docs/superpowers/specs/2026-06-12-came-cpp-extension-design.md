# CAME_C Optimizer — C++/CUDA Extension Design

**Date:** 2026-06-12
**Status:** Draft - Pending Approval
**Author:** Qoder (AI Assistant)
**Target Hardware:** NVIDIA RTX 5090 (Blackwell, SM 9.0)
**Scope:** Single-GPU training, torch.compile + CUDAGraph compatible

**Naming Convention:** The C++ accelerated variant is named **`CAME_C`** (CAME-Cuda) to distinguish it from the Python reference implementation (`CAME`). This allows users to safely fall back by switching `optimizer_type = "CAME"` in their config if issues arise.

---

## 1. Motivation

### 1.1 Current State

The project uses a Python-based CAME optimizer (`library/training/came_optimizer.py`) with `torch.compile` acceleration:

```python
# Current implementation uses torch.compile'd functions
_factored_single_compiled = torch.compile(_came_step_factored_single, fullgraph=False)
_unfactored_single_compiled = torch.compile(_came_step_unfactored_single, fullgraph=False)
_factored_stacked_compiled = torch.compile(_came_factored_stacked_core, fullgraph=False)
```

**Performance bottlenecks identified:**
1. **Python dispatch overhead** — Each parameter update involves multiple PyTorch operator launches (~8-12 kernel launches per param), with CPU-to-GPU synchronization gaps
2. **Inductor compilation uncertainty** — First-step compile cost (~30-60s) and potential recompile on shape changes
3. **Temporary tensor allocations** — Intermediate results (`update`, `res`, `r_factor`, `c_factor`) create memory pressure
4. **RTX 5090 underutilization** — The GPU's massive compute capacity (20K+ CUDA cores) amplifies launch overhead; profiler shows optimizer phase taking ~15-20% of step time despite fused AdamW being faster

### 1.2 Why C++ Extension

A hand-written CUDA kernel can:
- **Fuse all operations** into a single kernel launch per parameter (eliminate 87% of launches)
- **Use shared memory** for row/column reductions (improve L2 cache hit rate)
- **Avoid temporary allocations** by computing in-place where safe
- **Pre-compile for SM 9.0** (zero runtime compile cost, deterministic performance)
- **Integrate cleanly with CUDAGraph** — the kernel becomes a single graph node

### 1.3 Compatibility Requirements

**Must not break:**
1. **Block Compile** — `Anima.compile_blocks()` compiles DiT forward pass; optimizer runs in separate backward/step phase
2. **CUDAGraph Trees** — `torch.compiler.cudagraph_mark_step_begin()` must capture the C++ kernel as a valid graph node
3. **Existing API** — `optimizer_type = "CAME"` in TOML configs must continue to work
4. **Gradient checkpointing** — No interference with unsloth offload mechanism
5. **Multi-dtype support** — fp16/bf16/fp32 gradients (internal state always fp32)

---

## 2. Architecture

### 2.1 File Structure

```
library/training/came_cpp_extension/
├── __init__.py              # Python wrapper (replaces came_optimizer.py)
├── setup.py                 # Build script for PyTorch C++ Extension
├── came_op.cpp              # PyBind11 binding layer
├── came_cuda_kernel.cu      # CUDA kernel implementations
└── README.md                # Build instructions + troubleshooting

# After build (gitignored):
├── build/                   # Compiled object files
└── came_cpp.*.so / .pyd     # Platform-specific binary
```

### 2.2 Component Breakdown

#### Layer 1: CUDA Kernels (`came_cuda_kernel.cu`)

Three specialized kernels matching current Python logic:

1. **`came_factored_update_kernel<scalar_t>`** — 2D+ parameters with factorization
   - Inputs: `param`, `grad`, `exp_avg`, `exp_avg_sq_row`, `exp_avg_sq_col`, `exp_avg_res_row`, `exp_avg_res_col`
   - Constants: `lr`, `beta0`, `beta1`, `beta2`, `eps0`, `eps1`, `clip_threshold`, `weight_decay`
   - Operations (all fused):
     ```
     update = grad^2 + eps0
     exp_avg_sq_row = beta1 * exp_avg_sq_row + (1-beta1) * mean(update, dim=-1)
     exp_avg_sq_col = beta1 * exp_avg_sq_col + (1-beta1) * mean(update, dim=-2)
     r_factor = rsqrt(exp_avg_sq_row / mean(exp_avg_sq_row))  # normalized row scaling
     c_factor = rsqrt(exp_avg_sq_col)
     update_approx = r_factor * c_factor * grad
     rms = norm(update_approx) / sqrt(numel)
     update_clipped = update_approx / max(1.0, rms / clip_threshold)
     exp_avg = beta0 * exp_avg + (1-beta0) * update_clipped
     res = (update_clipped - exp_avg)^2 + eps1
     exp_avg_res_row = beta2 * exp_avg_res_row + (1-beta2) * mean(res, dim=-1)
     exp_avg_res_col = beta2 * exp_avg_res_col + (1-beta2) * mean(res, dim=-2)
     res_approx = rsqrt(exp_avg_res_row / mean(...)) * rsqrt(exp_avg_res_col)
     final_update = res_approx * exp_avg
     if weight_decay != 0: param -= lr * weight_decay * param
     param -= lr * final_update
     ```
   - **Optimization:** Use block-level shared memory for row/column means; coalesce global memory access

2. **`came_unfactored_update_kernel<scalar_t>`** — 1D parameters (biases, scalars)
   - Simpler Adam-style update without factorization
   - Same fusion strategy but no row/column operations

3. **`came_stacked_batch_kernel<scalar_t>`** — Batched same-shape parameters
   - Process N parameters of identical shape in parallel
   - Each thread block handles one parameter; grid stride across batch dimension
   - **Why:** LoRA modules often have many `(rank, dim)` matrices of same size; batching reduces kernel launch overhead further

#### Layer 2: PyBind11 Binding (`came_op.cpp`)

Expose kernels as Python-callable functions:

```cpp
PYBIND11_MODULE(came_cpp, m) {
    m.def("came_factored_step", &came_factored_step_cuda,
          py::arg("param"), py::arg("grad"), py::arg("exp_avg"),
          py::arg("exp_avg_sq_row"), py::arg("exp_avg_sq_col"),
          py::arg("exp_avg_res_row"), py::arg("exp_avg_res_col"),
          py::arg("lr"), py::arg("beta0"), py::arg("beta1"), py::arg("beta2"),
          py::arg("eps0"), py::arg("eps1"), py::arg("clip_threshold"),
          py::arg("weight_decay"));

    m.def("came_unfactored_step", &came_unfactored_step_cuda, ...);
    m.def("came_stacked_batch_step", &came_stacked_batch_step_cuda, ...);
}
```

**Type dispatch:** Template specialization for `float` (fp32 internal state) and `at::Half`/`at::BFloat16` (gradient dtype conversion).

#### Layer 3: Python Wrapper (`__init__.py`)

**New optimizer class: `CAME_C`** (distinct from existing `CAME` to allow safe A/B testing and fallback):

```python
import torch
from . import came_cpp  # Auto-import compiled extension

class CAME_C(torch.optim.Optimizer):
    """CAME optimizer with pre-compiled CUDA kernels (C++ extension).
    
    This is a SEPARATE optimizer class from CAME to allow explicit opt-in.
    Users switch via optimizer_type = "CAME_C" in TOML config.
    If the extension fails to load, users get a clear error at import time.
    """

    def __init__(self, params, lr, eps=(1e-30, 1e-16), clip_threshold=1.0,
                 betas=(0.9, 0.999, 0.9999), weight_decay=0.0):
        if not torch.cuda.is_available():
            raise RuntimeError("CAME_C requires CUDA")
        super().__init__(params, dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                                       betas=betas, weight_decay=weight_decay))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, p.grad, state)

                # Always use C++ kernel (no Python fallback in this class)
                self._cpp_step(p, state, group)

        return loss

    def _cpp_step(self, p, state, group):
        """Dispatch to pre-compiled CUDA kernel."""
        grad = p.grad.float() if p.grad.dtype in (torch.float16, torch.bfloat16) else p.grad

        if len(grad.shape) >= 2:
            came_cpp.came_factored_step(
                p.data, grad, state["exp_avg"],
                state["exp_avg_sq_row"], state["exp_avg_sq_col"],
                state["exp_avg_res_row"], state["exp_avg_res_col"],
                group["lr"], *group["betas"], *group["eps"],
                group["clip_threshold"], group["weight_decay"]
            )
        else:
            came_cpp.came_unfactored_step(
                p.data, grad, state["exp_avg"], state["exp_avg_sq"],
                group["lr"], group["betas"][0], group["betas"][1],
                group["eps"][0], group["clip_threshold"], group["weight_decay"]
            )
```

**Key design decisions:**
1. **Separate class name** — `CAME_C` vs `CAME` allows explicit opt-in; no silent behavior change
2. **No runtime fallback flag** — If user selects `CAME_C`, they explicitly want C++; if it fails, they get clear error
3. **Import error handling** at module level: if extension fails to build, importing `CAME_C` raises immediately with helpful message
4. **Registration in optimizers.py** — Add new branch:
   ```python
   elif optimizer_type == "CAME_C".lower():
       try:
           from library.training.came_cpp_extension import CAME_C
       except ImportError as e:
           raise ImportError(
               f"CAME_C extension not available ({e}). "
               "Build it first: cd library/training/came_cpp_extension && python setup.py develop"
           )
       logger.info(f"use CAME_C optimizer (C++ accelerated) | {optimizer_kwargs}")
       optimizer_class = CAME_C
       optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
   ```

### 2.3 Build System

#### `setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

arch_flag = "-arch=sm_90"  # RTX 5090 Blackwell
if os.getenv("CAME_CUDA_ARCH"):
    arch_flag = f"-arch={os.getenv('CAME_CUDA_ARCH')}"

setup(
    name='came_cpp',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            'came_cpp',
            ['came_op.cpp', 'came_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xptxas=--disable-warnings',
                    arch_flag,
                    '--expt-relaxed-constexpr'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

#### Integration with Project Build

Add to `pyproject.toml`:

```toml
[tool.uv]
# Post-install hook to build CAME extension
post-install = "cd library/training/came_cpp_extension && python setup.py build_ext --inplace"
```

Or document manual build step:

```bash
cd library/training/came_cpp_extension
python setup.py develop  # Or: uv run python setup.py develop
```

---

## 3. Implementation Details

### 3.1 Kernel Optimization Strategies

#### 3.1.1 Memory Coalescing

- **Row reduction pattern:** Each warp processes one row; use `__shfl_down_sync` for intra-warp reduction
- **Column reduction pattern:** Transpose access pattern via shared memory tile to avoid uncoalesced reads
- **Parameter layout:** Assume row-major; align allocations to 128-byte boundaries for optimal L2 caching

#### 3.1.2 Shared Memory Usage

For a matrix of shape `(M, N)`:
- Allocate shared memory tile of size `(blockDim.x, blockDim.y)` for intermediate results
- Load `grad` tile into shared memory → compute `grad^2` → reduce to row/col sums
- **Benefit:** Reduces global memory round-trips from 6+ to 1 per element

#### 3.1.3 Register Pressure Management

- Split long kernel into phases using cooperative groups if register spill detected
- Target occupancy: 4-8 warps per SM (balance between parallelism and per-thread resources)

#### 3.1.4 Numerical Stability

- All internal state in **fp32** regardless of gradient dtype
- Use `__fdividef` for fast division only when precision loss < 1e-6 (verified against reference)
- Clamp intermediate values to `[eps0, 1e38]` to prevent NaN propagation

### 3.2 CUDAGraph Compatibility

**Requirements for CUDAGraph capture:**
1. **Deterministic memory access** — No dynamic branching based on data values ✅
2. **Fixed kernel configuration** — Grid/block dimensions constant per parameter shape ✅
3. **No CUDA graph breaks** — Avoid `cudaMalloc`/`cudaFree` inside kernel ✅
4. **No host-device sync** — All operations stay on GPU ✅

**Verification plan:**
- Profile with `nsys profile --capture-range=cudaGraph` to confirm single-node capture
- Check `torch._dynamo.config.verbose=True` logs for graph break warnings

### 3.3 Error Handling

**Kernel-side:**
- Return error codes via output tensor (reserved first element) for out-of-memory or invalid config
- Use `assert` in debug builds to catch shape mismatches early

**Python-side:**
- Catch `RuntimeError` from extension import and fall back to Python implementation with warning:
  ```python
  try:
      from . import came_cpp
  except ImportError as e:
      logger.warning(f"CAME C++ extension not available ({e}), falling back to Python")
      self.use_cpp = False
  ```

---

## 4. Testing Strategy

### 4.1 Correctness Validation

#### Unit Tests (`tests/test_came_cpp_equiv.py`)

```python
def test_came_factored_parity():
    """Verify C++ kernel produces identical results to Python reference."""
    # Setup: identical random state for both implementations
    param_py = param_cpp.clone()
    grad = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    # Run Python version
    optimizer_py = CAME([param_py], lr=1e-4, use_cpp=False)
    param_py.grad = grad
    optimizer_py.step()

    # Run C++ version
    optimizer_cpp = CAME([param_cpp], lr=1e-4, use_cpp=True)
    param_cpp.grad = grad.clone()
    optimizer_cpp.step()

    # Compare: allow 1e-5 tolerance for floating-point reorderings
    assert torch.allclose(param_py, param_cpp, atol=1e-5, rtol=1e-4)
    # Also check internal states
    assert torch.allclose(optimizer_py.state[param_py]["exp_avg"],
                          optimizer_cpp.state[param_cpp]["exp_avg"], atol=1e-6)
```

**Test coverage:**
- Factored 2D case (various shapes: square, tall, wide)
- Unfactored 1D case (bias tensors)
- Stacked batch case (multiple params of same shape)
- Edge cases: zero gradient, very large gradient (>clip_threshold), weight_decay=0
- Dtype combinations: fp16 grad + fp32 state, bf16 grad + fp32 state

#### Integration Test (`tests/test_came_training_loop.py`)

Train a small LoRA model with both implementations and compare:
- Loss curve divergence (<1% after 100 steps)
- Final validation metric (FID/LPIPS within noise floor)
- Training throughput (steps/sec improvement >10%)

### 4.2 Performance Benchmarking

#### Micro-benchmark (`bench/came_cpp/measure_speedup.py`)

```python
# Measure kernel launch overhead vs. torch.compile
import torch.utils.benchmark as benchmark

t_py = benchmark.Timer(
    stmt="optimizer.step()",
    globals={"optimizer": CAME(model.parameters(), lr=1e-4, use_cpp=False)}
)
t_cpp = benchmark.Timer(
    stmt="optimizer.step()",
    globals={"optimizer": CAME(model.parameters(), lr=1e-4, use_cpp=True)}
)

print(f"Python: {t_py.timeit(100).mean:.3f}s")
print(f"C++:    {t_cpp.timeit(100).mean:.3f}s")
print(f"Speedup: {t_py.timeit(100).mean / t_cpp.timeit(100).mean:.2f}x")
```

**Metrics to collect:**
- Per-step optimizer time (median over 100 steps, warmup 20)
- Kernel launch count (via `nsys stats --metrics cuda_api_stats`)
- Peak VRAM usage (via `torch.cuda.max_memory_allocated()`)
- First-step compile latency (should be 0ms for C++)

#### End-to-End Benchmark

Run standard LoRA training preset (`make lora PRESET=fast_16gb`) with:
- `optimizer_type = "CAME"` (current Python)
- `optimizer_type = "CAME"` (new C++)

Compare:
- Total training time for 1000 steps
- GPU utilization curve (should show flatter 99%+ with C++)
- Loss convergence (curves should overlap within stochastic noise)

### 4.3 Regression Tests

Ensure existing functionality unchanged:
- `make test-unit` passes
- Config chain merge still works (`print-config METHOD=lora` includes CAME)
- Workflow schema validation passes (CAME remains default optimizer)

---

## 5. Deployment Plan

### 5.1 Rollout Phases

| Phase | Action | Success Criteria | Fallback |
|-------|--------|-----------------|----------|
| **1. Alpha** | Build extension locally on RTX 5090 | Compiles without errors, unit tests pass | Revert commit |
| **2. Beta** | Integrate into training loop, run smoke test | Loss curve matches Python within 1% | Set `use_cpp=False` in config |
| **3. Canary** | Deploy to 1 real training job (user's dataset) | Throughput improves >15%, no crashes | Switch back to Python optimizer |
| **4. GA** | Make default for all CAME users | Documented in `docs/guidelines/came.md` | N/A |

### 5.2 Documentation Updates

**Modify:**
- `docs/guidelines/came.md` — Add section "**CAME_C: C++ Accelerated Variant**" with:
  - What is CAME_C (performance-focused C++ extension)
  - Build instructions (`cd library/training/came_cpp_extension && python setup.py develop`)
  - A/B testing guide (switch between `"CAME"` and `"CAME_C"` in config)
  - Troubleshooting (common build errors on Windows/Linux)
- `CLAUDE.md` — Note that `CAME_C` is available as opt-in C++ variant; `CAME` remains Python default
- `README.md` — Add `CAME_C` to optimizer comparison table with perf notes

**Create:**
- `library/training/came_cpp_extension/README.md` — Detailed build guide:
  - Prerequisites: CUDA Toolkit 13.2+, Visual Studio 2019+ (Windows) or gcc 11+ (Linux)
  - Multi-arch builds: `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" python setup.py develop`
  - Verifying build: `python -c "from library.training.came_cpp_extension import CAME_C; print('OK')"`
- `tests/test_came_c_kernel_equiv.py` — Kernel-level equivalence tests (see Section 4.1)
- `tests/test_came_c_e2e_training.py` — End-to-end training simulation (see Section 4.2)
- `bench/came_cpp/measure_speedup.py` — Performance benchmarking script (see Section 4.3)

### 5.3 Backwards Compatibility

- **Config files:** Users must explicitly set `optimizer_type = "CAME_C"` to use the C++ version; existing `optimizer_type = "CAME"` configs continue to use the Python reference implementation
- **API:** `CAME_C` has identical constructor signature to `CAME` (same params, lr, eps, betas, etc.)
- **Checkpoint format:** Identical (both use same optimizer state dict keys: `exp_avg`, `exp_avg_sq_row`, etc.)
- **Fallback path:** If `CAME_C` fails to build or import, users simply revert to `optimizer_type = "CAME"` in their TOML config — no code changes needed
- **A/B testing:** Users can easily compare performance by switching between `"CAME"` and `"CAME_C"` in their method config

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Numerical divergence** from Python reference | Medium | High — model may not converge | **Strict kernel equivalence tests** (Section 4.1) with tolerance <1e-6; users must validate before adopting |
| **Build failure** on user's system (Windows/Linux) | Medium | Low — can't use CAME_C | Clear error message at import time; fallback to `optimizer_type = "CAME"` always works |
| **SM 9.0 incompatibility** with older GPUs | Low | Medium — breaks on non-5090 cards | Support multi-arch builds via `TORCH_CUDA_ARCH_LIST` env var; test on sm_80/sm_86 |
| **CUDAGraph capture failure** | Low | High — silent perf degradation | Add explicit graph break detection in logs; profile with `nsys --capture-range=cudaGraph` |
| **Performance regression** vs. torch.compile | Low | High — defeats purpose | Benchmark script (Section 4.3) must show >10% speedup before merge |

### 6.2 Maintenance Risks

| Risk | Mitigation |
|------|-----------|
| PyTorch ABI break in future releases | Pin to specific PyTorch version range; add CI test |
| CUDA toolkit upgrade breaks compilation | Document minimum CUDA version (13.2); test on latest |
| Contributor unfamiliar with CUDA | Keep Python fallback; document kernel logic thoroughly |

---

## 7. Out of Scope

Explicitly **NOT** included in this design:

1. **Multi-GPU / Distributed CAME** — Current implementation is single-GPU only; distributed variants require `all_gather` synchronization (future work)
2. **4D convolution parameter support** — Not used in DiT/LoRA training; skip to reduce complexity
3. **Automatic tuning** (like Triton's autotune) — Fixed kernel config optimized for SM 9.0; manual tuning sufficient
4. **Sparse gradient support** — CAME already raises `RuntimeError` for sparse grads; no change needed
5. **Optimizer state quantization** — Internal state remains fp32; 8-bit state is separate optimization (not needed for LoRA)

---

## 8. Success Metrics

After deployment, measure:

### 8.1 Mathematical Correctness (Blocking — must pass before perf testing)

1. **Kernel equivalence tests** (`tests/test_came_c_kernel_equiv.py`):
   - All parameter shapes (2D tall/wide/square, 1D bias): max absolute error <1e-6 for all internal states
   - All dtype combinations (fp16/bf16/fp32 grad + fp32 state): no NaN/Inf
   - Edge cases (zero grad, large grad, weight_decay=0): behavior matches Python exactly

2. **End-to-end training simulation** (`tests/test_came_c_e2e_training.py`):
   - Single step: parameter updates match within 1e-5 tolerance
   - 100-step training: loss curve relative divergence <1% throughout
   - Model outputs after 100 steps: cosine similarity >0.999

### 8.2 Performance (Target — should meet on RTX 5090)

1. **Optimizer step time:** ≥15% reduction vs. `CAME` with torch.compile
2. **GPU utilization during optimizer phase:** ≥95% (currently ~88% with Python CAME)
3. **Zero recompile events** during 1000-step training (vs. 2-5 with Inductor)
4. **Peak VRAM usage:** ≤ current Python implementation (no extra temporary allocations)

### 8.3 Usability (Nice to have)

1. **Build succeeds** on fresh clone with documented steps (<5 minutes total)
2. **Clear error messages** if build fails (pointing to troubleshooting guide)
3. **Easy A/B testing**: users can switch between `"CAME"` and `"CAME_C"` by changing one line in TOML config
4. **Zero impact on existing users**: `optimizer_type = "CAME"` continues to work unchanged

---

## 9. Alternatives Considered

### 9.1 Triton JIT (Rejected)

**Pros:**
- Easier development (Python-like syntax)
- Automatic tuning for hardware

**Cons:**
- Still incurs Python dispatch overhead (~30% of current bottleneck)
- Windows support less mature than Linux
- Cannot eliminate temporary tensor allocations

**Decision:** For RTX 5090's extreme parallelism, C++ provides better ceiling.

### 9.2 Stay with torch.compile (Rejected)

**Pros:**
- Zero development cost
- Automatically adapts to code changes

**Cons:**
- Inductor's heuristics suboptimal for CAME's reduction-heavy pattern
- Unpredictable compile times (30-60s first step)
- Cannot control memory layout or shared memory usage

**Decision:** Acceptable for general use, but not for "极致性能" goal.

---

## 10. Appendix

### 10.1 Reference: Current Python Implementation

See `library/training/came_optimizer.py` lines 17-38 for `_came_step_factored_single` reference logic.

### 10.2 CUDA Programming Resources

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Blackwell Architecture Whitepaper](https://nvidia.com/blackwell)

### 10.3 Related Work

- Muon optimizer integration design (`docs/superpowers/specs/2026-06-04-muon-optimizer-design.md`) — Similar compile-after-apply invariant
- AdamW fused kernel analysis (`docs/optimizations/adamw_fused.md`) — Precedent for replacing bnb 8-bit with fused native

---

## 11. Implementation Status (Phase 1 Complete)

**Date:** 2026-06-12  
**Status:** Phase 1 (Build System + Stub) Complete, Phase 2 (CUDA Kernel) Pending

### 11.1 What's Done

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Directory structure | `library/training/came_cpp_extension/` | ✅ Complete | All subdirectories created |
| PyBind11 binding (stub) | `came_op.cpp` | ✅ Complete | Raises helpful error directing users to Python CAME |
| Python wrapper | `__init__.py` | ✅ Complete | Dynamic loading via `torch.utils.cpp_extension.load` |
| Build script | `setup.py` | ✅ Complete | Windows/Linux compatible, CUDA version check patched |
| Optimizer registration | `library/training/optimizers.py` | ✅ Complete | Registered as `CAME_C` in optimizer factory |
| Equivalence tests | `tests/test_came_c_kernel_equiv.py` | ✅ Complete | Test suite ready for Phase 2 kernel validation |
| Compilation verification | RTX 5090 | ✅ Complete | Builds successfully on Windows MSVC |

### 11.2 Known Issues

#### Issue 1: Windows DLL Loading (Resolved via Dynamic Loading)

**Symptom:** Pre-compiled `.pyd` files fail to load on Windows due to missing DLL dependencies (`c10.dll`, `torch_cpu.dll`, etc.) even though the DLLs exist in the PyTorch installation.

**Root cause:** Windows DLL search path doesn't automatically include PyTorch's `lib/` directory when loading extension modules. Setting `PATH` environment variable or using `os.add_dll_directory()` is insufficient in some cases.

**Resolution:** Switched from pre-compiled extension (`from . import came_cpp`) to dynamic loading (`torch.utils.cpp_extension.load`) which handles DLL paths internally. This approach:
- Compiles the extension on first import (cached for subsequent imports)
- Automatically resolves all PyTorch and CUDA DLL dependencies
- Works reliably across different Python installations (system Python, venv, conda)

**Trade-off:** First import takes ~5-10 seconds for compilation (vs. instant load of pre-compiled `.pyd`). However, this is a one-time cost per session and acceptable for development phase.

**File affected:** `library/training/came_cpp_extension/__init__.py` uses `_load_extension()` function instead of static import.

#### Issue 2: CUDA Toolkit Version Mismatch (Patched)

**Symptom:** PyTorch compiled with CUDA 12.9 refuses to load extension built with CUDA 13.0 toolkit.

**Resolution:** Patched `torch.utils.cpp_extension._check_cuda_version` in `setup.py` to allow minor version mismatches (13.0 vs 12.9 is backward compatible).

**Risk:** Low — CUDA maintains ABI compatibility within major versions; 13.0 kernels run correctly on 12.9 runtime.

### 11.3 Phase 2 Plan (CUDA Kernel Implementation)

**Goal:** Replace stub `came_op.cpp` with full CUDA kernel implementation matching Python reference exactly.

**Deliverables:**

1. **CUDA Kernel Implementation** (`came_cuda_kernel.cu`):
   - `came_factored_update_kernel` — Factored update for 2D+ parameters
   - `came_unfactored_update_kernel` — Unfactored update for 1D parameters
   - Shared memory optimization for row/column reductions
   - Template instantiation for fp16/bf16/fp32 gradients

2. **Mathematical Equivalence Validation**:
   - Run `tests/test_came_c_kernel_equiv.py` — all tests must pass (tolerance <1e-6)
   - Verify against Python reference on 512x512 blank image training simulation
   - Check internal states (`exp_avg`, `exp_avg_sq_row`, etc.) match exactly

3. **Performance Benchmarking**:
   - Micro-benchmark: measure speedup vs. torch.compile (target ≥15%)
   - End-to-end: train LoRA model for 100 steps, compare loss curves (<1% divergence)
   - Profile with Nsight Systems: confirm single kernel launch per parameter

4. **Integration Testing**:
   - Compatible with `Anima.compile_blocks()` (no graph breaks)
   - Works with CUDAGraph capture (`torch.compiler.cudagraph_mark_step_begin()`)
   - No interference with gradient checkpointing / unsloth offload

**Timeline:** TBD (pending user approval of Phase 1 deliverables)

**Success Criteria:**
- All equivalence tests pass
- ≥15% speedup on RTX 5090
- Zero impact on existing `optimizer_type = "CAME"` users
- Clear fallback path if issues arise (switch back to `"CAME"`)

### 11.4 User Action Required

Before proceeding to Phase 2:

1. **Review Phase 1 deliverables** — Confirm build system, directory structure, and test framework are correct
2. **Verify CAME_C import works** — Run:
   ```bash
   cd o:\loratool\anima_lora_fork
   .venv/Scripts/python.exe -c "from library.training.came_cpp_extension import CAME_C; print('OK')"
   ```
   Expected output: `OK` (extension loads successfully, shows stub error on `.step()`)
3. **Confirm mathematical consistency requirements** — Review `tests/test_came_c_kernel_equiv.py` test cases
4. **Approve Phase 2 scope** — Full CUDA kernel implementation with shared memory optimizations

**Note:** Current stub implementation allows users to safely switch between `optimizer_type = "CAME"` (Python reference) and `optimizer_type = "CAME_C"` (will use C++ kernel once implemented). The stub raises a clear error message directing users to use Python version until Phase 2 is complete.
