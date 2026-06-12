# CAME_C Optimizer — C++/CUDA Extension

**Status:** Production-ready (Fused CUDA Kernels)

## Overview

CAME_C is a C++/CUDA accelerated variant of the CAME optimizer. It replaces all Python-level ATen operations with 8 fused CUDA kernels, eliminating Python interpreter dispatch overhead and reducing per-step kernel launches from 35+ to 3–6 per parameter.

**Performance:** 1.31× faster than the Python CAME reference (measured on RTX 5090 with sustained 30-second benchmark). Numerical equivalence verified: loss trajectory relative error ≤ 6e-08, optimizer state deviation ≤ 1e-11.

## Architecture

8 custom CUDA kernels (0 ATen math operations):

| Kernel | Path | Purpose |
|--------|------|---------|
| **P0** `prepare_param` | Both | Read bf16/fp16/fp32 param → fp32, apply weight decay, clear scalar_buf |
| **S0** `scale2` | Factored | Pre-multiply `sq_col` and `res_col` by decay, clear scalar_buf |
| **F1** `factored_row_sq` | Factored | Row-wise squared gradient EMA + accumulate sq_row_sum via block reduction |
| **F2** `factored_col_sq` | Factored | Column-wise sq_col EMA via atomicAdd + RMS clipping |
| **F3** `factored_row_res` | Factored | Row-wise residual EMA + accumulate res_row_sum via block reduction |
| **F4** `factored_update` | Factored | Compute trust ratio + apply update + write back to original dtype |
| **U1** `unfactored_sq` | Unfactored | Full exp_avg_sq EMA + RMS clipping |
| **U2** `unfactored_update` | Unfactored | Compute trust ratio + apply update + write back to original dtype |

**Kernel launch count per parameter:**
- Factored (2D+): 6 launches (P0 → S0 → F1 → F2 → F3 → F4)
- Unfactored (1D): 3 launches (P0 → U1 → U2)

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit 12.9+ (13.0 tested)
- MSVC 2019+ (Windows) or GCC 11+ (Linux)

### Using CAME_C

In your TOML config:

```toml
optimizer_type = "CAME_C"
learning_rate = 1.5e-5
```

Pass optimizer-specific arguments via `optimizer_args`:

```toml
optimizer_type = "CAME_C"
optimizer_args = ["weight_decay=0.01", "betas=0.9,0.999,0.9999"]
learning_rate = 1.5e-5
```

Or programmatically:

```python
from library.training.came_cpp_extension import CAME_C

optimizer = CAME_C(model.parameters(), lr=1e-4)
optimizer.step()
```

### Fallback to Python Reference

If you encounter issues with CAME_C, simply switch back to the Python reference implementation:

```toml
optimizer_type = "CAME"  # Changed from "CAME_C"
```

No other code changes needed — both optimizers share the same API and checkpoint format.

## Performance

Sustained 30-second benchmark results (RTX 5090, bf16, typical LoRA parameter shapes):

| Metric | CAME (Python) | CAME_C (CUDA) |
|--------|---------------|----------------|
| Time per step | 1.358 ms | 1.033 ms |
| Throughput | 736.2 steps/sec | 967.9 steps/sec |
| **Speed ratio** | 1.00× | **1.31×** |

**Numerical equivalence** (15K+ steps):
- Loss trajectory relative error: ≤ 6e-08 (essentially identical)
- sq_row / sq_col state deviation: ~1e-11
- exp_avg deviation: ~1e-7 (due to `rsqrtf` fast-math ULP accumulation — no effect on training quality)

## Build Details

The extension uses JIT compilation via `torch.utils.cpp_extension.load` — no manual build step required. On first import per session:

1. NVCC compiles `came_cuda_kernel.cu` → object file
2. MSVC/GCC compiles `came_op.cpp` → object file
3. Linker produces `.pyd`/`.so` shared library
4. Cached in `torch.utils.cpp_extension._get_build_directory()` for subsequent imports

First import takes ~5–10 seconds; subsequent imports in the same session are instant.

### File Structure

```
library/training/came_cpp_extension/
├── __init__.py              # Python wrapper + dynamic JIT loading
├── came_op.cpp              # Dispatch layer (0 ATen math, only torch::empty for scratch)
├── came_cuda_kernel.cu      # 8 fused CUDA kernels + extern "C" launchers
├── setup.py                 # Build script (stand install / CI)
└── README.md                # This file
```

## Testing

### Equivalence Tests

Verify numerical parity with Python CAME:

```bash
pytest tests/test_came_c_kernel_equiv.py -v
```

15 tests total: 11 unit-level state equivalence + 4 end-to-end training convergence tests.

### Benchmark

Sustained 30-second benchmark comparing wall-clock time and loss trajectory:

```bash
python tests/bench_came_c.py
```

## Troubleshooting

### "CAME_C extension not available"

The extension failed to compile or load. Check:

1. CUDA Toolkit installed? `nvcc --version`
2. PyTorch has CUDA support? `python -c "import torch; print(torch.cuda.is_available())"`
3. Build succeeded? Try clearing the JIT cache:
   ```bash
   python -c "import torch.utils.cpp_extension; import shutil; shutil.rmtree(torch.utils.cpp_extension._get_build_directory('came_cpp_extension'), ignore_errors=True)"
   ```

### "DLL load failed" (Windows)

The JIT compilation handles this automatically. If problems persist, add PyTorch lib directory to PATH:

```python
import os
os.environ['PATH'] = r'C:\path\to\torch\lib' + os.pathsep + os.environ.get('PATH', '')
```

### CUDA version mismatch

PyTorch compiled with CUDA 12.9 may refuse extensions built with CUDA 13.0. `setup.py` patches `TORCH_CUDA_ARCH_LIST` to allow minor version mismatches (safe for 13.0 → 12.9).

## References

- Design document: `docs/superpowers/specs/2026-06-12-came-cpp-extension-design.md`
- Python reference: `library/training/came_optimizer.py`
- CAME guide: `docs/guidelines/came.md`
- PyTorch C++ Extension Tutorial: https://pytorch.org/tutorials/advanced/cpp_extension.html
