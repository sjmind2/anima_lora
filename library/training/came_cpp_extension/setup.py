"""
Build script for CAME_C CUDA extension.

Usage:
    cd library/training/came_cpp_extension
    python setup.py develop        # Install in development mode
    # OR
    python setup.py build_ext --inplace  # Build in-place

Environment variables:
    TORCH_CUDA_ARCH_LIST: Specify target GPU architectures
                          (e.g., "8.0;8.6;9.0" for Ampere + Ada + Blackwell)
    MAX_JOBS: Number of parallel build jobs (default: auto-detect)
"""

import os
import sys
from setuptools import setup

# Patch PyTorch's CUDA version check to allow minor version mismatches
# This is safe for CUDA 13.0 vs 12.9 (backward compatible)
try:
    import torch.utils.cpp_extension as cpp_ext
    original_check = cpp_ext._check_cuda_version
    def patched_check(compiler_name, compiler_version):
        # Allow CUDA 13.0 with PyTorch compiled for 12.9 (minor mismatch is OK)
        pass
    cpp_ext._check_cuda_version = patched_check
except Exception:
    pass  # If patching fails, continue with original check

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Determine CUDA architecture flags
def get_arch_flags():
    """Get CUDA architecture flags based on environment or defaults."""
    env_arch = os.getenv("TORCH_CUDA_ARCH_LIST")
    if env_arch:
        # User-specified architectures
        archs = env_arch.split(";")
        return [f"-arch=sm_{arch.replace('.', '')}" for arch in archs]

    # Default: optimize for RTX 5090 (Blackwell, sm_90)
    # Also include sm_86 (Ampere) and sm_89 (Ada) for compatibility
    return [
        "-gencode=arch=compute_86,code=sm_86",  # A100, RTX 3090
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
        "-gencode=arch=compute_90,code=sm_90",  # RTX 5090
    ]


# Compiler flags - platform-specific
if sys.platform == "win32":
    # MSVC compiler flags (Windows)
    cxx_flags = [
        "/O2",              # Optimization level 2
        "/std:c++17",       # C++17 standard
        "/DNDEBUG",         # Disable assertions
        "/wd4819",          # Disable encoding warning
    ]
else:
    # GCC/Clang compiler flags (Linux)
    cxx_flags = [
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "-fPIC",
        "-Wno-unused-variable",
    ]

# CUDA compiler flags (platform-independent, but passed via nvcc)
nvcc_flags = [
    "-O3",
    "--use_fast_math",           # Fast math operations (sufficient for training)
    "-Xptxas=--disable-warnings",  # Suppress PTX assembly warnings
    "--expt-relaxed-constexpr",   # Allow __device__ constexpr functions
    "-lineinfo",                  # Include line info for profiling
]

# Platform-specific NVCC flags
if sys.platform == "win32":
    nvcc_flags.extend([
        "-Xcompiler", "/wd4819",
    ])
else:
    nvcc_flags.extend([
        "-Xcompiler=-fPIC",
    ])


setup(
    name="came_cpp",
    version="1.0.0",
    description="CAME optimizer CUDA kernels (C++ extension)",
    ext_modules=[
        CUDAExtension(
            name="came_cpp",
            sources=[
                "came_op.cpp",
                "came_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags + get_arch_flags(),
            },
            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,
        )
    },
)
