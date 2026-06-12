"""
Build script for LoKR CUDA extension.

Usage:
    cd networks/lora_modules/lokr_cpp_extension
    python setup.py develop
    # OR
    python setup.py build_ext --inplace

Environment variables:
    TORCH_CUDA_ARCH_LIST: Specify target GPU architectures
    MAX_JOBS: Number of parallel build jobs
"""

import os
import sys
from setuptools import setup

# Patch PyTorch's CUDA version check (same as CAME_C)
try:
    import torch.utils.cpp_extension as cpp_ext
    original_check = cpp_ext._check_cuda_version
    def patched_check(compiler_name, compiler_version):
        pass
    cpp_ext._check_cuda_version = patched_check
except Exception:
    pass

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_arch_flags():
    env_arch = os.getenv("TORCH_CUDA_ARCH_LIST")
    if env_arch:
        archs = env_arch.split(";")
        return [f"-arch=sm_{arch.replace('.', '')}" for arch in archs]
    return [
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


if sys.platform == "win32":
    cxx_flags = ["/O2", "/std:c++17", "/DNDEBUG", "/wd4819"]
else:
    cxx_flags = ["-O3", "-std=c++17", "-DNDEBUG", "-fPIC", "-Wno-unused-variable"]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-Xptxas=--disable-warnings",
    "--expt-relaxed-constexpr",
    "-lineinfo",
]

if sys.platform == "win32":
    nvcc_flags.extend(["-Xcompiler", "/wd4819"])
else:
    nvcc_flags.extend(["-Xcompiler=-fPIC"])


setup(
    name="lokr_cpp",
    version="1.0.0",
    description="LoKR fused CUDA kernels for Kronecker-factored LoRA",
    ext_modules=[
        CUDAExtension(
            name="lokr_cpp",
            sources=[
                "lokr_op.cpp",
                "lokr_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags + get_arch_flags(),
            },
            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
            ],
        )
    },
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,
        )
    },
)
