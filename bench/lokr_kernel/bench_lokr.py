#!/usr/bin/env python
"""Benchmark: LoKR fused CUDA kernels vs PyTorch reference.

Measures wall-clock time for forward, backward, and combined forward+backward
on Anima DiT realistic shapes. Compares:
  - CUDA kernel path (KronLinearFnCUDA / KronLinearTwoStageFnCUDA)
  - PyTorch reference path (KronLinearFn / KronLinearTwoStageFn)

Usage:
    uv run python bench/lokr_kernel/bench_lokr.py
    uv run python bench/lokr_kernel/bench_lokr.py --warmup 50 --iters 200

Output:
    bench/lokr_kernel/results/<YYYYMMDD-HHMM>/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bootstrap bench/ import
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from bench._common import make_run_dir, write_result


# ============================================================================
# Benchmark shapes — Anima DiT with factor=8
# ============================================================================

# Single-stage: (name, B, in_m, in_n, out_l, out_k)
SINGLE_STAGE_SHAPES = [
    ("qkv_proj",     4032, 8, 256, 8, 768),
    ("output_proj",  4032, 8, 256, 8, 256),
    ("gate_up_proj", 4032, 8, 256, 8, 1024),
    ("down_proj",    4032, 8, 1024, 8, 256),
]

# Two-stage: (name, B, in_m, in_n, out_l, out_k, r1, r2)
# Anima realistic shapes: lokr_factor=-1 → factorization(d, -1)
#   in_dim=2048 → (32,64); out_dim=2048→(32,64); out_dim=6144→(64,96);
#   out_dim=8192→(64,128); in_dim=8192→(64,128)
# r1=r2=8 (network_dim=8)
TWO_STAGE_SHAPES = [
    # Anima realistic shapes (decompose_both=true, lokr_factor=-1, dim=8)
    ("qkv_real",     4032, 32, 64, 64, 96, 8, 8),
    ("output_real",  4032, 32, 64, 32, 64, 8, 8),
    ("gate_up_real", 4032, 32, 64, 64, 128, 8, 8),
    ("down_real",    4032, 64, 128, 32, 64, 8, 8),
    # Small shapes for correctness microbench
    ("qkv_small",   512, 8, 64, 8, 64, 8, 8),
    ("output_small", 512, 8, 64, 8, 256, 8, 8),
]


def _bench_fn(fn_apply, args, warmup, iters):
    """Benchmark a function, returning median time in ms."""
    for _ in range(warmup):
        fn_apply(*args)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn_apply(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def bench_single_stage(name, B, in_m, in_n, out_l, out_k, warmup, iters, device="cuda"):
    """Benchmark single-stage forward+backward."""
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, in_m * in_n, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(out_l, in_m, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(out_k, in_n, device=device, dtype=dtype, requires_grad=True)
    scalar = torch.tensor(0.5, device=device)

    results = {}

    # Import both implementations
    from networks.lora_modules.lokr_cpp_extension import (
        KronLinearFnCUDA,
        _LOKR_KERNEL_AVAILABLE,
    )
    from networks.lora_modules.lokr import _KronLinearFn_PyTorch as KronLinearFnRef

    def _fwd_bwd(fn_cls):
        def run():
            if x.grad is not None:
                x.grad = None
                w1.grad = None
                w2.grad = None
            out = fn_cls.apply(x.contiguous(), w1, w2, scalar)
            out.backward(torch.ones_like(out))
        return run

    results["pytorch_ms"] = _bench_fn(_fwd_bwd(KronLinearFnRef), (), warmup, iters)

    if _LOKR_KERNEL_AVAILABLE:
        results["cuda_ms"] = _bench_fn(_fwd_bwd(KronLinearFnCUDA), (), warmup, iters)
        results["speedup"] = results["pytorch_ms"] / results["cuda_ms"]
    else:
        results["cuda_ms"] = None
        results["speedup"] = None

    return results


def bench_two_stage(name, B, in_m, in_n, out_l, out_k, r1, r2, warmup, iters, device="cuda"):
    """Benchmark two-stage forward+backward."""
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, in_m * in_n, device=device, dtype=dtype, requires_grad=True)
    w1_a = torch.randn(out_l, r1, device=device, dtype=dtype, requires_grad=True)
    w1_b = torch.randn(r1, in_m, device=device, dtype=dtype, requires_grad=True)
    w2_a = torch.randn(out_k, r2, device=device, dtype=dtype, requires_grad=True)
    w2_b = torch.randn(r2, in_n, device=device, dtype=dtype, requires_grad=True)
    scalar = torch.tensor(0.5, device=device)

    results = {}

    from networks.lora_modules.lokr_cpp_extension import (
        KronLinearTwoStageFnCUDA,
        _LOKR_KERNEL_AVAILABLE,
    )
    from networks.lora_modules.lokr import _KronLinearTwoStageFn_PyTorch as KronLinearTwoStageFnRef

    def _fwd_bwd(fn_cls):
        def run():
            if x.grad is not None:
                for p in [x, w1_a, w1_b, w2_a, w2_b]:
                    p.grad = None
            out = fn_cls.apply(x.contiguous(), w1_a, w1_b, w2_a, w2_b, scalar)
            out.backward(torch.ones_like(out))
        return run

    results["pytorch_ms"] = _bench_fn(_fwd_bwd(KronLinearTwoStageFnRef), (), warmup, iters)

    if _LOKR_KERNEL_AVAILABLE:
        results["cuda_ms"] = _bench_fn(_fwd_bwd(KronLinearTwoStageFnCUDA), (), warmup, iters)
        results["speedup"] = results["pytorch_ms"] / results["cuda_ms"]
    else:
        results["cuda_ms"] = None
        results["speedup"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoKR CUDA kernels")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timing iterations")
    parser.add_argument("--label", type=str, default=None, help="Run label")
    parser.add_argument("--no-large", action="store_true", help="Skip large shapes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA required for benchmarking")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    print()

    metrics = {"single_stage": {}, "two_stage": {}}

    shapes = SINGLE_STAGE_SHAPES if not args.no_large else [
        ("qkv_small", 64, 8, 64, 8, 64),
    ]
    print("=== Single-Stage (KronLinearFn) ===")
    for name, B, in_m, in_n, out_l, out_k in shapes:
        print(f"\n  {name}: B={B}, in_m={in_m}, in_n={in_n}, out_l={out_l}, out_k={out_k}")
        r = bench_single_stage(name, B, in_m, in_n, out_l, out_k, args.warmup, args.iters, device)
        r["B"] = B
        r["in_m"] = in_m
        r["in_n"] = in_n
        r["out_l"] = out_l
        r["out_k"] = out_k
        metrics["single_stage"][name] = r
        print(f"    PyTorch: {r['pytorch_ms']:.3f} ms/iter")
        if r["cuda_ms"] is not None:
            print(f"    CUDA:    {r['cuda_ms']:.3f} ms/iter  (speedup: {r['speedup']:.2f}x)")
        else:
            print(f"    CUDA:    N/A (extension not available)")

    ts_shapes = TWO_STAGE_SHAPES
    print("\n=== Two-Stage (KronLinearTwoStageFn) ===")
    for name, B, in_m, in_n, out_l, out_k, r1, r2 in ts_shapes:
        print(f"\n  {name}: B={B}, in_m={in_m}, in_n={in_n}, r1={r1}, r2={r2}")
        r = bench_two_stage(name, B, in_m, in_n, out_l, out_k, r1, r2, args.warmup, args.iters, device)
        r["B"] = B
        r["r1"] = r1
        r["r2"] = r2
        metrics["two_stage"][name] = r
        print(f"    PyTorch: {r['pytorch_ms']:.3f} ms/iter")
        if r["cuda_ms"] is not None:
            print(f"    CUDA:    {r['cuda_ms']:.3f} ms/iter  (speedup: {r['speedup']:.2f}x)")
        else:
            print(f"    CUDA:    N/A")

    # Write result envelope
    run_dir = make_run_dir("lokr_kernel", label=args.label)
    write_result(run_dir, script=__file__, args=args, metrics=metrics, device=device)
    print(f"\nResult written to: {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
