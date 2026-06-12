"""
Sustained benchmark: Python CAME vs CAME_C (fused CUDA kernels).

Runs each optimizer for ~30 seconds (after 30% warmup) and compares:
  - Wall-clock time per step
  - Final parameter values (numerical drift)
  - Loss trajectory divergence

Usage:
    python tests/bench_came_c.py
    python tests/bench_came_c.py --duration 60
    python tests/bench_came_c.py --shape 16 4096
"""

import argparse
import copy
import math
import time

import torch
import torch.nn as nn

from library.training.came_optimizer import CAME
from library.training.came_cpp_extension import CAME_C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM = 256
RANK = 16
BATCH_SIZE = 4
DTYPE = torch.float32
DEVICE = "cuda"
LR = 1e-4
BETAS = (0.9, 0.999, 0.9999)
SEED = 42


class SyntheticLoRA(nn.Module):
    def __init__(self, dim=256, rank=16):
        super().__init__()
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=True)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


def make_pair():
    torch.manual_seed(SEED)
    base = SyntheticLoRA(dim=DIM, rank=RANK).to(device=DEVICE, dtype=DTYPE)
    model_py = copy.deepcopy(base)
    model_cpp = copy.deepcopy(base)
    opt_py = CAME(model_py.parameters(), lr=LR, betas=BETAS)
    opt_cpp = CAME_C(model_cpp.parameters(), lr=LR, betas=BETAS)
    return model_py, model_cpp, opt_py, opt_cpp


def train_step(model, optimizer, x, target):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x), target)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def bench_one(model, optimizer, label, duration_s, seed_offset=0):
    """Run optimizer steps until duration_s elapsed. Returns (steps, losses, total_time)."""
    t_start = time.perf_counter()
    steps = 0
    losses = []
    while True:
        torch.manual_seed(seed_offset + steps + 10000)
        x = torch.randn(BATCH_SIZE, 256, DIM, device=DEVICE, dtype=DTYPE)
        target = torch.randn_like(x)
        loss = train_step(model, optimizer, x, target)
        losses.append(loss)
        steps += 1
        if time.perf_counter() - t_start >= duration_s:
            break
    total = time.perf_counter() - t_start
    return steps, losses, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0, help="Benchmark duration in seconds")
    parser.add_argument("--warmup", type=float, default=0.3, help="Warmup fraction (0-1)")
    args = parser.parse_args()

    total_budget = args.duration
    warmup_frac = args.warmup
    warmup_s = total_budget * warmup_frac
    bench_s = total_budget * (1 - warmup_frac)

    print("=" * 66)
    print("  CAME  vs  CAME_C  Sustained Benchmark")
    print("=" * 66)
    print(f"  Model           : SyntheticLoRA(dim={DIM}, rank={RANK})")
    print(f"  Param shapes    : lora_A.weight {RANK}x{DIM},  lora_B.weight {DIM}x{RANK},  bias ({DIM},)")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Dtype           : {DTYPE}")
    print(f"  LR={LR}, betas={BETAS}")
    print(f"  Total budget    : {total_budget:.0f}s  (warmup {warmup_s:.0f}s + bench {bench_s:.0f}s)")
    print("-" * 66)

    # ---- Warmup (triggers JIT / torch.compile) ----
    print(f"\n  [warmup] Warming up for {warmup_s:.0f}s ...")
    model_py_w, model_cpp_w, opt_py_w, opt_cpp_w = make_pair()

    torch.manual_seed(99999)
    x_w = torch.randn(BATCH_SIZE, 256, DIM, device=DEVICE, dtype=DTYPE)
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warmup_s:
        train_step(model_py_w, opt_py_w, x_w, torch.randn_like(x_w))
        train_step(model_cpp_w, opt_cpp_w, x_w, torch.randn_like(x_w))
    del model_py_w, model_cpp_w, opt_py_w, opt_cpp_w
    print("  [warmup] Done.")

    # ---- Benchmark ----
    torch.cuda.synchronize()

    # Python CAME
    model_py, model_cpp, opt_py, opt_cpp = make_pair()
    shared_seed = 50000  # same seed offset for identical data streams

    print(f"\n  [bench] Running Python CAME for {bench_s:.0f}s ...")
    torch.cuda.synchronize()
    steps_py, losses_py, time_py = bench_one(model_py, opt_py, "CAME", bench_s, seed_offset=shared_seed)
    torch.cuda.synchronize()

    print(f"  [bench] Running CAME_C for {bench_s:.0f}s ...")
    torch.cuda.synchronize()
    steps_cpp, losses_cpp, time_cpp = bench_one(model_cpp, opt_cpp, "CAME_C", bench_s, seed_offset=shared_seed)
    torch.cuda.synchronize()

    # ---- Results ----
    time_per_step_py = time_py / steps_py * 1000
    time_per_step_cpp = time_cpp / steps_cpp * 1000
    ratio = (time_cpp / steps_cpp) / (time_py / steps_py)  # per-step ratio

    print()
    print("=" * 66)
    print("  RESULTS")
    print("=" * 66)
    print(f"  {'Metric':<30s} {'CAME (Python)':>14s} {'CAME_C (CUDA)':>14s}")
    print("-" * 66)
    print(f"  {'Total steps':<30s} {steps_py:>14d} {steps_cpp:>14d}")
    print(f"  {'Wall-clock time (ms)':<30s} {time_py*1000:>14.1f} {time_cpp*1000:>14.1f}")
    print(f"  {'Time per step (ms)':<30s} {time_per_step_py:>14.3f} {time_per_step_cpp:>14.3f}")
    print(f"  {'Throughput (steps/sec)':<30s} {steps_py/time_py:>14.1f} {steps_cpp/time_cpp:>14.1f}")
    print(f"  {'Speed ratio (C++/Python)':<30s} {ratio:>14.3f}x")
    if ratio < 1.0:
        print(f"  {'CAME_C advantage':<30s} {1/ratio:>14.2f}x FASTER")
    else:
        print(f"  {'CAME_C disadvantage':<30s} {ratio:>14.2f}x slower")
    print()

    # ---- Numerical comparison ----
    print("-" * 66)
    print("  NUMERICAL COMPARISON (final state)")
    print("-" * 66)

    # Compare per-parameter final values
    for (n1, p1), (n2, p2) in zip(model_py.named_parameters(), model_cpp.named_parameters()):
        assert n1 == n2
        max_diff = (p1.data - p2.data).abs().max().item()
        rel_diff = max_diff / (p1.data.abs().max().item() + 1e-30)
        print(f"  {n1:<30s} max_abs_diff={max_diff:.2e}  rel_diff={rel_diff:.2e}")

    # Compare optimizer states
    state_py = opt_py.state
    state_cpp = opt_cpp.state
    for (n1, p1), (n2, p2) in zip(model_py.named_parameters(), model_cpp.named_parameters()):
        sp = state_py.get(p1, {})
        sc = state_cpp.get(p2, {})
        for key in sp:
            if key == "step":
                continue
            if isinstance(sp[key], torch.Tensor) and isinstance(sc.get(key), torch.Tensor):
                max_diff = (sp[key].float() - sc[key].float()).abs().max().item()
                rel_diff = max_diff / (sp[key].float().abs().max().item() + 1e-30)
                print(f"  {n1}.{key:<24s} max_abs_diff={max_diff:.2e}  rel_diff={rel_diff:.2e}")

    # Loss trajectory comparison
    min_steps = min(len(losses_py), len(losses_cpp))
    print()
    print("-" * 66)
    print("  LOSS TRAJECTORY")
    print("-" * 66)
    print(f"  {'':30s} {'Python':>14s} {'CAME_C':>14s} {'RelErr':>10s}")
    # Sample at 10%, 25%, 50%, 75%, 100%
    for pct in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(pct * min_steps) - 1, min_steps - 1)
        lp, lc = losses_py[idx], losses_cpp[idx]
        denom = max(abs(lp), abs(lc), 1e-8)
        rel = abs(lp - lc) / denom
        label = f"Step {idx+1:>5d} ({pct*100:.0f}%)"
        print(f"  {label:<30s} {lp:>14.6f} {lc:>14.6f} {rel:>10.2e}")

    print()
    print("=" * 66)


if __name__ == "__main__":
    main()
