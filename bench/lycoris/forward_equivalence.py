#!/usr/bin/env python
"""Forward numerical-correctness bench for LyCORIS adapter modules.

Creates small LOCON / LOHA / LOKR modules over ``nn.Linear`` with seeded
(known) decomposed weights, runs a forward pass in eval mode, and compares
the output against a hand-assembled reference weight.  Reports max absolute
error and relative L2 error per variant.

No pre-trained model or dataset is required — the bench is self-contained.
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench._common import make_run_dir, write_result
from library.log import setup_logging
from networks.lora_modules.locon import LoConModule
from networks.lora_modules.loha import LohaModule
from networks.lora_modules.lokr import LokrModule
from networks.lora_modules.lycoris_functional import make_kron

setup_logging()
logger = logging.getLogger(__name__)


def _rand_input(batch, features, dtype, device, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, features, generator=gen, dtype=dtype, device=device)


def _scalar_val(mod):
    s = mod.scalar
    return s.data.float() if isinstance(s, nn.Parameter) else s.float()


def _check_locon(in_f, out_f, rank, alpha, dtype, device, seed):
    torch.manual_seed(seed)
    org = nn.Linear(in_f, out_f, bias=False)
    org.weight.data.zero_()
    mod = LoConModule("bench_locon", org, lora_dim=rank, alpha=alpha)
    mod.org_forward = lambda x, _w=org.weight: F.linear(x, _w.to(x.dtype))
    mod.eval()

    w_up = mod.lora_up.weight.data.float()
    w_down = mod.lora_down.weight.data.float()
    expected_w = (w_up @ w_down).view(mod.shape)
    expected_w = expected_w * mod.scale * _scalar_val(mod)

    x = _rand_input(4, in_f, dtype, device, seed + 100)
    with torch.no_grad():
        out = mod(x).float()

    ref = F.linear(x.float(), expected_w) * mod.multiplier
    diff = (out - ref).abs()
    return {"max_abs_error": diff.max().item(), "rel_l2": (diff.norm() / ref.norm().clamp_min(1e-12)).item()}


def _check_loha(in_f, out_f, rank, alpha, dtype, device, seed):
    torch.manual_seed(seed)
    org = nn.Linear(in_f, out_f, bias=False)
    org.weight.data.zero_()
    mod = LohaModule("bench_loha", org, lora_dim=rank, alpha=alpha)
    mod.org_forward = lambda x, _w=org.weight: F.linear(x, _w.to(x.dtype))
    mod.eval()

    w1a = mod.hada_w1_a.data.float()
    w1b = mod.hada_w1_b.data.float()
    w2a = mod.hada_w2_a.data.float()
    w2b = mod.hada_w2_b.data.float()
    scale = torch.tensor(mod.scale, dtype=torch.float32)
    expected_w = ((w1a @ w1b) * (w2a @ w2b)) * scale * _scalar_val(mod)

    x = _rand_input(4, in_f, dtype, device, seed + 200)
    with torch.no_grad():
        out = mod(x).float()

    ref = F.linear(x.float(), expected_w) * mod.multiplier
    diff = (out - ref).abs()
    return {"max_abs_error": diff.max().item(), "rel_l2": (diff.norm() / ref.norm().clamp_min(1e-12)).item()}


def _check_lokr(in_f, out_f, rank, alpha, dtype, device, seed, decompose_both=False):
    torch.manual_seed(seed)
    org = nn.Linear(in_f, out_f, bias=False)
    org.weight.data.zero_()
    mod = LokrModule(
        "bench_lokr", org, lora_dim=rank, alpha=alpha,
        decompose_both=decompose_both,
    )
    mod.org_forward = lambda x, _w=org.weight: F.linear(x, _w.to(x.dtype))
    mod.eval()

    if mod.use_w1:
        w1 = mod.lokr_w1.data.float()
    else:
        w1 = mod.lokr_w1_a.data.float() @ mod.lokr_w1_b.data.float()

    if mod.use_w2:
        w2 = mod.lokr_w2.data.float()
    else:
        w2a = mod.lokr_w2_a.data.float()
        w2b = mod.lokr_w2_b.data.float()
        if w2b.dim() > 2:
            r, o, *k = w2b.shape
            w2 = (w2a @ w2b.view(r, -1)).view(-1, o, *k)
        else:
            w2 = w2a @ w2b

    expected_w = make_kron(w1, w2, mod.scale) * _scalar_val(mod)
    expected_w = expected_w.view(mod.shape)

    x = _rand_input(4, in_f, dtype, device, seed + 300)
    with torch.no_grad():
        out = mod(x).float()

    ref = F.linear(x.float(), expected_w) * mod.multiplier
    diff = (out - ref).abs()
    return {
        "max_abs_error": diff.max().item(),
        "rel_l2": (diff.norm() / ref.norm().clamp_min(1e-12)).item(),
        "use_w1": mod.use_w1,
        "use_w2": mod.use_w2,
    }


def _aggregate(rows):
    return {
        "max_abs_error_mean": sum(r["max_abs_error"] for r in rows) / len(rows),
        "max_abs_error_max": max(r["max_abs_error"] for r in rows),
        "rel_l2_mean": sum(r["rel_l2"] for r in rows) / len(rows),
        "rel_l2_max": max(r["rel_l2"] for r in rows),
        "n_trials": len(rows),
    }


def _run_trials(check_fn, n_trials, **kw):
    rows = []
    base_seed = kw.pop("base_seed", 42)
    for t in range(n_trials):
        rows.append(check_fn(seed=base_seed + t * 7, **kw))
    return _aggregate(rows), rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_features", type=int, default=16)
    p.add_argument("--out_features", type=int, default=32)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None, help="default: rank (scale=1)")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_trials", type=int, default=8)
    p.add_argument("--label", default="forward-equiv")
    args = p.parse_args()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    alpha = args.alpha if args.alpha is not None else args.rank

    metrics = {}
    common = dict(in_f=args.in_features, out_f=args.out_features,
                  rank=args.rank, alpha=alpha, dtype=dtype, device=device)

    locon_agg, _ = _run_trials(_check_locon, args.n_trials, **common, base_seed=args.seed)
    metrics["locon"] = locon_agg
    print(f"LOCON    max_abs={locon_agg['max_abs_error_max']:.2e}  rel_l2={locon_agg['rel_l2_max']:.2e}")

    loha_agg, _ = _run_trials(_check_loha, args.n_trials, **common, base_seed=args.seed + 1000)
    metrics["loha"] = loha_agg
    print(f"LOHA     max_abs={loha_agg['max_abs_error_max']:.2e}  rel_l2={loha_agg['rel_l2_max']:.2e}")

    lokr_agg, _ = _run_trials(_check_lokr, args.n_trials, **common, base_seed=args.seed + 2000)
    metrics["lokr"] = lokr_agg
    print(f"LOKR     max_abs={lokr_agg['max_abs_error_max']:.2e}  rel_l2={lokr_agg['rel_l2_max']:.2e}")

    dec_kw = {**common, "rank": min(args.rank, 2), "decompose_both": True}
    lokr_dec_agg, lokr_dec_rows = _run_trials(_check_lokr, args.n_trials, **dec_kw, base_seed=args.seed + 3000)
    uses = lokr_dec_rows[0] if lokr_dec_rows else {}
    metrics["lokr_decomposed"] = {**lokr_dec_agg, "use_w1": uses.get("use_w1"), "use_w2": uses.get("use_w2")}
    print(f"LOKR(dc) max_abs={lokr_dec_agg['max_abs_error_max']:.2e}  rel_l2={lokr_dec_agg['rel_l2_max']:.2e}")

    threshold = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 5e-2}[args.dtype]
    all_pass = True
    for name, agg in metrics.items():
        err = agg["max_abs_error_max"]
        tag = "PASS" if err <= threshold else "FAIL"
        if tag == "FAIL":
            all_pass = False
        print(f"  {tag}  {name}: max_abs_error={err:.2e} (threshold={threshold:.0e})")
    metrics["verdict"] = "PASS" if all_pass else "FAIL"
    metrics["threshold"] = threshold

    run_dir = make_run_dir("lycoris", label=args.label)
    result_path = write_result(
        run_dir, script=__file__, args=args,
        label=args.label, metrics=metrics, device=device,
    )
    logger.info(f"result -> {result_path}")


if __name__ == "__main__":
    main()
