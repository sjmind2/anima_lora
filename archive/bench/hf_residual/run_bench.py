#!/usr/bin/env python
"""HF-residual Tier 1 diagnostic — no training, no adapter.

Probes whether the structural LF/HF velocity split proposed in
``docs/proposal/hf_residual_adapter.md`` has any headroom before paying
the cost of building the adapter. Two questions:

1. **Headroom Δ_LF.** How much FM loss is left on the table when the
   adapter is missing? Forward base on full `x_t` (baseline) vs forward
   base on `x_t^L = blur(x_t, σ_low)` (LF-only). Gap = work the HF
   adapter has to recover.
2. **Signal budget.** How much HF mass is actually present per σ_t? If
   `Var(x_t^H) / Var(x_t)` is tiny at the timesteps that matter, the
   adapter has nothing to model.

Both are swept across ``--sigma_low_divs`` (default ``2,4,8``) so the
divisor can be re-tuned for the HF-residual setup rather than inherited
from VR-loss / FEI-router co-tuning. See README.md for verdict
thresholds.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.runtime.fei import fei_sigma_low, gaussian_blur_2d  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


log = logging.getLogger("hf_residual_tier1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-preview3-base.safetensors",
        help="Frozen base DiT used for both full and LF forwards.",
    )
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument(
        "--bucket",
        default=None,
        help="Latent bucket WxH. Defaults to the most populous bucket in --data_dir.",
    )
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--num_timesteps", type=int, default=6)
    p.add_argument("--num_noise", type=int, default=16)
    p.add_argument("--noise_batch_size", type=int, default=8)
    p.add_argument("--t_min", type=float, default=0.10)
    p.add_argument("--t_max", type=float, default=0.85)
    p.add_argument(
        "--sigma_low_divs",
        default="2,4,8",
        help="Comma-separated FEI sigma_low divisors to sweep. "
             "σ_low = min(H_lat, W_lat) / div.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def discover_samples(data_dir: Path, bucket: str | None, num_samples: int, seed: int):
    """Pair (latent npz, text-emb safetensors) under the chosen bucket."""
    res_re = re.compile(r"_(\d{3,5})x(\d{3,5})_anima\.npz$")
    npz_paths = sorted(glob.glob(str(data_dir / "*_anima.npz")))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` in {data_dir}")
    by_bucket: dict[str, list[tuple[str, str, str, str]]] = {}
    for p in npz_paths:
        name = Path(p).name
        m = res_re.search(name)
        if not m:
            continue
        stem = name[: m.start()]
        te = data_dir / f"{stem}_anima_te.safetensors"
        if not te.exists():
            continue
        with np.load(p) as z:
            for k in z.keys():
                if k.startswith("latents_"):
                    bk = k.removeprefix("latents_")
                    by_bucket.setdefault(bk, []).append((stem, k, p, str(te)))
                    break
    if not by_bucket:
        raise SystemExit("no paired (latent, TE) samples found")
    chosen = bucket or max(by_bucket, key=lambda k: len(by_bucket[k]))
    if chosen not in by_bucket:
        raise SystemExit(
            f"bucket {chosen!r} not found. Top buckets: "
            f"{sorted(((k, len(v)) for k, v in by_bucket.items()), key=lambda x: -x[1])[:10]}"
        )
    pool = by_bucket[chosen]
    if len(pool) < num_samples:
        raise SystemExit(
            f"bucket {chosen!r} has only {len(pool)} samples; need {num_samples}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=num_samples, replace=False)
    return chosen, [pool[i] for i in idx]


def load_pair(npz_path: str, latent_key: str, te_path: str, device, dtype):
    z = np.load(npz_path)
    x0 = torch.from_numpy(z[latent_key]).to(device=device, dtype=dtype)  # (C, H, W)
    z.close()
    te_data = load_file(te_path)
    crossattn = te_data["crossattn_emb_v0"].to(device=device, dtype=dtype)
    return x0, crossattn


@torch.inference_mode()
def predict_velocity_batch(anima, x_t_batch, t_scalar, crossattn_B):
    """Return u_pred = base(x_t, t, te) shaped (B, C, H, W)."""
    B, C, H, W = x_t_batch.shape
    x_5d = x_t_batch.unsqueeze(2)
    timesteps = torch.full(
        (B,), t_scalar * 1000.0, dtype=x_t_batch.dtype, device=x_t_batch.device
    )
    padding_mask = torch.zeros(
        B, 1, H, W, dtype=x_t_batch.dtype, device=x_t_batch.device
    )
    return anima(x_5d, timesteps, crossattn_B, padding_mask=padding_mask).squeeze(2)


def variance_fraction(x_high: torch.Tensor, x_full: torch.Tensor) -> float:
    """||x_high||² / ||x_full||² — band-energy fraction, in fp32."""
    h = x_high.float()
    f = x_full.float()
    num = h.pow(2).sum().item()
    den = f.pow(2).sum().clamp_min(1e-30).item()
    return num / den


def mean_sq(x: torch.Tensor) -> float:
    """Mean of squared values, in fp32 (= FM loss when x is residual)."""
    return x.float().pow(2).mean().item()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = make_run_dir("hf_residual", label=args.label)
    log.info(f"output → {run_dir}")

    divs = [float(s) for s in args.sigma_low_divs.split(",") if s.strip()]
    log.info(f"sweeping sigma_low_div ∈ {divs}")

    data_dir = Path(args.data_dir)
    bucket, samples = discover_samples(data_dir, args.bucket, args.num_samples, args.seed)
    log.info(f"bucket={bucket} num_samples={len(samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    log.info(f"loading frozen base DiT: {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).eval().requires_grad_(False)
    anima.reset_mod_guidance()

    timesteps = torch.linspace(args.t_min, args.t_max, args.num_timesteps)
    log.info(f"timesteps: {[round(t.item(), 3) for t in timesteps]}")

    h_lat, w_lat = (int(x) for x in bucket.split("x"))
    sigma_lows = {div: fei_sigma_low(h_lat, w_lat, div) for div in divs}
    log.info(f"sigma_low per div: {sigma_lows}")

    rng = torch.Generator(device=device).manual_seed(args.seed)

    rows: list[dict] = []
    total_pairs = len(samples) * len(timesteps)
    pbar = tqdm(total=total_pairs, desc="hf-tier1", dynamic_ncols=True)

    for si, (stem, latent_key, npz_path, te_path) in enumerate(samples):
        x0, crossattn = load_pair(npz_path, latent_key, te_path, device, dtype)
        x0_4d = x0.unsqueeze(0)
        crossattn_B = crossattn.unsqueeze(0)

        for t_scalar in timesteps:
            t = float(t_scalar)
            alpha_t = 1.0 - t
            sigma_t = t

            N = args.num_noise
            eps = torch.randn(
                (N, *x0.shape), generator=rng, device=device, dtype=dtype
            )
            x_t = alpha_t * x0_4d + sigma_t * eps  # (N, C, H, W)
            u_target = eps - x0_4d  # (N, C, H, W)

            # Full forward (independent of div) — cache once and reuse.
            u_pred_full_chunks = []
            for i in range(0, N, args.noise_batch_size):
                xb = x_t[i : i + args.noise_batch_size]
                B_eff = xb.shape[0]
                ca_B = crossattn_B.expand(B_eff, -1, -1).contiguous()
                u_pred_full_chunks.append(predict_velocity_batch(anima, xb, t, ca_B))
            u_pred_full = torch.cat(u_pred_full_chunks, dim=0)
            loss_full = mean_sq(u_pred_full - u_target)

            for div, sigma_low in sigma_lows.items():
                # Inference-style band split: x_t^L = blur(x_t).
                x_t_L = gaussian_blur_2d(x_t, sigma_low)
                x_t_H = x_t - x_t_L

                u_pred_lf_chunks = []
                for i in range(0, N, args.noise_batch_size):
                    xb_L = x_t_L[i : i + args.noise_batch_size]
                    B_eff = xb_L.shape[0]
                    ca_B = crossattn_B.expand(B_eff, -1, -1).contiguous()
                    u_pred_lf_chunks.append(predict_velocity_batch(anima, xb_L, t, ca_B))
                u_pred_lf = torch.cat(u_pred_lf_chunks, dim=0)

                loss_lf = mean_sq(u_pred_lf - u_target)
                gap = loss_lf - loss_full
                gap_ratio = gap / max(loss_full, 1e-30)

                # Band decomposition of the target velocity:
                u_target_L = gaussian_blur_2d(u_target, sigma_low)
                u_target_H = u_target - u_target_L
                # And of the predictions:
                u_pred_full_L = gaussian_blur_2d(u_pred_full, sigma_low)
                u_pred_full_H = u_pred_full - u_pred_full_L
                u_pred_lf_L = gaussian_blur_2d(u_pred_lf, sigma_low)
                u_pred_lf_H = u_pred_lf - u_pred_lf_L

                # Per-band FM losses:
                loss_full_L = mean_sq(u_pred_full_L - u_target_L)
                loss_full_H = mean_sq(u_pred_full_H - u_target_H)
                loss_lf_L = mean_sq(u_pred_lf_L - u_target_L)
                loss_lf_H = mean_sq(u_pred_lf_H - u_target_H)

                # The HF adapter's training residual is:
                #   adapter_target = u_target - base(x_t^L)
                # Its norm fraction tells you the budget the adapter
                # has to fit (in HF-band) vs LF (which should be small).
                adapter_target = u_target - u_pred_lf
                adapter_target_L = gaussian_blur_2d(adapter_target, sigma_low)
                adapter_target_H = adapter_target - adapter_target_L
                adapter_target_norm = mean_sq(adapter_target)
                adapter_target_H_norm = mean_sq(adapter_target_H)
                adapter_target_L_norm = mean_sq(adapter_target_L)
                adapter_target_H_frac = (
                    adapter_target_H_norm
                    / max(adapter_target_norm, 1e-30)
                )

                rows.append({
                    "stem": stem,
                    "t": t,
                    "sigma_low_div": div,
                    "sigma_low": sigma_low,
                    "bucket": bucket,
                    # Headline:
                    "loss_full": loss_full,
                    "loss_lf": loss_lf,
                    "gap": gap,
                    "gap_ratio": gap_ratio,
                    # Per-band FM loss (full forward):
                    "loss_full_lband": loss_full_L,
                    "loss_full_hband": loss_full_H,
                    # Per-band FM loss (LF-only forward):
                    "loss_lf_lband": loss_lf_L,
                    "loss_lf_hband": loss_lf_H,
                    # Input-side band fractions:
                    "xt_hband_frac": variance_fraction(x_t_H, x_t),
                    "utarget_hband_frac": variance_fraction(u_target_H, u_target),
                    # Output-side band fractions (what the base actually emits):
                    "upred_full_hband_frac": variance_fraction(u_pred_full_H, u_pred_full),
                    "upred_lf_hband_frac": variance_fraction(u_pred_lf_H, u_pred_lf),
                    # Adapter's job:
                    "adapter_target_norm": adapter_target_norm,
                    "adapter_target_hband_norm": adapter_target_H_norm,
                    "adapter_target_lband_norm": adapter_target_L_norm,
                    "adapter_target_hband_frac": adapter_target_H_frac,
                })

            postfix = {
                "t": f"{t:.2f}",
                "loss_full": f"{loss_full:.3f}",
                "gap[div=4]": f"{[r['gap'] for r in rows if r['stem']==stem and r['t']==t and r['sigma_low_div']==4.0][0]:.3f}"
                if any(d == 4.0 for d in divs) else "-",
            }
            pbar.set_postfix(postfix)
            pbar.update(1)

        del x0, x0_4d, crossattn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    csv_path = run_dir / "per_sample_t.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregations: per div × per t.
    def agg(vals: list[float]) -> dict:
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
        a = np.array(vals, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std()),
            "n": int(a.size),
        }

    def filter_div(div: float):
        return [r for r in rows if r["sigma_low_div"] == div]

    summary: dict = {
        "bucket": bucket,
        "n_samples": len(samples),
        "n_timesteps": len(timesteps),
        "n_noise": args.num_noise,
        "n_rows": len(rows),
        "dit": args.dit,
        "sigma_lows": sigma_lows,
        "per_div": {},
    }

    for div in divs:
        sub = filter_div(div)
        per_t: dict[str, dict] = {}
        for t_bin in sorted({round(r["t"], 3) for r in sub}):
            sub_t = [r for r in sub if round(r["t"], 3) == t_bin]
            per_t[f"{t_bin:.3f}"] = {
                "gap_ratio": agg([r["gap_ratio"] for r in sub_t]),
                "gap": agg([r["gap"] for r in sub_t]),
                "loss_full": agg([r["loss_full"] for r in sub_t]),
                "loss_lf": agg([r["loss_lf"] for r in sub_t]),
                "xt_hband_frac": agg([r["xt_hband_frac"] for r in sub_t]),
                "utarget_hband_frac": agg([r["utarget_hband_frac"] for r in sub_t]),
                "upred_full_hband_frac": agg([r["upred_full_hband_frac"] for r in sub_t]),
                "upred_lf_hband_frac": agg([r["upred_lf_hband_frac"] for r in sub_t]),
                "adapter_target_hband_frac": agg([r["adapter_target_hband_frac"] for r in sub_t]),
            }

        summary["per_div"][str(div)] = {
            "sigma_low": sigma_lows[div],
            "gap_ratio_overall": agg([r["gap_ratio"] for r in sub]),
            "gap_overall": agg([r["gap"] for r in sub]),
            "loss_full_overall": agg([r["loss_full"] for r in sub]),
            "loss_lf_overall": agg([r["loss_lf"] for r in sub]),
            "xt_hband_frac_overall": agg([r["xt_hband_frac"] for r in sub]),
            "utarget_hband_frac_overall": agg([r["utarget_hband_frac"] for r in sub]),
            "upred_full_hband_frac_overall": agg([r["upred_full_hband_frac"] for r in sub]),
            "upred_lf_hband_frac_overall": agg([r["upred_lf_hband_frac"] for r in sub]),
            "adapter_target_hband_frac_overall": agg(
                [r["adapter_target_hband_frac"] for r in sub]
            ),
            "per_t": per_t,
        }

    # Verdict — keyed on the central div (4.0) if present, else first.
    pivot_div = 4.0 if 4.0 in divs else divs[0]
    pivot = summary["per_div"][str(pivot_div)]
    gap_ratio_med = pivot["gap_ratio_overall"]["median"]
    xt_h_frac_med = pivot["xt_hband_frac_overall"]["median"]

    # Thresholds (see README): we want gap_ratio above noise (~0.05) AND
    # HF band carrying meaningful mass (~0.05). If both are dead, the
    # adapter has nothing to do; if either is huge, the LF input is so
    # crippled that the adapter has to learn most of the model.
    if gap_ratio_med >= 0.10 and xt_h_frac_med >= 0.05:
        verdict = "HEADROOM"
    elif gap_ratio_med >= 0.03:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"
    summary["verdict"] = verdict
    summary["verdict_pivot_div"] = pivot_div

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=summary,
        artifacts=["per_sample_t.csv", "summary.json"],
        label=args.label,
        device=device,
    )
    log.info(f"[bench] verdict: {verdict} (pivot div={pivot_div})")
    log.info(
        f"[bench]   gap_ratio median={gap_ratio_med:.3f} "
        f"xt_hband_frac median={xt_h_frac_med:.3f}"
    )
    for div in divs:
        d = summary["per_div"][str(div)]
        log.info(
            f"[bench]   div={div:>4} σ_low={d['sigma_low']:.2f}  "
            f"gap_ratio={d['gap_ratio_overall']['median']:+.3f}  "
            f"xt_H/xt={d['xt_hband_frac_overall']['median']:.3f}  "
            f"adapter_H/all={d['adapter_target_hband_frac_overall']['median']:.3f}"
        )


if __name__ == "__main__":
    main()
