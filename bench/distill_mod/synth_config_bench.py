"""Compare teacher-synthesized clean latents across (cfg_scale, num_steps) configs.

Question we're answering: is the default (CFG=4, steps=28) materially different
from cheaper alternatives like (CFG=2.5, steps=20) for Phase 2 of
``scripts/distill_mod/prep.py``? If the x0 distributions match within noise,
the trainer doesn't care which trajectory produced them.

Method:
  - Pick N stems from the cached TE pool, stratified across a small bucket
    allowlist (constant-token invariant).
  - For each (cfg, steps) config × each stem: run the frozen teacher from
    the SAME per-stem noise + prompt → clean latent. Time each run.
  - Compute per-config-pair metrics (vs the baseline config, idx 0):
      * mean per-element MSE
      * mean cosine similarity (flattened)
      * per-channel mean/std drift (max over channels)
  - Compare per-config distribution stats against the cached real latents
    sitting next to each TE file.

Run-dir artifacts:
  per_sample.csv   one row per (stem, config_pair)
  per_config.csv   one row per config (timings + distribution stats vs real)
  result.json      standard envelope

Usage::

    python -m bench.distill_mod.synth_config_bench
    python -m bench.distill_mod.synth_config_bench \\
        --configs 4.0:28,2.5:20,4.0:20,2.5:28 --n_per_bucket 6
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as _load_safetensors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from scripts.distill_mod.synth import denoise_one  # noqa: E402
from scripts.distill_mod.uncond import UNCOND_TE_FILENAME  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _parse_configs(spec: str) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        cfg_str, steps_str = tok.split(":")
        out.append((float(cfg_str), int(steps_str)))
    if not out:
        raise SystemExit(f"--configs is empty: {spec!r}")
    return out


def _parse_buckets(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        h_str, w_str = tok.split("x")
        out.append((int(h_str), int(w_str)))
    return out


def _channel_stats(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(C,) per-channel mean and std over (N, H, W)."""
    flat = x.reshape(x.shape[0], -1)
    return flat.mean(dim=1).cpu().numpy(), flat.std(dim=1).cpu().numpy()


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Per-tensor MSE + cosine. ``a``/``b`` are (C, H, W) float32 CPU."""
    af = a.flatten().double()
    bf = b.flatten().double()
    mse = float(((af - bf) ** 2).mean().item())
    cos = float(
        (af @ bf / (af.norm() * bf.norm()).clamp_min(1e-12)).item()
    )
    return {"mse": mse, "cos": cos}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_dir", type=str, default="post_image_dataset/lora"
    )
    parser.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="4.0:28,2.5:20,4.0:20,2.5:28",
        help=(
            "Comma-separated `cfg:steps` list. First entry is the baseline "
            "everything else is compared against (default 4.0:28)."
        ),
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default="832x1248,1152x896",
        help="Pixel-bucket allowlist (HxW) — keeps the static-token assumption.",
    )
    parser.add_argument(
        "--n_per_bucket",
        type=int,
        default=6,
        help="Stems per bucket (total N = n_per_bucket * len(buckets)).",
    )
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=1.0,
        help="Anima production flow_shift (configs/base.toml).",
    )
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile of the DiT block stack.",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=0,
        help="TE cache variant index to use as positive prompt.",
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Dump each synthesized latent to the run dir (debugging only — big).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional run-dir label suffix (defaults to len(configs)x_<N>).",
    )
    args = parser.parse_args()

    configs = _parse_configs(args.configs)
    buckets = _parse_buckets(args.buckets)
    cache_dir = Path(args.cache_dir)
    uncond_path = cache_dir / UNCOND_TE_FILENAME
    if not uncond_path.exists():
        raise SystemExit(
            f"Missing uncond sidecar at {uncond_path}. Run `make distill-prep "
            f"--skip_synth` first."
        )

    from library.anima import weights as anima_utils
    from library.anima.models import Anima
    from library.io.cache import (
        discover_cached_pairs,
        get_latent_resolution,
        load_cached_latents,
        load_cached_text_features,
    )

    # ── pick samples ──────────────────────────────────────────────────────
    all_pairs = discover_cached_pairs(str(cache_dir))
    by_bucket: dict[tuple[int, int], list] = {}
    for p in all_pairs:
        try:
            res = get_latent_resolution(p.npz_path)  # "HxW" in latent units
            H_lat, W_lat = (int(x) for x in res.split("x"))
        except Exception:
            continue
        key = (H_lat * 8, W_lat * 8)
        if key not in {(h, w) for h, w in buckets}:
            continue
        by_bucket.setdefault(key, []).append(p)

    rng = np.random.default_rng(int(args.shuffle_seed))
    selected: list = []
    for key in sorted(by_bucket):
        items = by_bucket[key]
        rng.shuffle(items)
        items = items[: args.n_per_bucket]
        h_pix, w_pix = key
        logger.info(f"bucket {h_pix}x{w_pix}: {len(items)} pair(s)")
        selected.extend(items)
    if not selected:
        raise SystemExit(
            f"No cached pairs match buckets={buckets} in {cache_dir}. "
            f"Either widen --buckets or check the cache."
        )
    N = len(selected)
    logger.info(f"selected {N} stems across {len(by_bucket)} bucket(s)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ── load model (once) ─────────────────────────────────────────────────
    logger.info(f"loading DiT teacher: {args.dit}")
    model: Anima = anima_utils.load_anima_model(
        device,
        args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    if args.blocks_to_swap > 0:
        model.enable_block_swap(args.blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
    else:
        model.to(device)
    model.set_static_token_count(4096)
    model.eval()

    if not args.no_compile and args.blocks_to_swap == 0:
        try:
            model.compile_core(mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"compile_core failed ({e}); falling back to eager.")

    # ── load uncond sidecar ───────────────────────────────────────────────
    sd_uncond = _load_safetensors(str(uncond_path))
    crossattn_neg = (
        sd_uncond["crossattn_emb"].to(device=device, dtype=dtype).unsqueeze(0).contiguous()
    )

    # ── per-stem inner work ───────────────────────────────────────────────
    # Storage: latents[config_idx] = list of (C, H_lat, W_lat) float32 CPU
    cfg_latents: list[list[torch.Tensor]] = [[] for _ in configs]
    cfg_real_latents: list[torch.Tensor] = []  # real cached latent per stem
    cfg_timings: list[list[float]] = [[] for _ in configs]
    stems: list[str] = []
    shapes: list[tuple[int, int]] = []  # (H_lat, W_lat) per stem

    label = args.label or f"{len(configs)}cfg-N{N}"
    run_dir = make_run_dir("distill_mod", label=label)
    latents_dir = run_dir / "latents"
    if args.save_latents:
        latents_dir.mkdir(exist_ok=True)

    for sample_idx, pair in enumerate(selected):
        try:
            res_str = get_latent_resolution(pair.npz_path)
            H_lat, W_lat = (int(x) for x in res_str.split("x"))
        except Exception as e:
            logger.warning(f"skip {pair.stem}: bad latent NPZ ({e})")
            continue

        crossattn_pos, _ = load_cached_text_features(pair.te_path, variant=args.variant)
        if crossattn_pos is None:
            logger.warning(f"skip {pair.stem}: no crossattn_emb")
            continue
        crossattn_pos = crossattn_pos.to(device=device, dtype=dtype).unsqueeze(0)

        # mirror synth.py per-sample seed convention
        per_seed = (int(args.seed) * 1_000_003 + sample_idx) & 0x7FFFFFFF

        stems.append(pair.stem)
        shapes.append((H_lat, W_lat))

        real_lat, _, _, _ = load_cached_latents(pair.npz_path)  # (C, H, W)
        cfg_real_latents.append(real_lat.float())

        for ci, (cfg_scale, num_steps) in enumerate(configs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            clean = denoise_one(
                model,
                crossattn_pos,
                crossattn_neg,
                H_lat=H_lat,
                W_lat=W_lat,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                flow_shift=args.flow_shift,
                seed=per_seed,
                device=device,
                dtype=dtype,
            )  # (1, 16, H_lat, W_lat) float32 CPU
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            cfg_timings[ci].append(elapsed)

            lat = clean.squeeze(0)  # (16, H_lat, W_lat)
            cfg_latents[ci].append(lat)
            if args.save_latents:
                np.savez(
                    latents_dir
                    / f"{pair.stem}_{H_lat}x{W_lat}_cfg{cfg_scale}_s{num_steps}.npz",
                    latents=lat.numpy(),
                )

        logger.info(
            f"[{sample_idx + 1}/{N}] {pair.stem} {H_lat}x{W_lat}  "
            + "  ".join(
                f"cfg{c}/s{s}={cfg_timings[i][-1]:.2f}s"
                for i, (c, s) in enumerate(configs)
            )
        )

    # ── per-sample pair metrics (baseline=configs[0]) ─────────────────────
    per_sample_rows: list[dict[str, object]] = []
    for i, stem in enumerate(stems):
        row: dict[str, object] = {"stem": stem, "H_lat": shapes[i][0], "W_lat": shapes[i][1]}
        baseline = cfg_latents[0][i]
        for ci, (cfg_scale, num_steps) in enumerate(configs):
            tag = f"cfg{cfg_scale}_s{num_steps}"
            if ci == 0:
                continue
            m = _pair_metrics(baseline, cfg_latents[ci][i])
            row[f"mse_vs_baseline__{tag}"] = m["mse"]
            row[f"cos_vs_baseline__{tag}"] = m["cos"]
        per_sample_rows.append(row)

    # write per_sample.csv
    if per_sample_rows:
        keys = list(per_sample_rows[0].keys())
        with open(run_dir / "per_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_sample_rows)

    # ── per-config distribution stats ─────────────────────────────────────
    # Cat per-config latents across stems: (N*H*W, C) per channel stats.
    def _stack_for_stats(latents: list[torch.Tensor]) -> torch.Tensor:
        # latents have varying H_lat/W_lat — flatten each to (C, -1), cat on dim=1.
        flat = [lat.reshape(lat.shape[0], -1) for lat in latents]
        return torch.cat(flat, dim=1)  # (C, total_elems)

    real_flat = _stack_for_stats(cfg_real_latents)
    real_mean = real_flat.mean(dim=1).cpu().numpy()
    real_std = real_flat.std(dim=1).cpu().numpy()

    per_config_rows: list[dict[str, object]] = []
    for ci, (cfg_scale, num_steps) in enumerate(configs):
        flat = _stack_for_stats(cfg_latents[ci])
        ch_mean = flat.mean(dim=1).cpu().numpy()
        ch_std = flat.std(dim=1).cpu().numpy()
        mean_drift_vs_real = float(np.max(np.abs(ch_mean - real_mean)))
        std_drift_vs_real = float(np.max(np.abs(ch_std - real_std)))
        timings = np.asarray(cfg_timings[ci])
        per_config_rows.append(
            {
                "cfg_scale": cfg_scale,
                "num_steps": num_steps,
                "mean_time_s": float(timings.mean()),
                "p50_time_s": float(np.median(timings)),
                "max_abs_chmean_drift_vs_real": mean_drift_vs_real,
                "max_abs_chstd_drift_vs_real": std_drift_vs_real,
                "global_mean": float(ch_mean.mean()),
                "global_std": float(ch_std.mean()),
            }
        )

    with open(run_dir / "per_config.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(per_config_rows[0].keys()))
        wri.writeheader()
        wri.writerows(per_config_rows)

    # ── aggregate metrics for the envelope ────────────────────────────────
    aggregate: dict[str, object] = {
        "n_samples": N,
        "buckets": [list(b) for b in by_bucket.keys()],
        "real_channel_mean_global": float(real_mean.mean()),
        "real_channel_std_global": float(real_std.mean()),
        "per_config": per_config_rows,
    }
    pairs_summary: list[dict[str, object]] = []
    base_cfg, base_steps = configs[0]
    for ci, (cfg_scale, num_steps) in enumerate(configs):
        if ci == 0:
            continue
        mses = [
            float(r[f"mse_vs_baseline__cfg{cfg_scale}_s{num_steps}"])
            for r in per_sample_rows
        ]
        coss = [
            float(r[f"cos_vs_baseline__cfg{cfg_scale}_s{num_steps}"])
            for r in per_sample_rows
        ]
        pairs_summary.append(
            {
                "baseline": f"cfg{base_cfg}_s{base_steps}",
                "compare": f"cfg{cfg_scale}_s{num_steps}",
                "mean_mse": float(np.mean(mses)),
                "p50_mse": float(np.median(mses)),
                "mean_cos": float(np.mean(coss)),
                "min_cos": float(np.min(coss)),
            }
        )
    aggregate["pair_summary"] = pairs_summary

    artifacts = ["per_sample.csv", "per_config.csv"]
    if args.save_latents:
        artifacts.append("latents/")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=aggregate,
        label=label,
        artifacts=artifacts,
        device=device,
    )
    logger.info(f"wrote results → {run_dir}")
    for row in pairs_summary:
        logger.info(
            f"  {row['baseline']} vs {row['compare']}: "
            f"mean_mse={row['mean_mse']:.4f}  mean_cos={row['mean_cos']:.4f}  "
            f"min_cos={row['min_cos']:.4f}"
        )


if __name__ == "__main__":
    main()
