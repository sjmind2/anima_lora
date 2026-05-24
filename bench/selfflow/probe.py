#!/usr/bin/env python
"""Self-Flow feasibility probe on Anima's frozen DiT.

See README.md. Self-Flow (arXiv 2603.06507) strengthens a flow model's
*internal* representations during training via Dual-Timestep Scheduling + an
EMA-teacher cosine alignment loss. Porting it to our **frozen-DiT LoRA**
regime only makes sense if two necessary conditions hold on the frozen
backbone + our dataset:

  1. INFORMATION ASYMMETRY  — a cleaner-noised view yields layer-k features a
     noisier view does not (`asym_k` well below 1). This is the premise DTS
     relies on; if the frozen DiT has already collapsed the noise-level gap by
     layer k, there is nothing to infer and the method is dead.

  2. NON-TRIVIAL ALIGNMENT TARGET — an MLP head `h` mapping the noisier view's
     layer-l feature to the cleaner view's layer-k feature cannot *already*
     saturate to cos≈1 with the backbone frozen. If it can, the rep loss
     exerts no pressure on the LoRA adapter and the method collapses to
     DTS-only.

Construction (per latent, per sampled timestep pair): draw two timesteps, sort
to `(τ_lo, τ_hi)`, share one noise sample ε, and build two in-distribution
views — teacher `x_{τ_lo}` (cleaner) and student `x_{τ_hi}` (noisier) — each at
its own conditioning timestep. Tap the residual-stream output of block l and
block k via forward hooks. Frozen base only; no adapter.

This probe tests NECESSARY, not sufficient, conditions. A pass means "worth a
training A/B" (see README); a fail kills the idea cheaply.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# bench/ is not an installed package — bootstrap the repo root onto sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

log = logging.getLogger("selfflow_probe")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit", required=True, help="Base DiT safetensors (frozen).")
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument(
        "--bucket",
        default=None,
        help="Latent-dim bucket WxH filter. Default: most common bucket in --data_dir.",
    )
    p.add_argument("--num_samples", type=int, default=12)
    p.add_argument(
        "--num_timesteps",
        type=int,
        default=8,
        help="Number of (τ_lo, τ_hi) timestep pairs sampled per latent.",
    )
    p.add_argument(
        "--layer_l", type=int, default=6, help="Early (student) block index."
    )
    p.add_argument(
        "--layer_k", type=int, default=18, help="Late (teacher) block index."
    )
    p.add_argument("--t_min", type=float, default=0.10)
    p.add_argument("--t_max", type=float, default=0.90)
    p.add_argument(
        "--token_cap",
        type=int,
        default=512,
        help="Tokens kept per (latent, pair) for the headroom MLP (memory bound).",
    )
    p.add_argument("--headroom_epochs", type=int, default=600)
    p.add_argument("--headroom_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def discover_samples(data_dir: Path, bucket: str | None, num_samples: int, seed: int):
    """Return (bucket, [(stem, latent_key, npz_path, te_path), ...]).

    Mirrors bench/fm_vr_headroom: npz is `{stem}_{Wpix}x{Hpix}_anima.npz`, TE
    sidecar `{stem}_anima_te.safetensors`.
    """
    res_re = re.compile(r"_(\d{3,5})x(\d{3,5})_anima\.npz$")
    # Caches live in per-artist subdirs (post_image_dataset/lora/<artist>/), so
    # glob recursively and resolve each TE sidecar next to its own npz.
    npz_paths = sorted(glob.glob(str(data_dir / "**" / "*_anima.npz"), recursive=True))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` under {data_dir} (searched recursively)")
    by_bucket: dict[str, list[tuple[str, str, str, str]]] = {}
    for p in npz_paths:
        name = Path(p).name
        m = res_re.search(name)
        if not m:
            continue
        stem = name[: m.start()]
        te = Path(p).parent / f"{stem}_anima_te.safetensors"
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
            f"bucket {chosen!r} not found. Available: "
            f"{sorted((k, len(v)) for k, v in by_bucket.items())[:10]}"
        )
    pool = by_bucket[chosen]
    if len(pool) < num_samples:
        raise SystemExit(
            f"bucket {chosen!r} has {len(pool)} paired samples; need {num_samples}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=num_samples, replace=False)
    return chosen, [pool[i] for i in idx]


def load_pair(npz_path: str, latent_key: str, te_path: str, device, dtype):
    z = np.load(npz_path)
    x0 = torch.from_numpy(z[latent_key]).to(device=device, dtype=dtype)  # (C, H, W)
    z.close()
    crossattn = load_file(te_path)["crossattn_emb_v0"].to(device=device, dtype=dtype)
    return x0, crossattn


@torch.inference_mode()
def block_features(anima, captures, x0_4d, t_scalar, crossattn, sigma, eps):
    """Noise x0 at level `sigma` with shared `eps`, run the frozen DiT at
    conditioning `t_scalar`, and return the captured (block_l, block_k)
    residual-stream features flattened to (tokens, D).

    x_t = (1 − σ)·x0 + σ·ε   (rectified-flow interpolation; σ == t here).
    """
    x_t = (1.0 - sigma) * x0_4d + sigma * eps  # (1, C, H, W)
    B, C, H, W = x_t.shape
    x_5d = x_t.unsqueeze(2)  # (1, C, 1, H, W)
    # The DiT forward expects timesteps in [0, 1] — both training
    # (samplers.py: `timesteps / 1000`) and inference (generation.py: `timesteps
    # /= 1000`) feed it that scale. (The `*1000` in get_timesteps_sigmas is
    # undone at every call site.) σ == t here, so condition at t_scalar directly.
    timesteps = torch.full((B,), t_scalar, dtype=x_t.dtype, device=x_t.device)
    padding_mask = torch.zeros(B, 1, H, W, dtype=x_t.dtype, device=x_t.device)
    captures.clear()
    anima(x_5d, timesteps, crossattn, padding_mask=padding_mask)
    # Block output is (B, T, H, W, D); D is the last dim. Flatten to (tokens, D).
    f_l = captures["l"].reshape(-1, captures["l"].shape[-1]).float()
    f_k = captures["k"].reshape(-1, captures["k"].shape[-1]).float()
    return f_l, f_k


def median_token_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).median().item())


class Head(torch.nn.Module):
    """The paper's h = MLP(features): D → D → D, GELU between."""

    def __init__(self, d: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, d)
        )

    def forward(self, x):
        return self.net(x)


def train_headroom(
    x: torch.Tensor,
    y: torch.Tensor,
    groups: np.ndarray,
    *,
    epochs: int,
    lr: float,
    device,
    seed: int,
) -> dict:
    """Fit h to maximize cos(h(x), y); report eval cosine on an IMAGE-level split.

    x = student layer-l features, y = teacher layer-k features (paired tokens).
    ``groups`` is the source-image id per token — the eval set holds out whole
    images, so the ceiling measures cross-image generalization rather than
    memorized per-image structure (a token-level split leaks and inflates it).
    The ceiling reached with the BACKBONE FROZEN tells us whether the rep loss
    target is trivially satisfiable by the head alone (→ no adapter pressure).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_eval_groups = max(1, len(uniq) // 5)
    eval_groups = set(uniq[:n_eval_groups].tolist())
    ev_mask = np.array([g in eval_groups for g in groups])
    ev = torch.from_numpy(np.nonzero(ev_mask)[0])
    tr = torch.from_numpy(np.nonzero(~ev_mask)[0])
    d = x.shape[-1]
    head = Head(d).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    x_tr, y_tr = x[tr].to(device), y[tr].to(device)
    x_ev, y_ev = x[ev].to(device), y[ev].to(device)

    raw_floor = median_token_cos(x_ev, y_ev)  # no-head baseline (l vs k directly)
    bsz = min(8192, x_tr.shape[0])
    for ep in range(epochs):
        idx = torch.randint(0, x_tr.shape[0], (bsz,), device=device)
        loss = (1.0 - F.cosine_similarity(head(x_tr[idx]), y_tr[idx], dim=-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        eval_cos = median_token_cos(head(x_ev), y_ev)
        train_cos = median_token_cos(head(x_tr[:bsz]), y_tr[:bsz])
    return {
        "headroom_eval_cos": eval_cos,
        "headroom_train_cos": train_cos,
        "raw_cos_floor": raw_floor,
        "headroom_lift_vs_raw": eval_cos - raw_floor,
        "n_tokens": int(x.shape[0]),
        "n_eval_images": int(n_eval_groups),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = make_run_dir("selfflow", label=args.label)
    log.info(f"output → {run_dir}")

    if args.layer_l >= args.layer_k:
        raise SystemExit(f"need layer_l < layer_k (got {args.layer_l}, {args.layer_k})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    data_dir = Path(args.data_dir)
    bucket, samples = discover_samples(
        data_dir, args.bucket, args.num_samples, args.seed
    )
    log.info(f"bucket={bucket} num_samples={len(samples)}")

    log.info(f"loading frozen DiT: {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).eval().requires_grad_(False)
    anima.reset_mod_guidance()

    n_blocks = len(anima.blocks)
    if args.layer_k >= n_blocks:
        raise SystemExit(f"layer_k {args.layer_k} >= n_blocks {n_blocks}")

    # Hooks on the residual-stream output of block l and block k (NOT compiled —
    # submodule hooks must fire eagerly). Matches the functional-loss tap.
    captures: dict[str, torch.Tensor] = {}

    def mk_hook(key):
        def hook(_m, _i, out):
            captures[key] = out.detach()

        return hook

    h_l = anima.blocks[args.layer_l].register_forward_hook(mk_hook("l"))
    h_k = anima.blocks[args.layer_k].register_forward_hook(mk_hook("k"))

    rng = torch.Generator(device=device).manual_seed(args.seed)
    tcap = args.token_cap

    rows: list[dict] = []
    # Stored as numpy fp32 — the captures are inference tensors, and a numpy
    # round-trip is the clean way to feed them into the autograd-tracked
    # headroom MLP later without "inference tensors saved for backward" errors.
    # fp32, NOT fp16: late-block (e.g. k=18) residual-stream values overflow
    # fp16's 65504 ceiling → inf → nan cosine. bf16 holds them; keep that range.
    fl_student_all, fk_teacher_all = [], []
    token_groups: list[np.ndarray] = []  # source-image id per token (image-level split)
    total = len(samples) * args.num_timesteps
    pbar = tqdm(total=total, desc="selfflow-probe", dynamic_ncols=True)
    try:
        for si, (stem, latent_key, npz_path, te_path) in enumerate(samples):
            x0, crossattn = load_pair(npz_path, latent_key, te_path, device, dtype)
            x0_4d = x0.unsqueeze(0)  # (1, C, H, W)
            ca = crossattn.unsqueeze(0)  # (1, S, D_text)
            eps_base = torch.randn(
                x0_4d.shape, generator=rng, device=device, dtype=dtype
            )

            for _ in range(args.num_timesteps):
                two = (
                    torch.rand(2, generator=rng, device=device)
                    * (args.t_max - args.t_min)
                    + args.t_min
                )
                t_lo, t_hi = float(two.min()), float(two.max())

                # Cleaner (teacher) and noisier (student) views, shared ε.
                fl_tea, fk_tea = block_features(
                    anima, captures, x0_4d, t_lo, ca, t_lo, eps_base
                )
                fl_stu, fk_stu = block_features(
                    anima, captures, x0_4d, t_hi, ca, t_hi, eps_base
                )

                # Metric 1: info asymmetry — does the clean view's layer-k
                # feature differ from the noisy view's? (≈1 ⇒ nothing to infer)
                asym_k = median_token_cos(fk_stu, fk_tea)
                asym_l = median_token_cos(fl_stu, fl_tea)

                # Collect tokens for the headroom MLP: map student-l → teacher-k.
                ntok = fl_stu.shape[0]
                sel = torch.randperm(ntok, device=device)[: min(tcap, ntok)]
                fl_student_all.append(fl_stu[sel].float().cpu().numpy())
                fk_teacher_all.append(fk_tea[sel].float().cpu().numpy())
                token_groups.append(np.full(len(sel), si, dtype=np.int32))

                rows.append(
                    {
                        "stem": stem,
                        "t_lo": round(t_lo, 4),
                        "t_hi": round(t_hi, 4),
                        "asym_k": asym_k,
                        "asym_l": asym_l,
                    }
                )
                pbar.set_postfix(
                    {"t": f"{t_lo:.2f}/{t_hi:.2f}", "asym_k": f"{asym_k:.3f}"}
                )
                pbar.update(1)
            del x0, x0_4d, ca, eps_base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        h_l.remove()
        h_k.remove()
        pbar.close()

    # Headroom: fit h on all collected (student-l, teacher-k) token pairs.
    # from_numpy yields normal (non-inference) tensors usable by autograd.
    x = torch.from_numpy(np.concatenate(fl_student_all, axis=0)).float()
    y = torch.from_numpy(np.concatenate(fk_teacher_all, axis=0)).float()
    groups = np.concatenate(token_groups, axis=0)
    log.info(f"headroom MLP: fitting on {x.shape[0]} tokens (D={x.shape[1]})")
    headroom = train_headroom(
        x,
        y,
        groups,
        epochs=args.headroom_epochs,
        lr=args.headroom_lr,
        device=device,
        seed=args.seed,
    )

    # Per-pair CSV.
    csv_path = run_dir / "per_pair.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    asym_k_arr = np.array([r["asym_k"] for r in rows], dtype=np.float64)
    asym_l_arr = np.array([r["asym_l"] for r in rows], dtype=np.float64)
    t_hi_arr = np.array([r["t_hi"] for r in rows], dtype=np.float64)
    t_lo_arr = np.array([r["t_lo"] for r in rows], dtype=np.float64)
    asym_k_med = float(np.median(asym_k_arr))
    headroom_cos = headroom["headroom_eval_cos"]

    # Favorable regime — wide noise gap at high σ, where the paper's gains
    # concentrate. If asymmetry exists anywhere on a frozen backbone, it's here.
    fav = (t_hi_arr >= 0.7) & ((t_hi_arr - t_lo_arr) >= 0.4)
    asym_k_fav_med = float(np.median(asym_k_arr[fav])) if fav.any() else float("nan")

    # Verdict — asymmetry is the make-or-break premise; headroom is the
    # adapter-pressure check. Thresholds are heuristic (see README) and meant
    # to be recalibrated once we have a baseline run.
    if asym_k_med >= 0.97:
        verdict = "NO-ASYMMETRY"
    elif headroom_cos >= 0.95:
        verdict = "TRIVIAL"
    elif 0.30 <= headroom_cos < 0.95 and headroom["headroom_lift_vs_raw"] >= 0.05:
        verdict = "VIABLE"
    else:
        verdict = "MARGINAL"

    metrics = {
        "bucket": bucket,
        "n_pairs": len(rows),
        "layer_l": args.layer_l,
        "layer_k": args.layer_k,
        "n_blocks": n_blocks,
        "asym_k_median": asym_k_med,
        "asym_k_mean": float(asym_k_arr.mean()),
        "asym_k_favorable_median": asym_k_fav_med,
        "asym_k_favorable_n": int(fav.sum()),
        "asym_l_median": float(np.median(asym_l_arr)),
        **headroom,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=["per_pair.csv"],
        label=args.label,
        device=device,
    )

    log.info(f"[bench] verdict: {verdict}")
    log.info(f"[bench]   asym_k median       = {asym_k_med:.3f}  (≈1 ⇒ no asymmetry)")
    log.info(
        f"[bench]   asym_k favorable    = {asym_k_fav_med:.3f}  "
        f"(high-σ wide-gap, n={int(fav.sum())})"
    )
    log.info(
        f"[bench]   headroom eval cos   = {headroom_cos:.3f}  "
        f"(raw floor {headroom['raw_cos_floor']:.3f}, "
        f"lift {headroom['headroom_lift_vs_raw']:+.3f}, "
        f"{headroom['n_eval_images']} held-out imgs)"
    )


if __name__ == "__main__":
    main()
