# HF-residual Tier 1 diagnostic

Cheap, no-training kill-switch for the structural LF/HF velocity split
proposed in `docs/proposal/hf_residual_adapter.md`. Two questions:

1. **Headroom (`gap_ratio`)** — How much FM loss does the *bare base* leave
   on the table when fed the LF half of `x_t` instead of the full `x_t`?
   This is the slack an HF adapter could in principle close.
2. **Signal budget (`xt_hband_frac`, `adapter_target_hband_frac`)** — How
   much HF mass is actually present in `x_t` and in the residual the
   adapter would be trained against?

If headroom is dead, the proposal is dead — the LF-input base already
predicts everything and there's nothing for the adapter to do. If
headroom is huge, the LF-input base has been crippled — the adapter has
to relearn most of the model, defeating the "small adapter" pitch. We
want a Goldilocks band.

## What it measures

For each `(sample, t, ε)` and each `σ_low`:

```
x_t       = α_t · x_0 + σ_t · ε
x_t^L     = blur(x_t, σ_low)          # inference-style split
x_t^H     = x_t − x_t^L
u_target  = ε − x_0

u_pred_full = base(x_t,   t, te)
u_pred_lf   = base(x_t^L, t, te)

loss_full = ‖u_pred_full − u_target‖²
loss_lf   = ‖u_pred_lf   − u_target‖²
gap       = loss_lf − loss_full
gap_ratio = gap / loss_full           # ← headline
```

Plus the band decomposition of all four quantities using the same blur
kernel, so the report tells you which band the loss lives in and what
the adapter's training target (`u_target − u_pred_lf`) looks like.

`gap_ratio` is the key number: **the fraction of FM loss that an HF
adapter could possibly recover** if it were perfect at predicting
`u_target − base(x_t^L)`.

## Verdict thresholds (pivot div = 4.0)

Heuristic, on the σ_low = D/4 row (the doc's default inherited from FEI):

| gap_ratio median | xt_hband_frac median | verdict |
|---|---|---|
| ≥ 0.10 | ≥ 0.05 | **HEADROOM** — proceed to Tier 2 |
| 0.03 – 0.10 | any | **MARGINAL** — worth a check at div=8 (less aggressive blur) |
| < 0.03 | any | **DEAD** — shelve the proposal |

The right edge of "huge" isn't gated here — eyeball `gap_ratio` >> 1.0
manually; it means the LF-only base is so degraded the adapter has to
do most of the model's job, which defeats the "small adapter" pitch.

## Running

```bash
cd anima_lora
uv run python bench/hf_residual/run_bench.py \
  --num_samples 6 --num_timesteps 6 --num_noise 16 \
  --sigma_low_divs 2,4,8 \
  --label tier1-baseline
```

Defaults: K=6 × T=6 × N=16 × 3 σ_low divisors on the most-populous
latent bucket from `post_image_dataset/lora/`. The base DiT defaults to
`models/diffusion_models/anima-preview3-base.safetensors`. Compute is
~150 batched forwards at the bucket's resolution — well under an hour
on a single GPU.

## Outputs

- `result.json` — standard `bench/_common.py` envelope plus the summary
  (`metrics` field), including the verdict.
- `summary.json` — same summary as `metrics`, also written standalone.
- `per_sample_t.csv` — one row per (sample, t, σ_low_div) for plotting
  or post-hoc filtering.

## Caveats

- **Inference-style band split only.** `x_t^L = blur(x_t)`, *not* the
  training-style `α_t · blur(x_0) + σ_t · ε`. The proposal uses the
  inference form in the live loop, so that's what we measure. The two
  forms differ by `σ_t · (ε − blur(ε))`; that residual is small at low
  σ_t but non-trivial at high σ_t.
- **One bucket.** Aspect-invariance of `σ_low = D/div` is already
  validated by `bench/fera/probe_fei_dataset.py`. If we want to confirm
  on a non-square bucket later, pass `--bucket`.
- **Headline frozen at base.** No LoRA loaded. If the proposal is
  framed as "base+HF-adapter vs base+LoRA", a future variant of this
  bench should add a third arm `base + frozen LoRA on x_t^L` — but
  Tier 1 is about whether the *structural* split has any headroom at
  all.

## Next steps if verdict ≥ MARGINAL

Proceed to Tier 2 (matched-compute A/B between LoRA r=8, HF-residual
without `ctx_L`, HF-residual with `ctx_L`) — see
`docs/proposal/hf_residual_adapter.md` §"Validation gates" Phase 2.
