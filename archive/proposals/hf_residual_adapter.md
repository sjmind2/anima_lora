# HF-Residual Adapter

> **Status: shelved 2026-05-14.** The Phase 0 gate this proposal defined for
> itself ([Validation gates](#validation-gates) → "Fail mode: if λ drifts,
> e.g. settles at −0.7, … the whole proposal is dead") failed against the
> first live training run with the just-shipped `vr/lambda_ema` logging
> (`output/logs/lora_default_20260514-1921/`): `vr/lambda_ema` started at
> −0.89 and walked to −0.72 by step ~586, mean ≈ −0.75 — nowhere near the
> bench's `−0.996 ± 0.002`. That bench (`bench/fm_vr_headroom/results/
> 20260514-1300-tlora-vs-base/`) was measured against a **merged T-LoRA**
> trainable + base frozen reference, where `u_pred ≈ u_pred^L` is true by
> construction; the `λ → −1` reading was a setup artifact, not an Anima-data
> property. Independent corroboration: the Tier 1 HF-residual diagnostic
> (`archive/bench/hf_residual/results/20260514-1851-tier1-baseline/`,
> [[project_hf_residual_tier1]]) showed gap_ratio 0.045 overall and ~80% of
> signal concentrated at t > 0.7, so the "replaces LoRA" framing was already
> weak before Phase 0 failed.
>
> Retained as a historical record of the LF/HF structural-split reasoning;
> not a live work item. If revisited, reframe as high-t gated HF refinement
> stacked *on top of* LoRA (per the Tier 1 memo), not as a LoRA replacement.

---

**Status:** proposal. Gated behind a TensorBoard-observable precondition (see
[Validation gates](#validation-gates)).

One-line: replace the weight-additive LoRA + VR-loss pair with a structural
**LF/HF velocity split** — the frozen base predicts the FEI-low-pass velocity,
a small dedicated adapter predicts the high-frequency residual velocity from
`x_t^H` directly. Trains without a no-grad reference forward, formalizes what
the λ → −1 collapse of VR loss already told us, and frees the adapter from
the base's full-band featurization.

## Motivation

`docs/experimental/vr_loss.md` derives that the EMA λ converges to ≈ −0.996
on Anima, at which point the AsymFlow §5.2 loss collapses to:

```
||y − z||²  =  ||Δu_adapter + x_0^H||²
```

where `Δu_adapter = u_pred − u_pred^L` and `x_0^H = x_0 − x_0^L` is the FEI
high-frequency band. Read literally: **the adapter's effective delta against
the base's bypass prediction is being trained to match the high-frequency
velocity residual.**

The current implementation recovers `Δu_adapter` by *subtraction* — one
gradient forward on `x_t` plus one no-grad forward on `x_t^L`, ~+40% step
cost (`vr_loss.md:286-298`). The subtraction is bookkeeping for a quantity
the loss could supervise directly if the adapter had its own input pathway.

This proposal makes the decomposition structural: train a small adapter
that *consumes* `x_t^H` and *outputs* the HF velocity directly. The base's
LF forward becomes the actual computation, not an auxiliary control variate.

## The transform

Standard FM target factorizes by linearity:

```
u_target = ε − x_0
         = (ε − x_0^L) + (x_0^L − x_0)
         = u_target^L  +  u_target^H
```

with `u_target^H = −x_0^H`. At inference, the latent state splits cheaply:

```
x_t^L = gaussian_blur_2d(x_t, σ_low)
x_t^H = x_t − x_t^L
```

(`library/runtime/fei.py::gaussian_blur_2d`, the same kernel used by the
FEI router and the VR loss).

## Proposed setup

### Training

```
x_0^L      = blur(x_0, σ_low)                  # σ_low = min(H,W) / vr_fei_sigma_low_div
x_t^L      = (1−σ_t)·x_0^L + σ_t·ε
x_t^H      = x_t − x_t^L

u_pred^L   = base(x_t^L, t, te)                # frozen base, no_grad
u_pred^H   = adapter(x_t^H, t, te, ctx_L)      # small grad-trained branch
u_pred     = u_pred^L + u_pred^H

loss       = ||u_pred − (ε − x_0)||²           # plain FM, no λ, no z
```

`ctx_L` is a cheap LF context summary (see [Architecture](#architecture)
below) — without it the HF branch loses spatial grounding.

Step cost (per the cost table in `vr_loss.md`):

| | grad fwd (base) | grad fwd (adapter) | no-grad fwd (base on LF) | bwd | net |
|---|---|---|---|---|---|
| Standard FM            | 1 |  – | 0 | 1 | 1× (baseline) |
| VR loss (current)      | 1 |  – | 1 | 1 | ~1.4× |
| HF-residual (proposed) | 0 |  1 | 1 | 1 (adapter only) | **< 1×** |

The no-grad base forward on `x_t^L` replaces the gradient-tracked base
forward of standard FM. There is no separate "reference" pass — the base
is doing its actual job (LF prediction) and is no-grad because it is
frozen. The only gradient path is through the small HF adapter.

### Inference

```
for σ_t in schedule:
    x_t^L = blur(x_t, σ_low)
    x_t^H = x_t − x_t^L
    u_total = base(x_t^L, t, te) + adapter(x_t^H, t, te, ctx_L)
    x_t = euler_step(x_t, u_total, σ_t, σ_{t+1})
```

Same step count as a vanilla Anima run. Per-step cost is base-on-LF plus
adapter — adapter is small, so cost is dominated by the base. Comparable
to (slightly above) standard inference; comparable to (below) Spectrum
inference if adapter is also Chebyshev-cached.

### Architecture

The HF branch is opinionated by what HF actually needs:

- **Input:** patches of `x_t^H` at the native latent resolution. Same patch
  embedding as the DiT (or a fresh thin one — `x_t^H` has much smaller
  dynamic range than `x_t`, so a small embedder is enough).
- **Backbone:** ~4-6 DiT-style blocks at reduced dim (target ~50-100M
  params, vs the base's ~2B). HF features are predominantly local — window
  attention or a thin conv stem is a credible alternative; pick after a
  small ablation, not upfront.
- **Conditioning:** the same text cross-attention as the base + AdaLN
  from `t`. Same as a normal Anima block.
- **LF context (`ctx_L`):** an average-pooled summary of an early base
  block's features on `x_t^L`. Captured via a forward hook (same machinery
  REPA capture already uses, `vr_loss.md:256-258`), spliced into the HF
  blocks' AdaLN. This keeps the HF branch spatially grounded without
  forcing it through the full-band representation.

The HF backbone is a separate module on disk — not a LoRA. There is no
weight-baking path for it; see [What it gives up](#what-it-gives-up).

## What it buys

1. **Step cost drops below baseline FM.** The +40% no-grad forward becomes
   the *only* base forward and replaces the gradient-tracked one. The
   adapter forward is cheap. This is the only quantitatively certain win.
2. **Narrower input, smaller model.** `x_t^H` has materially smaller
   dynamic range than `x_t` (LF band carries most of the variance). A
   dedicated branch can run at lower capacity and still match what the
   weight-additive LoRA does today.
3. **Adapter is free to pick its own featurization.** No constraint to
   express HF refinement through representations tuned for full-band
   prediction. Window attention, conv stems, smaller blocks all become
   credible.
4. **Standard FM loss.** No λ, no EMA, no cov/var bookkeeping, no aux
   dict, no semi-gradient caveats with postfix. The VR machinery
   disappears.
5. **Composable with Spectrum.** The HF branch is a separate forward —
   it gets its own Chebyshev forecaster, decoupled from the base's
   prediction cadence. Could schedule HF refinements at every step while
   skipping LF base forwards on cached steps.

## What it gives up

1. **Merge-to-DiT story breaks.** `scripts/merge_to_dit.py` bakes a LoRA
   into base weights for vanilla ComfyUI compatibility. A structurally
   separate HF backbone cannot be baked — it lives as its own checkpoint
   forever and requires a custom ComfyUI node to consume. Same situation
   as IP-Adapter / EasyControl today.
2. **Lossy if the HF branch starves on context.** Pure HF input doesn't
   tell the branch *where* a sharp edge belongs. `ctx_L` mitigates but
   may not be enough; needs to be measured.
3. **Adapter ecosystem rewrites.** OrthoLoRA / T-LoRA / HydraLoRA / FeRA
   / ReFT all assume weight-additive injection on the base. None of
   them port to this architecture directly. Either the HF backbone
   ships as a standalone method (parallel to IP-Adapter), or the LoRA
   family stays as-is and HF-residual is offered as a separate option.
4. **No measured quality win.** ρ² = 0.9999 says the *loss-level*
   reduction is already maximal in the current weight-additive setup.
   Whether the structural split improves optimizer dynamics or sample
   quality is unknown until Phase 2.
5. **DCW / mod-guidance / IP-Adapter interactions need re-derivation.**
   Each sampler-level correction assumes the model outputs a single
   `u_pred`. They mostly still hold if `u_pred = u_pred^L + u_pred^H`
   is the externally visible velocity, but each one needs an explicit
   check rather than inheritance from the weight-additive setup.

## Validation gates

Gate the proposal on **observable training-time signal** before paying
the refactor cost.

### Phase 0 — λ stability on real data (live; just-shipped TB logging)

The newly-added `vr/lambda_ema` and `vr/lambda_batch` curves in
TensorBoard (`train.py:151-158`) make this directly observable on any
VR-enabled run.

**Pass criteria:**
- `vr/lambda_ema` stays in `[−1.05, −0.95]` after a short warm-in
  (~200 steps) through end of training.
- `vr/lambda_batch` noise band is tight enough that the EMA is meaningful
  (rough check: per-step λ_batch within ±0.1 of EMA in steady state).

**Fail mode:** if λ drifts (e.g. settles at −0.7, or walks), the LF/HF
decomposition is *not* the right story on Anima data. The whole proposal
is dead and the VR-loss bench's λ=−0.996 result was an artifact of the
merged-T-LoRA-against-base setup it was measured on.

### Phase 1 — fixed-λ=−1 ablation (cheap)

Run two short matched A/Bs (~2.5k steps each) on the same dataset:

- (a) current EMA-learned λ
- (b) fixed `λ = −1` (delete cov/var, EMA, state dict)

**Pass criteria:** validation FM loss curve and eyeball-quality A/B
match within noise. Confirms the EMA is a free parameter the data
doesn't need.

This is independently useful regardless of whether Phase 2 ships — it
simplifies the VR loss handler and is the cheapest item on the VR open
questions list (`vr_loss.md:332-336`).

### Phase 2 — HF-only prototype, frozen base (1-2 weeks)

Build the minimal HF adapter on top of the frozen base and train at
small scale. Compare against weight-additive LoRA+VR at matched compute
(*step budget × step cost*, so HF-residual gets to run more steps in
the same wall-clock).

**Pass criteria:**
- Wall-clock-matched FM loss at least matches LoRA+VR.
- Eyeball A/B on the test prompt set holds or wins.
- HF branch parameter count < weight-additive LoRA at same quality, OR
  matched param count at strictly faster wall-clock.

This is the actual decision point. The proposal ships if Phase 2 passes
and is shelved otherwise.

### Phase 3 — productionization

GUI tab, config block, save/load (separate checkpoint, since no
weight-bake), custom ComfyUI node (`custom_nodes/comfyui-anima-hf-residual/`),
docs entry. Mechanical work; only undertaken after Phase 2.

## Open questions

- **LF context width.** How much spatial grounding does the HF branch
  need? AdaLN-from-pooled-LF is the floor (single vector); cross-attn
  into low-res LF features is the ceiling (full spatial map). Sweep
  cheap → expensive in Phase 2.
- **σ_low choice.** Inherits `vr_fei_sigma_low_div = 4.0` from VR loss.
  But the HF branch is the *only* consumer here (no router co-tuning),
  so the divisor can be re-tuned for sample quality rather than ρ²
  headroom.
- **HF branch initialization.** Zero-init the final out-projection so
  early-training behavior reduces to vanilla base-only FM (the LF target
  is what `ε − x_0` collapses to when x_0^H = 0 in expectation). Same
  pattern as LoRA's zero-`B` init.
- **Interaction with DCW.** DCW operates on the externally visible
  `u_pred` to compute `x0_pred`. If `u_pred = u_pred^L + u_pred^H` is
  the externally visible velocity, DCW inherits cleanly. Worth a
  one-step sanity check after Phase 2 trains a checkpoint.
- **Interaction with Spectrum.** Spectrum caches block outputs. With
  two separate forwards (base on `x_t^L` + HF on `x_t^H`), the cache
  splits cleanly — each branch gets its own Chebyshev forecaster.
  Possibly compounds the Spectrum win. Worth a follow-up bench.
- **Per-band λ revisited?** The headroom bench falsified per-band λ
  *for the weight-additive setup* (`vr_loss.md:206-216`). In the
  structural-split setup, the loss is just plain FM — there is no λ.
  But the question of whether the HF branch should weight its squared
  error spatially (or by frequency sub-band) is the analog and probably
  worth a sweep.

## Implementation map (preview, only if Phase 2 passes)

| Layer | File | Role |
|---|---|---|
| Module | `networks/methods/hf_residual.py` | HF backbone (small DiT-style) + LF context hook |
| Training integration | `train.py::get_noise_pred_and_target` | LF forward (no-grad on base), HF forward (grad on adapter), sum, standard FM loss |
| Loss handler | `library/training/losses.py::_flow_match_loss` | Unchanged — no VR aux needed |
| Config gate | `configs/methods/hf_residual.toml` | New file: σ_low, HF backbone hyperparams, LF context channel |
| Inference | `inference.py` | Split `x_t` into LF/HF each step, sum velocities |
| ComfyUI | `custom_nodes/comfyui-anima-hf-residual/` | Standalone node (no weight-bake path) |
| Bench | `bench/hf_residual/` | Wall-clock-matched A/B vs LoRA+VR + standard LoRA |

## References

- AsymFlow §5.2 (arXiv:2605.12964) — variance-reduction with control
  variate; this proposal is the structural reading of why λ collapses
  to −1 on Anima.
- `docs/experimental/vr_loss.md` — the loss-level integration this
  proposal generalizes.
- `bench/fm_vr_headroom/results/20260514-1300-tlora-vs-base/` — the
  ρ² = 0.9999, λ_global = −0.996 ± 0.002 measurement that motivates
  the structural split.
- `library/runtime/fei.py::gaussian_blur_2d` — the kernel that defines
  the LF/HF split (shared with the FEI router).
- `[[project_vr_loss_status]]` — bench-falsified per-element / per-band
  λ; informs why the structural form drops λ entirely instead of
  trying to refine it.
- `[[project_fera_probe_2band_decision]]` — why 2 bands (LF / HF), not
  3, on Anima latents.
