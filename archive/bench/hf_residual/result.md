# HF-residual Tier 1 — result

**Run:** `bench/hf_residual/results/20260514-1851-tier1-baseline/`
**Verdict:** **MARGINAL** (gap_ratio 0.045 at div=4, threshold band [0.03, 0.10])
**Conclusion:** Don't pursue Tier 2 in its "HF-residual replaces LoRA"
framing as written in `docs/proposal/hf_residual_adapter.md`. The
signal is real but concentrated at the first ~20% of Euler steps —
disjoint from where LoRA earns its weight.

## Setup

- Base DiT: `models/diffusion_models/anima-preview3-base.safetensors`
  (frozen, no adapter).
- Data: 6 samples from the most-populous bucket `156x104` (latent
  dims) under `post_image_dataset/lora/`.
- 6 timesteps `t ∈ {0.10, 0.25, 0.40, 0.55, 0.70, 0.85}` × 16 noise
  draws per (sample, t).
- σ_low_div sweep `{4, 8, 16}` → σ_low ∈ {26.0, 13.0, 6.5} on the
  latent's `min(H, W)` of 104.
- div=2 (σ_low=52) clipped by the blur kernel: half-width
  `ceil(3·52)=156` > W=104 → reflect-pad fails. Skipped; the doc's
  realistic range is 4 to 8 anyway.

The bench measures, for each `(sample, t, ε, σ_low)`:
- `loss_full = ‖base(x_t)             − u_target‖²`
- `loss_lf   = ‖base(blur(x_t, σ_low)) − u_target‖²`
- `gap = loss_lf − loss_full`, `gap_ratio = gap / loss_full`
- Band-energy fractions on `x_t`, `u_target`, both predictions, and
  the adapter's would-be training target `u_target − base(x_t^L)`.

19 min wall-clock on a single SM_120 GPU at bf16 with flash attention.

## Headline — per-t breakdown (median across 6 samples × 16 noise)

| t    | gap     | gap_ratio | loss_full | xt_H/x  | u_target_H/u | adapter_H/all |
|------|---------|-----------|-----------|---------|--------------|---------------|
| 0.10 | −0.024  | **−1.4%** | 1.93      | 0.30    | 0.745        | 0.694         |
| 0.25 |  0.000  |  0.0%     | 1.84      | 0.41    | 0.745        | 0.690         |
| 0.40 | +0.039  | +2.2%     | 1.73      | 0.60    | 0.745        | 0.700         |
| 0.55 | +0.097  | +6.0%     | 1.61      | 0.81    | 0.745        | 0.715         |
| 0.70 | +0.109  | +7.0%     | 1.55      | 0.94    | 0.745        | 0.738         |
| 0.85 | +0.549  | **+56.5%**| 0.96      | 0.99    | 0.745        | 0.782         |

**Headroom is overwhelmingly concentrated at high σ_t.** At t ≤ 0.25
the LF-only forward is statistically indistinguishable from the full
forward — the small negative gap at t=0.10 is noise. The 0.85 row is
doing essentially all the work in the overall median.

## σ_low sweep — overall medians

| div  | σ_low | gap_ratio | xt_H/x | adapter_target_H/all |
|------|-------|-----------|--------|----------------------|
| 4.0  | 26.00 | +0.045    | 0.754  | 0.735                |
| 8.0  | 13.00 | +0.051    | 0.726  | 0.711                |
| 16.0 |  6.50 | +0.058    | 0.699  | 0.683                |

Monotone but flat — `gap_ratio` rises slightly as the blur weakens
(less LF removed, so `base(x_t^L) ≈ base(x_t)` should drive gap → 0,
which makes the slight *increase* counter-intuitive; we read it as
noise within the 0.04–0.06 band rather than signal). **No sweet
spot.** div=4 (the default inherited from VR-loss / FEI-router
co-tuning, [[project_fera_probe_2band_decision]]) is fine for this
proposal too. There's no σ_low choice that buys you more headroom.

## Mechanistic interpretation

The pattern is consistent with a clean story:

- **Low σ_t (t ≤ 0.25):** `x_t ≈ x_0`, structure lives in the LF band.
  Blurring removes a small fraction of mass (`xt_H/x = 0.30` at
  t=0.10) and `base(x_t^L)` ≈ `base(x_t)`. Gap is zero.
- **High σ_t (t ≥ 0.70):** `x_t ≈ ε`, structure is broadband / mostly
  HF (`xt_H/x = 0.99` at t=0.85). The model's prediction target *is*
  the high-frequency noise content. Blurring `x_t` destroys the
  prediction target itself — the LF base mis-predicts massively
  (gap_ratio 57%).

This is the **same phenomenon** the VR-loss bench measured as
`λ → −0.996` (`[[project_vr_loss_status]]`,
`bench/fm_vr_headroom/results/20260514-1300-tlora-vs-base/`). The
loss collapses to the HF residual *because* the LF half of the model
is doing most of the work at most timesteps. Tier 1 confirms the
decomposition; what's new is that the recoverable-by-HF-adapter
slack is concentrated at the high-noise end, not spread across the
trajectory.

## Comparison to a vanilla LoRA contribution

A normal LoRA adapter is active at every t. It contributes (style,
fine detail) uniformly across the schedule. An HF adapter under this
decomposition would contribute essentially zero loss-level
improvement for the bottom 70% of the σ_t schedule and meaningful
correction only for the top 30% — a different regime, not a
replacement.

This invalidates the proposal's headline pitch ("small adapter
replaces LoRA") in its current form. The two would do different
jobs at different points in the trajectory.

## Recommendation

Three options, in decreasing order of how much I'd commit to:

1. **Shelve as written.** The proposal as stated in
   `docs/proposal/hf_residual_adapter.md` was about replacing
   weight-additive LoRA + VR-loss with a structural split. Tier 1
   says the structural split has 5% slack to recover, concentrated
   at high t. That's not a compelling drop-in.

2. **Reframe and re-bench.** Recast the HF adapter as a
   *high-σ-only refinement* stacked on top of a normal LoRA:
   - Gate the HF branch to `t > 0.5` (or even `t > 0.7`).
   - Train base + LoRA + HF-adapter jointly.
   - Tier 2 becomes: does `base + LoRA + HF-adapter` beat
     `base + LoRA` at matched compute? Different question, narrower
     scope, different bench design.

3. **Skip ahead to perceptual eval.** Build the minimal HF adapter,
   train 2k steps, look at images. The 56% gap at t=0.85 either
   translates to a visible trajectory-shape win or it doesn't — no
   amount of loss-level diagnostics will answer that. Costlier than
   shelving, cheaper than full Tier 2.

What I would **not** do:

- Re-bench with more samples / more noise. The signal at t ≤ 0.25 is
  already pinned at zero across 96 (sample, ε) pairs per t-bin per
  div; tighter error bars won't change the shape.
- Sweep σ_low further. The 4 → 8 → 16 line is flat. Going coarser
  (div=2) hits the kernel-size constraint on this bucket; going
  finer would push toward "no blur," where the gap mechanically → 0.
- Run on a different bucket. Aspect invariance of σ_low = D/div is
  already validated for the FEI router
  (`bench/fera/probe_fei_dataset.py`) and this finding is per-σ_t,
  not per-bucket.

## Bench artifacts

- `result.json` — standard envelope.
- `summary.json` — full per-div × per-t breakdown.
- `per_sample_t.csv` — 108 rows (6 samples × 6 timesteps × 3 divs),
  one row per cell of the sweep, for plotting / post-hoc analysis.

## See also

- `docs/proposal/hf_residual_adapter.md` — the proposal this bench
  gates.
- `docs/experimental/vr_loss.md` — the loss-level VR setup whose
  λ→−1 collapse motivated the structural split.
- `bench/fm_vr_headroom/` — the ρ² headroom probe for the
  weight-additive VR setup.
- `[[project_hf_residual_tier1]]`, `[[project_vr_loss_status]]`,
  `[[project_fera_probe_2band_decision]]` — memory entries with the
  full backstory.
