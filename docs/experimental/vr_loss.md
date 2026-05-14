# Variance-Reduced FM Loss (AsymFlow §5.2)

Training-loss-level integration of the control-variate correction from
Chen et al., *Asymmetric Flow Matching for Pixel-Space Generation*
(arXiv:2605.12964 §5.2). The flow-matching MSE estimator is variance-reduced
by pairing each step with a no-grad forward of the **base DiT** on the
FEI-low-passed latent. For LoRA-family runs the base DiT is frozen and the
adapter is additive, so we get the "frozen reference" by reusing the
trainable DiT with `network.set_multiplier(0)` for the no-grad pass — no
second model copy in VRAM. ~99.8% of the per-sample loss variance was found
recoverable on Anima at the global-λ optimum in the headroom bench
(`bench/fm_vr_headroom/results/20260514-1300-tlora-vs-base/`, verdict
**HEADROOM**).

This is a **loss-level change**, not a new adapter — the trained checkpoint
inferences identically to a standard FM-trained one. Composes with every
adapter family in `networks/lora_anima/`.

## TL;DR

```
# Standard FM (per element, in velocity-target form)
y        = u_pred − (ε − x_0)                       # = (x_0 − x̂_0) / σ_t
L_FM     = ||y||²

# VR loss (this addition)
x_0^L    = gaussian_blur_2d(x_0, σ_low)             # FEI-aligned low-pass
x_t^L    = (1 − σ_t) · x_0^L + σ_t · ε              # paired noisy input, SAME ε
u_pred^L = anima(x_t^L, t, te)  with mult=0         # no-grad bypass forward
z        = u_pred^L − (ε − x_0^L)                   # control variate residual
L_VR     = ||y + λ · z||²                           # gradient flows through y only
```

with `λ = −Cov(y, z) / Var(z)` estimated online (per-batch covariance + EMA
across batches, β default 0.01). One extra **no-grad** forward per step
(~+40% step cost). The "frozen reference" is the trainable DiT itself with
`network.set_multiplier(0)` during the no-grad pass — equivalent to a frozen
base DiT for LoRA-family runs (the base is frozen, adapters are additive),
with zero extra VRAM for a second model copy.

## Quick start

```bash
# Enable on any LoRA-family run:
make lora --vr_loss_weight 1.0
make lora-gui GUI_PRESETS=tlora --vr_loss_weight 1.0
python tasks.py lora --vr_loss_weight 1.0

# Or flip the keys in configs/methods/lora.toml (commented out by default):
#   vr_loss_weight = 1.0
#   vr_fei_sigma_low_div = 4.0
```

VR is gated off by default (`vr_loss_weight = 0.0`). When `> 0`, the trainer
runs one extra no-grad forward per step through the trainable DiT with
`network.set_multiplier(0)` (zeros both LoRA and ReFT contributions). No
extra model is loaded into VRAM. The only cost is the extra forward
(~+40% step time); low-VRAM presets can run this — they just pay the
compute.

## What it actually does

```
                              latents x_0  (B, C, H_lat, W_lat)
                                    │
                       ┌────────────┴───────────┐
                       │                        │
                       │              gaussian_blur_2d(σ_low)
                       │                        │   library/runtime/fei.py
                       │                        ▼
                       │                       x_0^L
                       │                        │
                       │   ε ───────┐           │   ε (same draw)
                       ▼            ▼           ▼            ▼
              (1−σ_t)·x_0 + σ_t·ε       (1−σ_t)·x_0^L + σ_t·ε
                       │                        │
                       │                        ▼
                       │            ┌─ same DiT, mult=0 (no_grad, bf16)
                       │            │  (no separate model load)
                       ▼            ▼
            trainable DiT      u_pred^L
                  │                  │
              u_pred                 │
                  │                  │
                  └──── y = u_pred − (ε − x_0) ──┐
                                                 │
                  z = u_pred^L − (ε − x_0^L) ────┤
                                                 ▼
                              loss = ||y + λ · z||²    (per element)
                                          │
                              λ_batch = −Cov(y_det, z) / Var(z)
                              λ_ema   ← (1−β)·λ_ema + β·λ_batch
```

### Choice of `x_0^L`

```
σ_low = min(H_lat, W_lat) / vr_fei_sigma_low_div    # default 4.0
x_0^L = gaussian_blur_2d(x_0, σ_low)
```

Same FEI kernel that drives the live HydraLoRA / FeRA routing on
`router_source = "fei"`. Three reasons:

1. The model's adapter routing is already shaped around this band split, so
   the control variate inherits the same inductive bias instead of inventing
   a new one.
2. `library/runtime/fei.py::gaussian_blur_2d` is in-tree, fp32-safe, and
   kernel-cached — no new module.
3. The 2-band FEI gives a free diagnostic axis: per-FEI-band ρ² in the bench
   already shows the high-band ρ² is what carries the headroom (≈ 0.998
   mid-t median).

Default `vr_fei_sigma_low_div = 4.0` matches the live training default in
`configs/gui-methods/fera.toml` and `configs/gui-methods/hydralora_fei.toml`.

### λ estimation

`λ = −Cov(y, z) / Var(z)` minimizes `Var(y + λz)`. We use the *global* form
(scalar λ over all latent positions and batch elements), not the per-element
or per-band variants:

```python
y_d = (u_pred − (ε − x_0)).detach()                 # detached residual
cov = (y_d * z).sum()
var = (z * z).sum().clamp_min(1e-12)
λ_batch = float(−(cov / var))
λ_ema   = (1 − β) · λ_ema + β · λ_batch             # β = vr_lambda_beta, default 0.01
```

`λ_ema` is initialized to `λ_batch` on step 0 (no warm-up window). Bench
confirmed `λ_global = −0.996 ± 0.002` across all 36 (sample, t) pairs at
N=32, so the online estimator converges fast and a small β is well
conditioned.

Per-element λ and per-FEI-band `λ_k` are deferred to v2 / v3.

### Adapter-bypass reference

The "frozen reference" is the trainable DiT itself, run no-grad with the
LoRA network's multiplier temporarily set to 0:

```python
_orig_mult = float(getattr(network, "multiplier", 1.0))
network.set_multiplier(0.0)         # zeros LoRA + ReFT (network.py:860-865)
try:
    with torch.no_grad():
        ref_pred = anima(x_t_L_5d, timesteps, crossattn_emb, padding_mask=padding_mask, **kw)
finally:
    network.set_multiplier(_orig_mult)
```

For LoRA-family runs this is bit-equivalent to a frozen copy of the base
DiT: the base weights are frozen for the whole training run, and adapters
are *additive residuals* on top — turning the multiplier to zero collapses
the model to its base. No `--vr_frozen_ref_dit` flag, no second model copy
in VRAM, no `static_token_count` / `trim_crossattn_kv` mirroring to keep in
sync.

`set_multiplier(0)` covers both `LoRA` / `OrthoLoRA` / `HydraLoRA` /
`StackedExperts` *and* `ReFT` (the network walks both lists in one call).
Postfix's `network.append_postfix` modifies `crossattn_emb` *before* the
DiT call, not the DiT itself, so the bypass forward receives the same
postfix-appended tokens as the gradient forward — postfix is therefore
*not* nulled (it can't be: postfix isn't an additive residual on weights).
For runs that combine postfix with VR, this means the control variate
includes the postfix contribution. Plain LoRA-family runs are unaffected.

Hooks on the unet (REPA capture, functional MSE captures) fire on this
forward too, but they are consumed *before* the VR block runs — see the
order in `train.py::get_noise_pred_and_target`.

## Implementation map

| Layer | File | Role |
|---|---|---|
| CLI args | `library/anima/training.py` | `--vr_loss_weight`, `--vr_fei_sigma_low_div`, `--vr_sigma_min`, `--vr_lambda_beta` |
| Config gate | `configs/methods/lora.toml` | Commented `vr_loss_weight = 1.0` block; uncomment to enable |
| Forward + stash | `train.py::get_noise_pred_and_target` | Builds `x_0^L`, `x_t^L`, calls `network.set_multiplier(0)` + no-grad `anima(...)` + restore, stashes `ctx.aux['vr'] = {'z': ..., 'state': ...}` |
| Loss handler | `library/training/losses.py::_flow_matching_vr_loss` | Computes `(y + λ·z)²`, updates `state['lambda_ema']` in place |
| Composer gate | `library/training/losses.py::build_loss_composer` | Replaces `flow_match` → `flow_matching_vr` when `vr_loss_weight > 0` |
| Headroom bench | `bench/fm_vr_headroom/run_bench.py` | The Stage 0 ρ² probe (now in CI-style results) |
| Plan | `bench/fm_vr_headroom/proposal.md` | The integration plan this doc summarizes |

The trainer↔loss-handler contract is the `ctx.aux['vr']` dict:

```python
self._extras_for_step["vr"] = {
    "z": z_residual.detach(),      # (B, C, H_lat, W_lat), no_grad
    "state": self._vr_state,       # {"lambda_ema": float | None, "lambda_batch": float}
}
```

Both `_extras_for_step` and the `state` dict are reset / persistent per the
trainer's existing conventions: `_extras_for_step` clears every step,
`_vr_state` persists across steps (mutated in place by the loss handler so
the EMA survives).

## Compute cost

Per training step:

| | grad fwd | no-grad fwd | bwd | net |
|---|---|---|---|---|
| Standard FM | 1 | 0 | 1 | 1× |
| VR loss     | 1 | 1 | 1 | ~1.4× |

The no-grad forward runs in `torch.no_grad()` inside the same
`accelerator.autocast()` scope as the main forward — no checkpointing,
no graph save, no second backward. On a 5060 Ti at the typical Anima
bucket (`128×128` latent), the extra forward is ~40% the cost of the
gradient-tracked forward, so net step cost is ~1.4×.

If block swap is on, the bypass forward also pays the swap cost. That's
compute, not memory, but it slightly inflates the 1.4× figure on heavily
swapped presets. Forward hooks (REPA capture, functional MSE capture)
also fire on the bypass forward — they're benign because the captured
state is consumed *before* the VR block runs.

Net training win requires VR to give >1.4× effective convergence. The
paper reports +0.96 HPSv3 from VR alone on AsymFLUX.2 klein
(Table 3) but no wall-clock figure, so this has to be re-demonstrated for
Anima — see "Validation status" below.

## Memory

No extra VRAM beyond what standard FM uses. The control-variate forward
reuses the trainable DiT (with adapter multiplier=0), so there is no
second 2B model held in memory. Low-VRAM presets (`low_vram`, `fast_16gb`)
can run VR — they just pay the ~+40% compute.

## Config knobs

| Flag | Default | Meaning |
|---|---|---|
| `vr_loss_weight` | `0.0` | Gate **and** overall scale on the VR loss term. `0.0` disables (standard FM); `1.0` matches the paper recipe; smaller values let other losses contribute relatively more. |
| `vr_fei_sigma_low_div` | `4.0` | Divisor for `σ_low = min(H_lat, W_lat) / div`. Matches the live FEI default. |
| `vr_sigma_min` | `1e-3` | Defensive floor on `σ_t` in the `1/σ_t` factor (AsymFlow §6.1). Not consumed by the v1 loss handler (we work in velocity-target form so `σ_t` cancels) but kept on the parser for v2's per-element-λ extension. |
| `vr_lambda_beta` | `0.01` | EMA rate on `λ`. `λ_ema ← (1−β)·λ_ema + β·λ_batch`. |

## Validation status

| Stage | What | Where | Status |
|---|---|---|---|
| v0 | Frozen-ref + decorrelated-ε-null headroom bench | `bench/fm_vr_headroom/results/20260514-1300-tlora-vs-base/` | ✅ HEADROOM — `ρ²_high_band` mid-t median **0.998**, null gap **+0.988**, `λ_global` **−0.996 ± 0.002**, flat from t=0.10–0.85 |
| v0.5 | Re-run with `anima-preview2` as trainable to bound the trainable-similarity bias | (optional) | Not blocking v1 |
| v1 | Wire VR loss into `train.py` + `library/training/losses.py` | **this doc** | ✅ Compiles, loss-handler unit-checked, pre-existing tests green (`test_loss_registry.py`, `test_config.py`, `test_smoke.py`) |
| v1.5 | Stage 1 A/B: standard FM vs VR on a short LoRA run | `bench/fm_vr_headroom/training_ab.py` (not yet written) | Pending — HPSv3 ≥ +0.02 at fixed step OR ≥ +0.01 at fixed wall-clock, robust across two prompt subsets |
| v2 | EMA frozen ref, per-element λ, LPIPS correction | — | Conditional on v1.5 quality regression at long step counts |
| v3 | Per-band `λ_k` via `_fera_fecl_bands` | — | Conditional on v1.5 showing FEI-bin-dependent gain |

## Risks

1. **Trainable ≠ frozen divergence is small at step 0.** The Stage 0 bench
   used a T-LoRA-merged checkpoint as trainable against a frozen base DiT —
   ρ² = 0.998 with λ pinned to −1 might be measuring "two near-identical
   models always agree on input-pair deltas" rather than "VR cancels real
   per-step gradient noise." The null gap (+0.988) rules out the literal
   shared-model artifact, but Stage 1 A/B is the only way to fully resolve.
   With the adapter-bypass implementation the divergence *grows* across
   training (the trainable DiT keeps drifting; the bypass forward is always
   the current base prediction — fixed across the whole run), which is what
   we want from a variance-reduction control variate.
2. **Bias risk.** `E[z] = (x_0^L − E[ref_pred^L | x_t^L])` is non-zero because
   the reference is biased. With a fixed reference (always base DiT) the
   bias is constant in `λ`, so the variance minimum is still at
   `λ* = −Cov/Var`, but the *expected* loss differs from standard FM. Paper
   resolves with an LPIPS perceptual correction term — v1 skips it. v2 may
   need it if Stage 1 shows long-step quality regression.
3. **Interaction with other losses.** REPA / FECL / functional / soft-tokens
   all stay active and unchanged; only `flow_match` is swapped for
   `flow_matching_vr`. If REPA or FECL silently depended on the FM term's
   native variance, second-order effects are possible. Stage 1 is the only
   way to surface this.
4. **Postfix interaction.** The bypass forward zeros LoRA + ReFT but
   *cannot* null `network.append_postfix` (postfix prepends tokens to
   `crossattn_emb` and is not an additive residual on DiT weights). For
   `postfix + VR` runs the control variate therefore includes the postfix
   contribution, which still cancels noise but biases `λ` differently than
   a pure-base reference would. Plain LoRA-family runs are unaffected.
5. **Static-shape compile.** The extra forward inherits the main forward's
   static padded shape, so `torch.compile`'s `_run_blocks` cache is reused
   for free. The bypass forward runs under `torch.no_grad()`, so gradient
   checkpointing saves no activations and there is no second backward.
6. **Memory.** None. The trainable DiT is the only DiT in VRAM.

## Open questions

- **Reference granularity** — v1 always reads "pure base" (multiplier=0
  on every step). A v2 variant could use the *current* trainable adapter
  at some scale (e.g. multiplier=0.5 or the resumed multiplier) as the
  control variate — that's a one-line change to the `set_multiplier` call,
  but needs thinking about whether the residual `z` stays decorrelated
  from the gradient, since the bypass forward now shares a non-trivial
  function with the gradient forward. v1's pure-base choice keeps `z`
  cleanly independent of the trainable LoRA's current state.
- **Multi-band schedule** — single global `λ` vs `K=3` bands using
  `_fera_fecl_bands`. The bench reports `low_band__` / `high_band__` ρ²
  already; if Stage 1 shows FEI-bin-dependent gain we promote to per-band.
- **CFG-dropout interaction** — v1 uses the *same* (possibly dropped)
  crossattn_emb for both forwards in a step, so cancellation is preserved.
- **σ_min clamping** — paper uses `σ_min = 0.04` on the `1/σ_t` factor.
  v1 works in velocity-target form (`y = u_pred − target`), where `σ_t`
  algebraically cancels, so no clamp is needed. `--vr_sigma_min` is kept on
  the parser for v2's per-element-λ extension.

## References

- AsymFlow paper (arXiv:2605.12964), Chen et al., 2026. §5.2 is the
  variance-reduction section; §6.1 covers the `σ_min` clamp.
- `bench/fm_vr_headroom/README.md` — the diagnostic that gated v1.
- `bench/fm_vr_headroom/proposal.md` — full integration plan.
- `docs/methods/hydra-lora.md` — FEI routing background.
- `[[project_fera_probe_2band_decision]]` — why we use 2 bands, not 3.
- `[[project_fm_val_loss_uninformative]]` — why Stage 1 needs HPSv3/VQA,
  not just val FM curves.
