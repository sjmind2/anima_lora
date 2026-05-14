# Staged 2D routing — co-train ortho-hydra, then add a timestep router

A two-phase routing curriculum for Anima's ortho-hydra / FEI-on-Hydra
setup. Instead of asking one router to disentangle content and timestep
simultaneously, train them in separate phases on top of frozen experts:

1. **Phase 1 — co-train ortho-hydra (no pipeline change).** The current
   default training run already produces shared-A + per-expert
   `(B_k, Q_basis_k)` with a per-Linear rank-R **content router** that
   pools `lx` (post-`lora_down` activations) over the sequence axis. This
   stays as-is.
2. **Phase 2 — add a timestep router.** Freeze experts and the content
   router. Introduce a small **`MLP(σ) → softmax_E` timestep router**.
   The final gate is the factorized product of the two:
   `g(x, σ) = normalize(g_content(x) ⊙ g_time(σ))`. Only the timestep
   router has gradient in this phase.
3. **Phase 3 — joint unfreeze (optional).** Unfreeze experts + content
   router at LR × 0.1 for ~1 epoch. Gated on phase 2 plateau.

The 2D in the title now lives in the **routers**, not in the **experts** —
experts stay amorphous (`E` of them, no `C × B` grid), and the gate
function `g: (x, σ) → Δ^E` is factorized as `g_c(x) ⊙ g_t(σ)`.

## TL;DR

```
Phase 1: ortho-hydra with **overlapping σ-band intervals** per expert
         e.g. E=6:  e0..e2 ∈ [0.0, 0.4]
                    e3..e5 ∈ [0.3, 0.7]
                    (extend to E=9 with [0.6, 1.0] for finer σ slicing)
         out-of-interval experts → softmax logit -∞ at that σ (existing
         specialize_experts_by_sigma_buckets mechanism, extended to
         overlapping intervals)
         → anima_hydra*_moe.safetensors  (existing format, new metadata key
                                           ss_sigma_expert_intervals)

Phase 2: --resume anima_hydra_moe.safetensors
         --freeze_experts --freeze_content_router --drop_sigma_mask
         --enable_time_router
         ↓
         time_gate    = softmax(MLP_τ(σ))        # ~1–4k params total
         final_gate   = (g_content ⊙ time_gate); renormalize
         → anima_hydra_2r_moe.safetensors  (adds time_router.* keys)

Phase 3: unfreeze experts + content router at LR×0.1 for ~1 epoch
         (gated on phase 2 sample-quality plateau)
```

## Why factorize the gate

The existing single-router options each commit to one input axis:

| Option | Router input | What it loses |
|---|---|---|
| Default ortho-hydra (`router_source="input"`) | pooled rank-R `lx` per layer | No σ awareness — same gate at all timesteps |
| `router_source="sigma"` | σ scalar (per-layer σ MLP) | No content awareness |
| `router_source="fei"` (FEI-on-Hydra) | FEI(z_t) | σ exposed only implicitly through DoG band energies |
| `specialize_experts_by_sigma_buckets` | (hard partition, no learning) | Locks the σ↔expert assignment a priori |

A factorized gate `g(x, σ) = g_c(x) ⊙ g_t(σ) / Z` lets each axis own its
own decomposition. The content router is already validated (rank-R router
rewiring on 2026-04-20, `docs/methods/hydra-lora.md` §"Fixes"). The
timestep router is small enough (~1–4k params) that it trains cheaply
even on a frozen network.

The two-phase split is what makes this not equivalent to "just train
both routers jointly":

- **Phase 1 gives the content router a clean denoise-loss signal** —
  no second router fighting for the same gradient.
- **Phase 2 gives the timestep router a clean denoise-loss signal** —
  experts and content router are fixed, so the only knob the optimizer
  has to reduce loss is "weight which expert is on at each σ."

Co-trained, the two routers would have to disentangle two confounded
gradients while the experts are also still moving. The staged version
solves three problems in series rather than one problem with three
unknowns.

### Cold-start is a non-issue here

Phase 1 is stock ortho-hydra. Shared-A's cold-start problem — the reason
HydraLoRA needs `expert_warmup_ratio` / `expert_init_std`
([[project_hydra_init_std_dead_code]]) and why the narrow-layer fallback
warns — is already solved by OrthoHydra's per-expert sequential SVD
slicing of `P_bases` (`docs/methods/hydra-lora.md` §"Orthogonalized
experts — fallback behavior"). No curriculum-as-warmup argument needed.

Phase 2 starts from a network where experts have already differentiated
under the content router. The timestep router sees a non-degenerate
mixture-of-experts surface from step 0 — no symmetry to break.

## What's in the repo, what's new

| Need | Status |
|---|---|
| Shared-A + per-expert ortho experts | **Exists.** `OrthoHydraLoRAExpModule` (`networks/lora_modules/ortho.py`). Default in `configs/methods/lora.toml` and `configs/gui-methods/hydralora_experimental.toml`. |
| Per-Linear rank-R content router on `lx` | **Exists.** `_compute_gate` in `networks/lora_modules/hydra.py` (post-2026-04-20 fix). |
| σ broadcast to every adapted module | **Exists.** `set_sigma` path on the network (used today by `router_source="sigma"`). Aliasing bug already fixed ([[project_set_sigma_aliasing_bug]]). |
| Non-overlapping σ-band hard mask | **Exists.** `specialize_experts_by_sigma_buckets` + `sigma_bucket_boundaries` in `networks/lora_modules/hydra.py:_register_sigma_band_partition`. |
| **Overlapping per-expert σ-interval mask** | **New, small.** Extend `_register_sigma_band_partition` to accept `sigma_expert_intervals: List[(σ_min, σ_max)]` of length E; mask logic becomes `σ ∈ [σ_min_e, σ_max_e]` per expert. Metadata stamp `ss_sigma_expert_intervals`. ~30 lines + tests. |
| **TimestepRouter module (`MLP(σ) → softmax_E`)** | **New.** ~50 lines: 2-layer MLP, SiLU, optional σ Fourier features, returns `(B, E)` gate. Lives at `networks/lora_modules/time_router.py`. |
| **Gate composition in HydraLoRAModule.forward** | **New, small.** ~5 lines in `_apply_expert_mixture` (or equivalent): replace `gate = content_gate` with `gate = normalize(content_gate * time_gate.broadcast)`. Gate when phase 2 is enabled; bypass otherwise. |
| **Freeze experts + content router** | **New, small.** Two flags (`--freeze_experts`, `--freeze_content_router`) that gate `requires_grad` in `factory.py`. ~15 lines. |
| **Save/load `time_router.*` in safetensors** | **New, small.** Add the time-router state to `network.state_dict()`; sniff time-router keys on load so existing single-router checkpoints still work. ~30 lines across `loading.py` and `save_weights`. |
| **ComfyUI mirror** | **New, small.** `custom_nodes/comfyui-hydralora/` already plumbs σ to its forward hooks (the σ-router variant); add the time-router branch with the same gate composition. ~40 lines + tests. |
| **Inference path** | **New, small.** Same as ComfyUI hook — the dynamic-routing path in `inference.py` reads σ per step anyway. ~10 lines. |

**Total new code: ~150–200 lines** plus tests. No new training mode, no
assembler, no k-means clustering, no per-slice subset bookkeeping — phase
1 is literally `make lora-gui GUI_PRESETS=hydralora_experimental`.

## Phase 1 — co-train ortho-hydra with overlapping σ intervals

Run today's default ortho-hydra config plus **per-expert overlapping σ
intervals**. Each expert is assigned a σ range; at every step, experts
whose interval doesn't contain the current σ get logit `-∞` before the
softmax — same mask mechanism the existing
`specialize_experts_by_sigma_buckets` uses, generalized from
non-overlapping edges to per-expert `(σ_min, σ_max)` pairs.

The deliverable is `output/ckpt/anima_hydra*_moe.safetensors` in the
existing format, with one new metadata key:
`ss_sigma_expert_intervals = [[0.0, 0.4], [0.0, 0.4], [0.0, 0.4],
[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]` (length E).

### Why overlap, not the existing non-overlapping partition

| | Non-overlapping (today) | **Overlapping (proposed)** |
|---|---|---|
| Each expert sees | exactly one σ band | a wider σ neighborhood; the overlap zone is seen by experts on both sides |
| σ boundary behavior | discontinuity: expert e3 (σ ≤ 0.4) and e4 (σ ≥ 0.4) have never seen each other's regime | smooth: both e3 and e4 are valid at σ ∈ [0.3, 0.4] |
| Inference with single-phase Hydra | clean — content router picks within band, hard mask picks band | **broken** — overlap zone has no unambiguous band, soft mask blends untrained regimes |
| Inference with phase-2 time router | (n/a; this isn't the design) | **fine** — the learned time router blends overlapping experts smoothly because both were trained on the overlap zone |

The "ambiguous band assignment at overlap zones" problem that kept
overlap off the table for single-phase Hydra is exactly what phase 2
solves. A hard partition needs an unambiguous σ→band map at inference;
the learned time router doesn't.

Two side benefits:

- **Implicit regularization on σ.** Each expert sees a slightly wider
  training distribution, so it can't collapse onto a single-σ feature.
- **Phase 2 has overlapping evidence to learn from.** At σ=0.35 in the
  E=6 example, three experts trained on [0, 0.4] and three on [0.3, 0.7]
  are all valid. The time router has six gradients pointing at six
  experts with overlapping competencies — exactly the supervision it
  needs to learn smooth σ-conditioned blends.

### Implementation

Extend `_register_sigma_band_partition`
(`networks/lora_modules/hydra.py:_register_sigma_band_partition`) to
accept a per-expert `intervals: List[(σ_min, σ_max)]` of length E
instead of (or alongside) `boundaries: List[float]` of length B+1.
At forward time, the mask becomes `σ ∈ [σ_min_e, σ_max_e]` rather than
`band[e] == band_at(σ)`. The existing non-overlapping API is a special
case of the new one (intervals derived from edges). Metadata stamp
`ss_sigma_expert_intervals` for ComfyUI/CLI reconstruction. ~30 lines.

### Diagnostics for phase-2 entry gate

- `‖router.weight‖` final / init per Linear (should be > 1.5× per the
  current exit criterion).
- Median normalized gate entropy ∈ [0.6, 0.95] across adapted modules,
  **measured per σ interval** (overlap zones should have higher entropy
  than band-interior σ since more experts are eligible).
- Mean dominant-top1 > 0.2 per σ interval (each interval's eligible
  experts are actually discriminating).
- Zero dead experts on the per-Linear expert-usage histogram (mean-gate,
  per [[project_fera_expert_usage_mean_gates]]).
- **Overlap-zone coverage.** Verify each overlap zone got enough
  training samples — under uniform σ sampling, overlap zone width × N
  ≥ ~5% of total samples per zone.

If any of these fail, phase 2 has nothing to compose against — the
content router didn't learn — and the right move is to fix phase 1, not
add a second router on top of a broken first one.

## Phase 2 — add a timestep router

### Architecture

```
σ : (B,)
  └─ Fourier features φ(σ) : (B, F)   [F = 16, lift to sin/cos × 8 freqs]
       └─ Linear(F → H) → SiLU → Linear(H → E) : (B, E)        [H = 32, E = num_experts]
            └─ softmax over E with temperature τ_t              # one TimestepRouter
                                                                  per Linear (shared
                                                                  across the sequence
                                                                  dim by construction)
```

One TimestepRouter per network — *not* per-Linear. σ is a global scalar
for the forward pass; the same time gate applies to every adapted Linear.
Per-Linear time routers would be redundant (no per-Linear signal in σ)
and would E× the parameter count for nothing.

Parameter count: `F + F·H + H + H·E + E ≈ 1.4k` at `F=16, H=32, E=4`.
Negligible vs. expert weights.

### Gate composition

```
g_content : (B, E)   from existing per-Linear rank-R router
g_time    : (B, E)   from network-level TimestepRouter
g_final   : (B, E) = normalize(g_content ⊙ g_time, dim=E)
```

Hadamard then renormalize. Multiplicative composition is the natural
factorized prior — `P(expert k | x, σ) ∝ P(expert k | x) · P(expert k |
σ)` under conditional independence — and matches how the
`specialize_experts_by_sigma_buckets` hard mask currently composes
(out-of-band logits → `-inf`, then softmax).

Alternative: additive in logits, `g_final = softmax(log g_content +
log g_time, dim=E)`. Equivalent up to normalization but numerically
stabler. Pick at implementation time.

### Init

Initialize the TimestepRouter's final Linear with `weight=0, bias=0` so
`g_time = uniform(1/E)` at step 0. With Hadamard composition + renormalize,
this leaves `g_final = g_content` at step 0 — phase 2 starts from a
network behaviorally identical to phase 1's endpoint. Gradient on the
timestep router is then purely the time-conditioning signal.

### Training signal

Phase 2 trains on the full dataset (same as phase 1) but with only the
timestep router unfrozen. Denoise loss is the only signal. Balance loss
should be **off** during phase 2 — the content router already balanced;
re-pulling toward uniform here would just fight the time conditioning we
want to learn.

Epochs: small. Phase 2 has ~1.5k trainable params; it should plateau in
~1–2 epochs at most. If it doesn't, the diagnosis is the
TimestepRouter is over-parameterized (H too large) or σ-conditioning
genuinely has no signal here, not that it needs longer training.

### What this is *not*

- Not a replacement for `specialize_experts_by_sigma_buckets`. The hard
  σ-band partition is a *structural prior* (you commit to band edges and
  enforce them); the learned time router is the soft *learned* version of
  the same idea. Use one or the other in phase 2, not both. The proposal
  defaults to learned; for very small datasets where data won't support a
  learned 2-layer MLP, the hard partition is a fallback.

## Phase 3 — joint unfreeze (gated)

If phase 2 plateaus *below* a non-staged baseline, the diagnosis is that
phase-1 experts are pointing in directions the time router can't fully
exploit by reweighting. Unfreeze experts + content router at LR × 0.1
for ~1 epoch.

Guardrail: stamp `‖W_k‖` and `‖router_content.weight‖` pre-phase-3.
After phase 3, kill if the relative drift exceeds 30% on most slots —
the curriculum has been erased and we're back to co-training.

## Bench plan — `bench/staged_2d_routers/`

Standard envelope via `bench/_common.py` (`make_run_dir`, `write_result`).

**Phase 0 runs (5 total):**

| Run | What | Wall-clock |
|---|---|---|
| A  | Stock ortho-hydra (`hydralora_experimental`), 12 epochs, **no σ mask** | 1× |
| A′ | Ortho-hydra + **overlapping σ-interval mask** (E=6, intervals as in TL;DR), 12 epochs | 1× |
| B  | Stock ortho-hydra + `router_source="sigma"` (one σ-router), 12 epochs | 1× |
| C  | A′ + phase 2 (frozen experts + content router, drop σ mask, 2 epochs time-router) | 1.15× |
| D  | C + phase 3 (joint unfreeze LR×0.1, 1 epoch) | 1.25× |

A vs. A′ isolates the *cost* of the overlap mask in single-phase use
(should be ≤ A on quality — overlap restricts per-step gradient to
in-interval experts, so quality could regress without phase 2 to
recover). A vs. C isolates the *value* of the two-phase recipe with
overlap. An optional A→C run without overlap (call it C₀) isolates
whether overlap is necessary or just nice-to-have; cheap to add if A′
turns out competitive with A.

**Metrics:**

- FM val loss (necessary, not sufficient — [[project_fm_val_loss_uninformative]]).
- Sample quality on 16-prompt × 3-seed inference grid (same eval harness
  as DCW benches).
- Router diagnostics on C and D: `‖time_router.weight‖` final/init,
  median time-gate entropy per σ-bucket (split into B=2 σ-bands at σ=0.125),
  per-expert time-gate distribution.
- Composition diagnostic: at each Linear, compare `g_content(x)` (frozen
  from phase 1) vs `g_final(x, σ)` (phase 2 output) — measure how much
  the time router actually rotated the gate.
- Expert-drift on D: `||W_k_phase3 − W_k_phase2|| / ||W_k_phase2||`.

**Exit gates.**

- C > A on sample quality → time router added value over single-router
  ortho-hydra. Greenlight ComfyUI mirror + ship.
- C ≈ A, C > B → factorized is at least as good as the σ-only router and
  preserves content info; ship as alternative.
- C < A → time router is fighting the content router. Investigate:
  - Time-gate near-uniform on C? → TimestepRouter capacity / signal too
    low; the σ scalar doesn't carry enough info. Try FEI features as the
    time-router input (the `router_source="fei"` info is σ-correlated via
    band energies).
  - Time-gate non-uniform but quality drops? → multiplicative composition
    is suppressing experts the content router wanted on. Try additive
    composition or temperature τ_t > 1 to flatten the time gate.
- D > C → phase 3 needed; ship the 3-phase recipe and update gui-method.
- D < C → phase 3 erases curriculum; ship the 2-phase recipe only.

## Decision tree

```
C > A on quality?
├── yes, C > B too        → ship 2-phase recipe + ComfyUI mirror
├── yes, C ≈ B            → orthogonal direction; report as alt to σ-router
└── no:
    ├── time-gate near-uniform  → time-router input wrong (σ scalar
    │                              under-informative); retry with FEI
    │                              features or σ + content cross
    └── time-gate active but
        quality drops             → composition rule wrong; try additive
                                    in logits or τ_t > 1; if still bad,
                                    archive (factorization assumption
                                    fails on this dataset)
```

## Risks

- **σ alone is too low-bandwidth.** If the content router has already
  captured σ-implicit information (via `lx` correlating with timestep),
  the time router has nothing left to learn. Mitigation: bench A's
  content-gate stability across σ-buckets at inference *before* phase 2.
  If `g_content(x_t)` is already σ-dependent (likely it is, since `lx`
  depends on `x_t` which depends on σ), then the time router only adds
  value by *explicit* σ conditioning that the content router can't
  express. Fallback: time-router input becomes `[σ, content_gate]`
  instead of just σ.
- **Multiplicative composition can starve experts.** If `g_content`
  weights expert 0 at 0.9 and `g_time` weights expert 0 at 0.1, the
  product is 0.09 → renormalized that's still small. The time router
  can effectively *veto* the content router's preferred expert. Wanted?
  Probably yes — that's the whole point of factorization — but it means
  phase 2 can hurt quality on prompts where the content router was right.
- **Phase 2 plateau looks like phase 1 plateau.** With ~1.5k params,
  the time router will plateau fast on FM loss. Don't mistake a fast
  plateau for "the time router didn't help" — measure on sample quality,
  not on val FM-MSE ([[project_fm_val_loss_uninformative]]).
- **Overlap-zone sample density.** Under uniform σ sampling, an overlap
  zone of width 0.1 (e.g. [0.3, 0.4] between two intervals) gets only
  ~10% of samples — and that 10% is shared across 2× the eligible
  experts, so per-expert gradient density in the overlap zone is ~5% of
  the interior. Too-narrow overlap zones won't actually teach the
  bordering experts to be compatible. Mitigation: σ-sample
  oversampling in the overlap zones (one-line bias in the sampler), or
  widen the overlap. The E=6 intervals in the TL;DR have ~25% overlap
  width which should be comfortably above the floor.
- **Choosing overlap width.** Too narrow (5%) = same as no overlap. Too
  wide (50%) = no σ-specialization, every expert sees almost all σ. The
  TL;DR's 25% (`[0.0, 0.4]` ∪ `[0.3, 0.7]` has overlap of 0.1 on a
  range of 0.4 per interval = 25%) is a reasonable starting point but
  worth ablating at phase 1.
- **Phase-2 composition with the hard σ mask.** Phase 2 in the
  default recipe **drops** the σ mask (`--drop_sigma_mask`) and lets
  the time router replace it. The hard mask was a phase-1
  symmetry-breaker; phase 2's job is exactly the soft version of the
  same prior. Keeping the mask in phase 2 is a fallback for the case
  where the time router can't learn the σ→expert assignment from
  scratch — ablation row in the bench.
- **OrthoHydra Q_basis interaction.** Each expert k has a Q_basis_k that
  selects an orthogonal slice of the SVD-of-shared-A subspace. The time
  router reweighting experts effectively reweights which subspace slices
  are active at which σ. This is the *desired* behavior (low-σ wants
  detail-slices, high-σ wants structure-slices) — and is the analogue of
  the "frequency band experts" the FeRA probe was trying to learn
  ([[project_fera_probe_2band_decision]]). Empirically check whether the
  learned time gate matches the σ-band sparsity pattern from the probe.

## What this proposal is not

- **Not a new adapter family.** Phase 1 trains stock ortho-hydra. Phase
  2 adds a small Linear+SiLU+Linear module + gate composition. Tier 1.5
  in `CONTRIBUTING.md` (efficiency / numerics revision + bench).
- **Not a replacement for σ-router or σ-band partition.** Those remain
  as one-router and hard-prior alternatives. Bench C vs. B decides
  which to recommend by default.
- **Not coupled to FeRA / independent-A.** Shared-A is the whole
  premise (HydraLoRA's contribution). The independent-A variant would
  need a separate proposal — the gate composition would be the same,
  but phase 1 cold-start no longer rides on OrthoHydra's slice
  partition.
- **Not a slicing / curriculum approach.** Earlier drafts of this
  proposal sliced the dataset by (content cluster × σ-band) and trained
  C·B specialist LoRAs in phase 1. That direction is preserved in git
  history for reference, but the user's call to lean on ortho-hydra +
  HydraLoRA's shared-A finding makes it unnecessary — co-training in
  phase 1 already produces differentiated experts without an explicit
  slicing.

## Open questions

1. **Time-router input.** σ scalar (Fourier-lifted) is the minimal
   bandwidth that's still defensible. Alternatives: σ + pooled-FEI(z_t),
   σ + content_gate (one-hot or soft), σ + per-block stage embedding
   (let the time router differentiate by block depth). Phase 0
   ablation.
2. **Gate composition rule.** Hadamard + renormalize vs. additive in
   logits vs. temperature-scaled mixture. Likely empirically close;
   default to additive-in-logits for stability and check Hadamard at
   phase 0.
3. **Is one TimestepRouter per network enough, or should it be per-block?**
   σ is global but the *useful* σ-conditioning per block may differ
   (early blocks structure, late blocks detail). Per-block time router =
   `n_blocks × ~1.5k` params, still cheap. Phase 0 ablation: per-network
   vs. per-block.
4. **Phase-2 epochs.** With ~1.5k params, probably 1 epoch is enough.
   But check whether the time router benefits from running on a
   curriculum (low-σ first, then high-σ) — same intuition as the
   original 2D slicing, applied to phase-2 supervision rather than
   phase-1 experts.
5. **Composition with DCW v4.** DCW is a per-step λ_i correction at the
   sampler boundary, conditioned on `(aspect, prompt_emb, g_obs)`. The
   time router is a per-step expert reweighting inside the DiT,
   conditioned on σ. Likely orthogonal — DCW corrects bias *after* the
   denoiser, the time router shapes *what the denoiser computes*. Bench
   check at C: does the optimal DCW λ change after adding the time
   router?
6. **Overlap interval geometry.** Are uniform overlapping intervals
   (E=6, three pairs of width 0.4 each with 0.1 overlap, evenly
   spaced) optimal, or should overlaps concentrate where the FM
   schedule actually transitions (e.g. dense overlap near σ≈0.7 where
   most generation work happens)? Ablate at phase 1 by comparing
   uniform vs. schedule-aware interval layouts.
