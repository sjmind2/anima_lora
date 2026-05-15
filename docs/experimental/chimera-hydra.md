# ChimeraHydra — dual-pool additive routing for timestep-aware MoE

Proposal: [`docs/proposal/chimera_hydra.md`](../proposal/chimera_hydra.md).

A single-phase MoE-LoRA recipe on top of the OrthoHydra Cayley parameterization. Two pools of B-heads share one A per adapted Linear:

- **content pool** (`K_c=3` by default) — routed by the per-layer rank-R router on pooled `lx` (the same router OrthoHydra already runs).
- **frequency pool** (`K_f=3` by default) — routed by a network-level `FreqRouter` fed `concat(FEI(z_t), sinusoidal-σ-features)`.

Pool outputs are **added**. No multiplicative gate, no σ-band overlap mask, no staged curriculum. Specialization is *structurally* enforced by router-input separation — the content router can't see σ; the freq router can't see pooled text features. Each pool's B-heads necessarily specialize along its router's available axis.

> Supersedes the earlier staged-2D design (multiplicative gate `g_c ⊙ g_t` + Phase 1/2/3 curriculum), preserved in git history. Staging existed to break gradient confounding in the multiplicative gate; additive composition removes the confounding directly so the curriculum is solving a problem ChimeraHydra doesn't have.

## Quick start

```bash
# Method-config path (canonical entry, default for benching):
make exp-chimera
python tasks.py exp-chimera        # cross-platform

# GUI-friendly per-variant path:
make lora-gui GUI_PRESETS=chimera_hydra
python tasks.py lora-gui chimera_hydra

# Inference against the latest *_chimera.safetensors —
# save distills the Cayley parameterization to the Hydra-MoE layout
# (shared lora_down + per-expert lora_ups.{i}, q/k/v defused) plus the
# top-level FreqRouter block; load rebuilds a HydraLoRAModule with
# num_experts_content > 0 for the dual-pool runtime form.
make test
```

`make exp-chimera` drives `configs/methods/chimera.toml`; `make lora-gui GUI_PRESETS=chimera_hydra` drives `configs/gui-methods/chimera_hydra.toml`. Both build the same module/router stack — pick whichever entry point matches your workflow.

## What it actually does

```
                          z_t  (B, C, H_lat, W_lat)
                                  │
              FrequencyEnergyIndicator (DoG pyramid → simplex)   (library/runtime/fei.py)
                                  │
                              FEI(z_t)
                                  │            σ ─► sinusoidal-features (t_embedder form)
                                  └─────────┬─────────┘
                                            │  concat
                                            ▼
                              FreqRouter (Linear → SiLU → Linear → softmax/τ)
                                            │              ◄── ONE per network, non-zero init
                                       π_f (B, K_f)
                                            │
              ╔═════════════════════════════╧═════════════════════════════╗
              ║                                                            ║
              ▼                                                            ▼
   ┌──────────────────┐                                       ┌──────────────────┐
   │  ChimeraHydra m0 │                                       │  ChimeraHydra m1 │   …
   │                  │                                       │                  │
   │  pooled lx  ──►  content_router (rank-R, per-layer)      │   (same)         │
   │                  │           │                           │                  │
   │                  │      π_c (B, K_c)                     │                  │
   │                  ▼                                       ▼                  │
   │  gate = [π_c | π_f]                                                          │
   │  P_eff (E, out, r) = P_bases @ Cayley(S_p)               (E = K_c + K_f)     │
   │                                                                              │
   │  Δy_c = bmm( (lx · λ · mask) , Σ_c π_c · P_eff[:K_c] )    ◄── content branch │
   │  Δy_f = bmm( (lx · λ),         Σ_f π_f · P_eff[K_c:] )    ◄── freq branch    │
   │  out  = base(x) + Δy_c + Δy_f                                                │
   └──────────────────────────────────────────────────────────────────────────────┘
```

Per adapted Linear:

```
ChimeraHydraLoRAExpModule(x) = base(x) + multiplier · ( Σ_c π_c · B_content[c] (A x)
                                                       + Σ_f π_f · B_freq   [f] (A x) )
```

`A = Cayley(S_q) · Q_basis` is the shared down-projection (one per Linear). `B_content[c] = P_bases[c] @ Cayley(S_p[c])` for `c < K_c`; `B_freq[f] = P_bases[K_c + f] @ Cayley(S_p[K_c + f])`. The SVD column space of the base weight is split by index — the first `K_c · r` columns are content, the next `K_f · r` are freq — so the two pools live in disjoint subspaces of A by construction.

The full gate `cat([π_c, π_f])` flows into the existing OrthoHydra einsum/bmm path; additive composition of the two pools is mathematically identical to single-pool routing with a partitioned gate vector. The only structural diff vs. OrthoHydra is **where the gate comes from**.

## T-LoRA per-branch composition

`use_timestep_mask = true` applies the rank mask `mask_t(σ)` to the **content branch only**. The freq branch keeps full rank at every t.

```
content : lx_c = lx · λ · mask_t(σ)
freq    : lx_f = lx · λ
```

Rationale (proposal §"T-LoRA integration"): T-LoRA's argument is that high-σ steps are where LoRA memorizes layout/identity — the content branch is exactly that risk surface. The freq branch *wants* full rank at high σ to learn coarse-stage denoising features. Per-branch masking falls out cleanly because the pools are physically separate.

Implemented as two bmm calls inside `ChimeraHydraLoRAExpModule.forward`. The path is shape-static under `torch.compile` (no Python-bool guard on mask state) so the extra bmm has no recompile cost.

## Where it sits in the three-axis matrix

The three-axis routing surface (plan2.md §three-axis-config) names the existing variants. ChimeraHydra is a **fourth dispatch cell on top** of the shared-A row, opt-in via a dedicated flag:

| Variant | `use_moe_style` | `route_per_layer` | `router_source` | Extra |
|---|---|---|---|---|
| Plain LoRA / OrthoLoRA / T-LoRA / ReFT | `False` | — | `"none"` | — |
| HydraLoRA (paper) | `"shared_A"` | `True` | `"input"` | — |
| σ-router on Hydra | `"shared_A"` | `True` | `"sigma"` | — |
| FEI-on-Hydra | `"shared_A"` | `True` | `"fei"` | — |
| FeRA (independent-A) | `"independent_A"` | `False` | `"fei"` | — |
| **ChimeraHydra (this doc)** | `"shared_A"` | `True` | `"input"` | `use_chimera_hydra=true` |

`LoRANetworkCfg.from_kwargs` pins the three-axis fields to `("shared_A", True, "input")` whenever `use_chimera_hydra=true`; passing any other value for those fields raises. The chimera flag is the only routing knob you set — the rest follow.

## Implementation map

| File | Role |
|------|------|
| `networks/lora_modules/chimera.py` | `ChimeraHydraLoRAExpModule` (training-only) — subclass of `OrthoHydraLoRAExpModule`. Owns content router (`Linear(r → K_c)` with small-std init), holds the broadcast `_freq_routing_weights` buffer (uniform 1/K_f placeholder, overwritten by the network), and overrides `forward` to issue the per-branch two-bmm composition. `_compute_gate` constructs `cat([π_c, π_f])`. Save distills its Cayley params away; load goes through `HydraLoRAModule` instead, so this class is never instantiated at inference / resume. |
| `networks/lora_modules/hydra.py` | `HydraLoRAModule` gained a `num_experts_content` kwarg. When `> 0` (load path from a distilled chimera file): narrows `self.router` to `(K_c, router_in_dim)`, registers a `_freq_routing_weights` buffer of size `K_f = E - K_c`, and `_compute_gate` cats `[π_c | π_f]`. `set_freq_routing_weights` / `clear_freq_routing_weights` mirror the slot-assign protocol. σ/FEI feature dims are rejected in this mode (the FreqRouter owns those axes). |
| `networks/lora_anima/network.py` | `FreqRouter` (Linear → SiLU → Linear → softmax/τ). `LoRANetwork.__init__` builds one when `cfg.use_chimera_hydra=True` and at least one chimera-aware module was constructed; `_wire_shared_freq_routing_buffers` aliases every chimera module's buffer to a single network-level tensor; `set_fei` fires the router on `concat(FEI, σ-features)` and broadcasts `π_f` via `set_freq_routing_weights` (direct slot assignment — preserves grad_fn). `_get_chimera_balance_loss` splits each module's gate at `K_c` and accumulates two independent Switch losses weighted by `_balance_w_content` / `_balance_w_freq`. Module-construction loop passes `num_experts_content = cfg.num_experts_content` into `HydraLoRAModule` when `cfg.use_chimera_hydra=True`. |
| `networks/lora_anima/config.py` | `LoRANetworkCfg.use_chimera_hydra` / `num_experts_content` / `num_experts_freq` / `balance_w_content` / `balance_w_freq` / `freq_router_init_std`. `from_kwargs` pins the three-axis fields when chimera is on; `from_weights` reconsumes the chimera-specific metadata stamps. |
| `networks/lora_anima/factory.py` | `create_network` builds the chimera spec via `resolve_network_spec` (Cayley `ChimeraHydraLoRAExpModule` for training); `create_network_from_weights` detects `ss_use_chimera_hydra="true"` at metadata load time, keeps the `chimera_hydra` spec but **overrides** `module_class = HydraLoRAModule` (the dual-pool runtime form). It also surfaces the chimera-specific σ/FEI dims into the cfg slots the FreqRouter reads (without these overrides the loader falls back to defaults and the FreqRouter ends up with the wrong input width). |
| `networks/lora_save.py` | `chimera_hydra_moe` save handler — runs `_convert_ortho_hydra_to_hydra` (Cayley → shared `lora_down` + stacked `lora_up_weight`) and `_build_hydra_moe_state_dict` (q/k/v defuse per-expert, clone per-Linear `router.*` into each component), then writes to `*_chimera.safetensors`. Top-level `freq_router.*` keys flow through both conversion steps unchanged. |
| `networks/__init__.py` | `NETWORK_REGISTRY["chimera_hydra"]` with `save_variant="chimera_hydra_moe"`. `resolve_network_spec` short-circuits to chimera_hydra when `use_chimera_hydra=true`. `_post_init_hydra` stamps `_use_chimera_hydra` + per-pool balance weights on the network for the balance-loss handler to consume. |
| `library/inference/models.py` | `_is_chimera_moe(path)` peeks `ss_use_chimera_hydra` from safetensors metadata. Chimera files take the existing Hydra-mode dynamic-hook branch but skip the `lora_unet_*` filter (so top-level `freq_router.*` survives) and pass `file=path` to `create_network_from_weights` so the metadata is read at the factory layer. |
| `library/training/router_conditioning.py` | Already routes `set_sigma` → `set_fei` once per step. The chimera path force-enables `use_fei_router` in `LoRANetwork.__init__` so the FEI/σ pipeline fires every step regardless of `cfg.router_source`. |
| `configs/methods/chimera.toml` | Method config driving `make exp-chimera`. |
| `configs/gui-methods/chimera_hydra.toml` | Self-contained variant config driving `make lora-gui GUI_PRESETS=chimera_hydra`. |
| `scripts/experimental_tasks/training.py::cmd_chimera` | `exp-chimera` shim. |

## Parameter count

Per adapted Linear at default `network_dim=32, num_experts_content=3, num_experts_freq=3` (so E=6):

```
S_p          (E, r, r)    = 6 · 32 · 32                ≈ 6.1k
S_q          (r, r)       = 32 · 32                    ≈ 1.0k
lambda_layer (1, r)       = 32                         (negligible)
content router            = r · K_c + K_c              ≈ 99
P_bases (frozen buffer)   = (E, out, r)                — not counted (buffer)
Q_basis (frozen buffer)   = (r, in)                    — not counted (buffer)
```

≈ 7.2k trainable params per chimera Linear — same OOM as OrthoHydra at E=6.

Network-level FreqRouter:

```
Linear(F_in → H) + Linear(H → K_f) = (2 + 16) · 64 + 64 · 3 + biases  ≈ 1.4k params
```

Negligible; one router serves the whole network.

Total scales like OrthoHydra at `E = K_c + K_f`: at the default `chimera.toml` regex (`*mlp.layer[12]`) on Anima's 28 blocks × 2 MLP layers, that's ~56 chimera modules × 7.2k ≈ 0.4M trainable params + 1.4k for the freq router. Negligible vs. the FeRA-style independent-A budget (~150M); roughly on par with the existing FEI-on-Hydra default.

## Knobs (`configs/methods/chimera.toml`)

| Param | Default | Notes |
|---|---|---|
| `use_chimera_hydra` | `true` | The activation flag. Triggers all three axis pins + builds the FreqRouter. |
| `num_experts_content` (`K_c`) | 3 | Content pool size. |
| `num_experts_freq` (`K_f`) | 3 | Freq pool size. Total `E = K_c + K_f` flows into the OrthoHydra disjoint-slice allocator. |
| `network_dim` (rank) | 32 | Per-Linear rank `r`. Slice width per expert is roughly `min(out, in) / E` — keep `K_c + K_f ≤ network_dim / 4` so each expert gets a meaningful subspace (proposal §"OrthoHydra slice allocation"). |
| `network_alpha` | 32 | LoRA scale `α / r = 1` by default. |
| `balance_loss_weight` | `1.0` | **Outer** balance multiplier (warmup-gated). At `1.0` the per-pool weights below are the only effective scalars on the balance term. Set to `0` to disable balance loss entirely (use only for ablation — the proposal warns about one-pool collapse without per-pool pressure). |
| `balance_w_content` | `2e-5` | `w_c` in `L_balance = w_c · switch(π_c) + w_f · switch(π_f)`. Matches the OrthoHydra production value (`[[project_hydra_balance_weight_ceiling]]`). |
| `balance_w_freq` | `2e-5` | `w_f`. Starts at the same value as `w_c`; raise if the freq pool collapses to uniform during the first 1k steps. |
| `fei_feature_dim` | `2` | FEI simplex bands (`e_low, e_high`). Same as the FEI-on-Hydra default. |
| `fei_sigma_low_div` | `4.0` | `σ_low = min(H_lat, W_lat) / div` for the DoG kernel. 2026-05-13 dataset sweep picked 4 over 8 (see [[project_fera_probe_2band_decision]]). |
| `sigma_feature_dim` | `16` | Width of the sinusoidal-σ slice fed to the FreqRouter (same functional form as the DiT t_embedder). Combined with FEI = 18-dim router input. |
| `freq_router_init_std` | `0.1` | `N(0, std)` on the FreqRouter's output Linear weights. **Non-zero is load-bearing** — zero-init would make the freq pool a fixed point of the additive composition (uniform gates ⇒ no gradient signal on the router weights). See `chimera.py::FreqRouter` docstring. |
| `router_hidden_dim` | `64` | FreqRouter MLP hidden width. Shared with `GlobalRouter` (FeRA), no chimera-specific knob. |
| `router_tau` | `0.7` | Softmax temperature on the FreqRouter output. Lower → sharper freq-pool specialization. |
| `network_router_lr_scale` | `1.0` | Multiplier on `unet_lr` for the per-layer content router AND the network-level FreqRouter. |
| `use_ortho` | `true` | Cayley-rotated SVD basis. Implicit in the chimera module; kept for the unrouted-fallback Linears (router_targets-excluded → OrthoLoRAExp). |
| `use_timestep_mask` | `true` | T-LoRA. Applied to the content branch only inside the chimera forward. |
| `min_rank` | `8` | T-LoRA floor — content branch retains at least this many ranks at every t. |
| `router_targets` | `.*(mlp\\.layer[12])$` | Regex matching which Linears become chimera leaves. Non-matching layers fall back to OrthoLoRAExp (single-pool Cayley, no router). |

## Save format

`output/ckpt/<output_name>_chimera.safetensors` keys (`output_name = "anima_chimera"` by default):

```
# Network-level FreqRouter — fp32
freq_router.net.0.weight                              (router_hidden_dim, fei_feature_dim + sigma_feature_dim)
freq_router.net.0.bias                                (router_hidden_dim,)
freq_router.net.2.weight                              (num_experts_freq, router_hidden_dim)
freq_router.net.2.bias                                (num_experts_freq,)

# Per-adapted-Linear distilled Hydra-MoE keys (q/k/v defused on attention prefixes)
lora_unet_<dotted_path>.lora_down.weight              (r, in)         shared across experts
lora_unet_<dotted_path>.lora_ups.0.weight             (out, r)        content expert 0
lora_unet_<dotted_path>.lora_ups.1.weight             (out, r)        content expert 1
...
lora_unet_<dotted_path>.lora_ups.{K_c-1}.weight       (out, r)        content expert K_c-1
lora_unet_<dotted_path>.lora_ups.{K_c}.weight         (out, r)        freq expert 0
...
lora_unet_<dotted_path>.lora_ups.{K_c+K_f-1}.weight   (out, r)        freq expert K_f-1
lora_unet_<dotted_path>.router.weight                 (K_c, r)        content router (K_c-narrowed)
lora_unet_<dotted_path>.router.bias                   (K_c,)
lora_unet_<dotted_path>.alpha                         ()
```

The expert axis runs `[content_0 … content_{K_c-1} | freq_0 … freq_{K_f-1}]` (content first, freq second). The on-disk layout matches the existing HydraLoRA MoE keyspace exactly — the only chimera-specific bits are (a) the K_c-narrowed `router` (vs. Hydra's E-wide router), (b) the top-level `freq_router.*` block, and (c) the `ss_use_chimera_hydra` metadata stamp.

**Distilled at save, dual-pool at load.** Save runs `_convert_ortho_hydra_to_hydra` to fold the Cayley `(S_p, S_q, P_bases, Q_basis, lambda_layer)` into shared `lora_down.weight` + per-expert `lora_ups.{i}.weight`, then `_build_hydra_moe_state_dict` defuses fused `qkv_proj` / `kv_proj` prefixes per-component (cloning the shared `lora_down`, `alpha`, `router.*` into each q/k/v split). Top-level `freq_router.*` passes through both steps untouched (neither matches `.S_p` / `.lora_up_weight` / a fused attention prefix). The chimera-specific save variant is `chimera_hydra_moe` in `networks/lora_save.py`.

Load (`library/inference/models.py::_is_chimera_moe`) sniffs `ss_use_chimera_hydra="true"` from metadata, then `networks/lora_anima/factory.py` overrides `module_class = HydraLoRAModule` (instead of the Cayley `ChimeraHydraLoRAExpModule`) and passes `num_experts_content = K_c` through to the module construction loop. `HydraLoRAModule` narrows its router to K_c outputs and registers a `_freq_routing_weights` buffer of size K_f; `_compute_gate` cats `[π_c | π_f]` to recover the full (B, E) gate. The Cayley class is therefore **training-only** — checkpoint resume silently drops the orthogonal parameterization and continues on `HydraLoRAModule` (matches the OrthoHydra → Hydra precedent in `_convert_ortho_hydra_to_hydra`).

**Metadata stamps:**

```
ss_network_spec                = "chimera_hydra"
ss_use_chimera_hydra           = "true"
ss_num_experts_content         = "3"
ss_num_experts_freq            = "3"
ss_chimera_fei_feature_dim     = "2"
ss_chimera_sigma_feature_dim   = "16"
ss_chimera_fei_sigma_low_div   = "4.0"
ss_use_moe_style               = "shared_A"
ss_route_per_layer             = "true"
ss_router_source               = "input"
```

The three-axis stamps fire under the standard MoE branch (chimera pins `("shared_A", True, "input")`), so the loader picks `hydra` via the key-sniff `.lora_up_weight` (post-distillation, 3-D after `_stack_lora_ups`) → `has_hydra=True`, then the `ss_use_chimera_hydra` override re-targets `module_class` to `HydraLoRAModule` with `num_experts_content > 0` and the network attaches a `FreqRouter` (rebuilt fresh, weights loaded from disk).

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `apply_router_conditioning` fires `set_sigma` → `set_fei` every step; chimera force-enables `use_fei_router` so the FEI/σ pipeline runs unconditionally. The FreqRouter executes with grad so `∂L_denoise/∂π_f` reaches its parameters via the slot-assigned `_freq_routing_weights` buffer (same contract as FeRA's GlobalRouter). |
| Standard inference | ✅ | `library/inference/models.py::_is_chimera_moe` routes the file through the Hydra-mode dynamic-hook path; the factory rebuilds a HydraLoRAModule(num_experts_content > 0) + FreqRouter, and `library/inference/adapters.py::compute_and_set_hydra_fei` fires `set_fei` per Euler step. |
| Spectrum inference | ⚠ | Per-step `set_fei` is wired, but on a Spectrum cached step the FEI/gate is updated while the cached features may have been forecast from a different gate distribution. Bench against `--spectrum` before relying on it. Same caveat as FeRA. |
| `torch.compile` | ✅ | Two bmm + the OrthoHydra batched Cayley solve. Shape-static under constant-token bucketing. The chimera forward issues two bmm calls regardless of mask state so dynamo doesn't recompile at the T-LoRA flip points. |
| `blocks_to_swap` | ✅ | Each chimera module replaces its base Linear in-place; block swap moves it with its parameters. The network-level FreqRouter stays on the main device. |
| `gradient_checkpointing` | ✅ | The adapter is a thin Linear-replacement; checkpointing at block granularity wraps it correctly. |
| Modulation guidance | ✅ orthogonal | AdaLN path is untouched. |
| T-LoRA | ✅ | Built-in (per-branch masking). |
| OrthoLoRA / ReFT | ⚠ partial | `use_ortho=true` is the chimera default. ReFT is designed against shared-A / plain-LoRA layouts; verify on a small bench before stacking. |
| DCW (scalar / v4) | ✅ orthogonal | Sampler-level correction; composes with anything upstream of the Euler step. |
| ComfyUI | ⚠ partial | The save now produces the Hydra-MoE on-disk layout the `comfyui-hydralora` node already understands (shared `lora_down` + `lora_ups.{i}` + q/k/v split + per-Linear `router.*`). What's still missing from the node: (a) reading `ss_use_chimera_hydra` + `ss_num_experts_content` to narrow the router to K_c outputs, (b) loading the top-level `freq_router.*` block and broadcasting `π_f` per step. Estimated ~60 lines of node-side work. |
| Static merge into DiT | ❌ | `scripts/merge_to_dit.py` refuses MoE methods by default (the router is sample-dependent and can't be folded into Linear weights). `--allow-partial` would drop the chimera portion entirely. |
| FeRA / hydra-moe loaded simultaneously | ❌ | One router scheme per checkpoint; `models.py` refuses two moe files in one `--lora_weight` list. |

## Cold-start risk and diagnostics

Two routers init random ⇒ risk one pool dominates while the other settles at uniform (local minimum). Three structural mitigations are built in:

1. **Per-pool balance loss** (`w_c`, `w_f` independent). A single combined balance term would let the optimizer satisfy the constraint by flattening one pool to uniform while concentrating the other.
2. **Non-zero FreqRouter init** (`freq_router_init_std=0.1`). Output is near-uniform but *not at* uniform at step 0; the freq router immediately differentiates as FEI/σ vary across the batch.
3. **Live diagnostic.** Watch per-pool `Σ‖π[k] − 1/K‖²` in the first 1k steps. If freq pool stays < 1e-3 (flat-uniform) while content diverges, raise `balance_w_freq` or warm-start with a FEI residual.

Persistent freq-pool flatness after warmup ⇒ the freq router has no signal the content router didn't already capture via `lx`-σ correlation. That's the redundancy failure mode (proposal §Risks #1 — the bench plan calls out a `C-fei` falsification cell that feeds FEI into the content router; if `C-fei ≈ ChimeraHydra`, the freq pool is redundant).

## What to measure

ChimeraHydra's bet: structurally enforced input separation makes the freq pool learn `σ`-aware refinement the content router can't, *without* phased training. The whole point hinges on whether (a) the freq pool actually trains and (b) it picks up signal the single-router OrthoHydra was leaving on the table.

1. **Per-pool gate entropy + divergence.** Median across chimera Linears. Both pools should diverge from uniform after warmup; freq pool diverging on σ-buckets is the load-bearing signal.
2. **Freq-gate variance across σ buckets at inference.** Bin σ ∈ [0,1] into 3–5 buckets; for each bucket, log mean π_f. Variance across buckets > 0.01 → freq router is using σ. Below floor ⇒ freq pool is dead weight.
3. **Branch contribution norms.** `‖Δy_c‖` vs. `‖Δy_f‖` per Linear, averaged over the dataset. Healthy ratio ~0.3–3.0; out-of-range = one pool dominating.
4. **Per-expert usage histograms per pool.** Argmax frequency across the K_c content experts and the K_f freq experts. Flat-ish distributions in each pool are the success case; one column near zero = collapse.
5. **A/B vs single-router OrthoHydra (FEI-on-Hydra in `configs/methods/lora.toml` default) at matched E.** Same dataset, matched epochs/lr, `num_experts = K_c + K_f = E`. Whichever wins on CMMD + sample quality tells us whether the dual-pool structural separation buys anything over a single FEI router on the same expert count.
6. **C-fei falsification.** Feed FEI into the content router (one cfg toggle on a separate run): if results match ChimeraHydra, the freq pool is redundant and the dual-pool design should be archived.
7. **Sample quality vs `make lora`.** CMMD ([[project_cmmd_val_signal]]) is the primary signal; FM val-MSE is uninformative on Anima ([[project_fm_val_loss_uninformative]]).

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `num_experts_content` (`K_c`) | 3 | 2, 3, 4 | Parity with today's E=6; `K_c=4, K_f=2` gives more content capacity (proposal C-split cell). |
| `num_experts_freq` (`K_f`) | 3 | 2, 3, 4 | `K_c=2, K_f=4` emphasizes freq routing. Watch K_f=3 collapsing to K_f=2 — FEI-on-Anima is bimodal ([[project_fera_probe_2band_decision]]). |
| `balance_w_content` | `2e-5` | `1e-6` … `5e-5` | Same safe range as the single-pool ortho-hydra ceiling ([[project_hydra_balance_weight_ceiling]]). |
| `balance_w_freq` | `2e-5` | `1e-6` … `5e-5` | Raise if freq pool stays uniform after warmup. |
| `freq_router_init_std` | `0.1` | `0.05`, `0.1`, `0.3` | Higher → freq pool starts further from uniform but signal-to-noise drops. **Never zero** (fixed point). |
| `router_tau` (FreqRouter) | `0.7` | `0.3`, `0.7`, `1.0`, `2.0` | Lower τ → sharper freq specialization, more sensitive to FEI noise. |
| `sigma_feature_dim` | `16` | `8`, `16`, `32` | Higher → richer φ(σ) representation feeding the freq router; trades against router-input dim. |
| `fei_sigma_low_div` | `4.0` | `2`, `4`, `8` | Same Pareto region as FEI-on-Hydra; 4 picked by the 2026-05-13 dataset sweep. |
| `network_dim` | 32 | 16, 32, 64 | Slice width per expert = `min(out, in) / E`. At `network_dim=16, E=6` slices get narrow — verify expressivity vs single-pool at matched total rank. |
| `multiplier` (inference) | 1.0 | 0.0, 0.5, 1.0, 1.5 | `0.0` short-circuits to frozen base for clean ablation. |

## Files

- [`networks/lora_modules/chimera.py`](../../networks/lora_modules/chimera.py) — `ChimeraHydraLoRAExpModule`.
- [`networks/lora_anima/network.py`](../../networks/lora_anima/network.py) — `FreqRouter`, `_wire_shared_freq_routing_buffers`, `set_freq_routing_weights`, `_get_chimera_balance_loss`, FreqRouter param group.
- [`networks/lora_anima/config.py`](../../networks/lora_anima/config.py) — chimera cfg fields + three-axis pin.
- [`networks/lora_anima/factory.py`](../../networks/lora_anima/factory.py) — chimera stamp detection at reload.
- [`networks/lora_save.py`](../../networks/lora_save.py) — `chimera_hydra_moe` save variant: Cayley → distilled Hydra-MoE layout + q/k/v defuse, with top-level `freq_router.*` passed through.
- [`networks/__init__.py`](../../networks/__init__.py) — `NETWORK_REGISTRY["chimera_hydra"]`, `resolve_network_spec` dispatch, `_post_init_hydra` per-pool stamping.
- [`configs/methods/chimera.toml`](../../configs/methods/chimera.toml) — canonical method config (`make exp-chimera`).
- [`configs/gui-methods/chimera_hydra.toml`](../../configs/gui-methods/chimera_hydra.toml) — GUI-friendly variant config.
- [`scripts/experimental_tasks/training.py`](../../scripts/experimental_tasks/training.py) — `cmd_chimera` shim.
- [`docs/proposal/chimera_hydra.md`](../proposal/chimera_hydra.md) — design rationale, bench plan, decision tree, risks.

## Status

**Experimental.** Code lands and round-trip is verified, but no bench results yet. ComfyUI mirror is not built — chimera checkpoints are reloadable only via the in-tree training/inference path. The proposal's bench plan (cells A / B / C / C+T / C-split / C-fei) is the prerequisite before promoting chimera to a default LoRA-family variant.

See [`docs/proposal/chimera_hydra.md`](../proposal/chimera_hydra.md) §"Decision tree" for the ship/archive criteria after bench results land.
