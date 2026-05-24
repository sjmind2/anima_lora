# Self-Flow feasibility probe

Pre-flight falsification for porting **Self-Flow** (Chefer, Esser et al.,
*Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*, BFL,
arXiv 2603.06507) into Anima's **frozen-DiT LoRA fine-tuning** regime.

Run `probe.py` *before* building any training path. It measures whether the
two necessary conditions for the method exist on Anima's frozen backbone and
our actual dataset. If they don't, the training A/B below is dead on arrival.

---

## What Self-Flow is

Self-Flow is a *from-scratch / pretraining-scale* training objective for
flow-matching DiTs. Its thesis: external representation alignment
(REPA-style — borrowing features from a frozen DINO/SigLIP encoder) is
fundamentally limited (it scales *inversely* — stronger encoders hurt — and
harms video/audio). So instead of importing representations, **learn them
inside the generative model**. Two mechanisms:

1. **Dual-Timestep Scheduling (DTS).** Sample two timesteps `t, s ~ p(t)`,
   pick a token mask `M` (ratio ≤ 0.5), and noise each token to *either* `t`
   or `s`:

   ```
   τ_i = s   if i ∈ M
         t   otherwise
   x_τ = diag(1−τ)·x_0 + diag(τ)·x_1
   ```

   Heterogeneous per-token corruption forces the model to infer the noisier
   tokens from the cleaner ones → it must build *global* relations instead of
   solving each token by local correlation. The marginal per-token noise
   distribution is preserved, so it doesn't break the flow dynamics. The paper
   reports DTS *alone* slightly improves vanilla flow matching (Fig 11b).

2. **Self-Flow proper.** Keep a student `f_θ` and an EMA teacher `f_θ'`. The
   teacher sees the cleaner view `x_{τ_min}` (noised uniformly at
   `τ_min = min{t,s}`); the student sees the mixed view `x_τ`. A representation
   loss aligns an *early* student layer `l` to a *later* teacher layer `k`
   (`l < k`) via cosine similarity, through an MLP head `h`:

   ```
   L_rep = −E cos( h_θ^(l)(x_τ, τ),  sg[ f_θ'^(k)(x_{τ_min}, τ_min) ] )
   L     = L_gen + γ · L_rep
   ```

   `h` is just `MLP(features)` — a small projection head on the student's
   layer-`l` hidden state. `sg` = stop-gradient onto the teacher.

Reported gains (at 200K–1M steps, full-backbone training): better text
rendering, structural coherence (faces/hands), temporal consistency, and
*positive* scaling where REPA plateaus.

## How it could benefit Anima — and the catch

We don't pretrain the backbone; we fine-tune **rank-r LoRA on a frozen DiT**
over a small dataset (e.g. ~80 artists / 2.5k images). The plausible upside in
that regime is **cross-concept disentanglement and structural/text coherence**
on a multi-concept library — exactly Self-Flow's qualitative wins. The
proposed cheap adaptation that makes it affordable here:

- **EMA-LoRA teacher**, not an EMA DiT: student = `base + LoRA(θ)`,
  teacher = `base + LoRA(EMA θ')`, frozen base shared. Only the adapter is
  EMA'd → trivial memory.
- **Reuse the functional-loss hooks** (`blocks[l].cross_attn.output_proj` /
  block output) to tap `f^(l)` / `f^(k)`; `h` is one new tiny MLP module; a
  new `selfflow_rep` entry in the loss registry.

The catch — and the reason for this probe — is structural: Self-Flow improves
*backbone* representations, but our backbone is frozen. The rep loss can only
move a rank-r delta, and the frozen base may **already** satisfy the alignment
target (so the adapter gets ~no gradient), and may be robust enough that the
clean-vs-noisy input gap leaves **no asymmetry** to learn from. Both are
empirical and cheap to check.

> Block-swap note: any real training run that adds the teacher's second
> forward must use `blocks_to_swap=0` — the offloader desyncs on a 2nd DiT
> forward (see `project_blockswap_extra_forwards_gradcache`).

---

## The probe — two necessary-condition metrics

`probe.py` loads cached `(latent, T5)` pairs and, on the **frozen base DiT**
(no adapter), constructs two views of each latent from a sorted timestep pair
`(τ_lo, τ_hi) = sort(t, s)` sharing one noise sample `ε` (so the *only*
difference is the noise *level*):

- **teacher view** `x_{τ_lo}` at conditioning `τ_lo` (the cleaner view),
- **student view** `x_{τ_hi}` at conditioning `τ_hi` (the noisier view).

Both are in-distribution (conditioning matches the input level). It taps the
residual-stream output of block `l` and block `k` via forward hooks and
reports:

1. **Information asymmetry** — `asym_k = median cos(f^k(student), f^k(teacher))`.
   This is the *core premise* of DTS: the cleaner view must carry layer-`k`
   features the noisier view doesn't. **If `asym_k ≈ 1`** (the frozen DiT has
   already collapsed the noise-level gap by layer `k`), there is nothing to
   infer → DTS and the rep loss are inert. **Make-or-break number.**

2. **Alignment-target non-triviality (headroom)** — train *only* the MLP head
   `h` (backbone frozen, no LoRA) to maximize `cos(h(f^l(student)), f^k(teacher))`
   on held-out tokens. **If `headroom ≈ 1`**, the target is satisfiable by the
   head alone → the rep loss exerts ~no pressure on the *adapter* → the method
   collapses to DTS-only. We want a *moderate* ceiling (room the adapter could
   move into), and `headroom` should lift meaningfully over the no-head floor
   `raw_cos` = `cos(f^l(student), f^k(teacher))`.

This probe tests **necessary, not sufficient** conditions. Passing means
"worth training"; the real test is the CMMD A/B below.

### Verdict
- `NO-ASYMMETRY` — `asym_k` ≥ 0.97: DTS gives the frozen backbone nothing to
  infer. Shelve.
- `TRIVIAL` — `headroom` ≥ 0.95: rep loss won't pressure the adapter; at best
  you get DTS-only. Consider testing DTS alone, skip the teacher path.
- `VIABLE` — asymmetry present (`asym_k` < 0.97) **and** non-trivial headroom
  (0.30 ≤ `headroom` < 0.95) with a clear lift over `raw_cos`: proceed to the
  training A/B.
- `MARGINAL` — anything else (e.g. `headroom` < 0.30: `l`→`k` barely
  predictable at all — the loss is likely just noise).

### Run
```bash
python bench/selfflow/probe.py --dit /path/to/anima-dit.safetensors \
    --num_samples 12 --num_timesteps 8 --layer_l 6 --layer_k 18 --label first
# results → bench/selfflow/results/<YYYYMMDD-HHMM>-first/{result.json,per_pair.csv}
```

---

## If the probe passes — the training A/B (the real test)

Identical LoRA runs on the existing 80-artist / 2.5k set, same seed/preset,
`validation_split_num` carving held-out val, **`blocks_to_swap=0`**, judged on
**CMMD** (`val/loss_average`, lower better — FM-MSE doesn't track quality here;
see `project_cmmd_val_signal`):

| Run | Objective | Isolates |
|-----|-----------|----------|
| (a) | `flow_match` baseline | control |
| (b) | + DTS only (per-token τ, no teacher, no 2nd forward) | does heterogeneous noise alone help? (runs even under block swap) |
| (c) | + DTS + `selfflow_rep` (EMA-LoRA teacher, `h` MLP, cosine `l<k`) | the full method |
| (d) | + `selfflow_rep` without DTS masking | is the masking load-bearing? (paper Fig 11a says yes) |

Watch in (c): **collapse** (EMA-LoRA teacher ≈ student → monitor `h`-output
feature norms / the cosine target not saturating to a constant) and **γ
sensitivity** (sweep the rep weight small first).
