# Design: Usage Guides for LoKR / LoHA / CAME / Workflow / Bucket Families

**Date:** 2026-05-30
**Status:** Approved

## Summary

Add five English usage guide documents under `docs/guidelines/` and update `README.md` with brief descriptions and links. These features (LoKR, LoHA, CAME optimizer, Workflow engine, Bucket Families) are implemented in the fork but currently undocumented in user-facing guides.

## Deliverables

### 1. New Documents (5 files, all in `docs/guidelines/`, all in English)

| # | File | Topic |
|---|------|-------|
| 1 | `docs/guidelines/lokr.md` | LoKR (Low-Rank Kronecker Product) usage guide |
| 2 | `docs/guidelines/loha.md` | LoHA (Low-Rank Hadamard Product) usage guide |
| 3 | `docs/guidelines/came.md` | CAME factorized optimizer usage guide |
| 4 | `docs/guidelines/bucket-families.md` | Bucket Families resolution bucketing guide |
| 5 | `docs/guidelines/workflow.md` | Workflow engine usage guide (including install) |

### 2. README.md Update

Insert a "What's new" section after the "Four things" paragraph (after line 13), before the `---` separator. Contains a table with one row per feature: name, one-line description, link to the guide. Existing install instructions are NOT modified.

---

## Document Designs

### 1. `docs/guidelines/lokr.md`

**Audience:** Users who want to train with LoKR instead of standard LoRA.

**Structure:**
1. Overview — Kronecker product factorization, structured high-rank approximation, how it differs from LoRA
2. Quick start — CLI (`network_type = "lokr"`) and GUI (method dropdown) usage
3. Configuration parameters — `network_dim`, `network_alpha`, `lokr_factor`, `decompose_both`, `scale_weight_norms`, `weight_decompose`, `use_scalar`, with default values and recommended ranges
4. Anima-specific notes — QKV fusion layer `lokr_factor` divisibility constraint (multiples of 3 or -1), no Conv2d so `conv_dim`/`conv_alpha`/`use_tucker` are inert
5. Inference — CLI auto-detection, ComfyUI LyCORIS loader compatibility
6. Compatibility matrix — what LoKR can/cannot stack with (no OrthoLoRA, no HydraLoRA; works with T-LoRA, Spectrum, Mod guidance, ReFT, P-GRAFT)
7. Recommended configs — for small datasets (dim=8, factor=-1, decompose_both=true), for large datasets (dim=16, factor=12)
8. Cross-reference — link to `docs/methods/lycoris-variants.md` for mathematical details

**Source material:** `networks/lora_modules/lokr.py`, `configs/methods/lokr.toml`, `configs/gui-methods/lokr.toml`, `docs/methods/lycoris-variants.md`

### 2. `docs/guidelines/loha.md`

**Audience:** Users who want to train with LoHA for higher effective rank.

**Structure:**
1. Overview — Hadamard product of two low-rank matrices, effective rank r² with ~2× LoRA params
2. Quick start — CLI and GUI usage
3. Configuration parameters — `network_dim`, `network_alpha`, `scale_weight_norms`, with emphasis on scale = alpha/dim and recommended range 0.1–0.5
4. Comparison with LoKR — effective rank, parameter count, DoRA support, autograd approach
5. Inference — same as LoKR (auto-detection by key prefix)
6. Compatibility — same mutual exclusivity as LoKR
7. Recommended configs — dim=32 alpha=16 (scale=0.5) as starting point
8. Cross-reference — link to `docs/methods/lycoris-variants.md`

**Source material:** `networks/lora_modules/loha.py`, `configs/methods/loha.toml`, `docs/methods/lycoris-variants.md`

### 3. `docs/guidelines/came.md`

**Audience:** Users who want to use CAME optimizer to save memory, especially with LyCORIS variants.

**Structure:**
1. Overview — factorized optimizer replacing full-matrix second moments with row/column moments
2. Why CAME — memory savings (no per-element `exp_avg_sq`), well-suited for LyCORIS factorized modules, default in Workflow
3. Usage — set `optimizer_type = "CAME"` in TOML config or Workflow; `optimizer_args` for betas/eps
4. Parameters — `betas` (default 0.9, 0.999, 0.9999), `eps` (1e-30, 1e-16), `clip_threshold` (1.0)
5. Comparison with other optimizers — table: AdamW / CAME / Adopt_Adv / Prodigy_Adv (memory, auto-lr, stability)
6. Recommended LR — 0.5–0.9× of equivalent AdamW learning rate
7. Known limitations — factored mode for 2D+ params only (1D falls back to Adam-style); `torch.compile` compatible but group stacking may need tuning
8. Cross-reference — link to `docs/optimizations/adv_optm_guide.md` for other optimizer options

**Source material:** `library/training/came_optimizer.py`, `library/training/optimizers.py`, `docs/optimizations/adv_optm_guide.md`

### 4. `docs/guidelines/bucket-families.md`

**Audience:** Users who want to understand and control resolution bucketing for training.

**Structure:**
1. Overview — what bucket families are; token count as the core concept; the balance between aspect-ratio diversity and compile performance
2. Two-step matching algorithm
   - Step 1: Area matching → select family (`|tc × 256 - image_area|` minimized)
   - Step 2: AR matching → select specific bucket within family (min AR difference, tie-break by area)
   - Worked example: 800×800 image → S family → specific bucket
3. Resize and crop — isotropic scale to cover bucket (Lanczos), center-crop to exact (W, H); less crop when image area ≈ tc × 256
4. Seven families table — Family, TC, standard pixel area, member resolutions, typical use case
5. Token count and performance
   - Low TC = faster training, less VRAM (fewer compiled graphs, smaller tensors)
   - Low TC = detail loss (fewer pixels seen by model)
   - S1 (1024 tokens, 0.26MP) vs L (4032 tokens, 1.03MP) concrete comparison
   - How `torch.compile` groups buckets by token count via `_native_flatten`
6. Multi-stage training strategy (links to Workflow guide)
   - Two-stage: low TC family (e.g., S1) for base → high TC family (e.g., S2/L) for detail refinement
   - How Workflow supports this: multiple Preprocess stages with different families → multiple Train stages with checkpoint continuation
   - Benefits: reduced total compute for large datasets; balance speed and detail
   - Risks: low-res features may interfere at high-res stage, potentially degrading final quality
   - Link to `docs/guidelines/workflow.md` multi-stage section
7. CLI usage — `--bucket_families` flag, dataset distribution analysis

**Source material:** `library/datasets/buckets.py`, `library/preprocess/images.py`, `scripts/preprocess/resize_images.py`, `library/anima/models.py` (compile_blocks)

### 5. `docs/guidelines/workflow.md`

**Audience:** Users and developers who want to use or contribute to the Workflow engine.

**Structure:**
1. Overview — WebUI + CLI automated multi-stage training pipeline; aiohttp + Vue 3 CDN; schema-driven dynamic forms
2. Installation (development environment)
   - Python dependencies: `uv sync` (includes aiohttp, pywebview)
   - Node.js: required for frontend development (building/testing JS changes)
   - pywebview system dependency: Windows WebView2 Runtime (usually pre-installed on Windows 10/11)
   - Launching from source: `python -m workflow [--no-gui] [--port 8765]`
3. Quick start: single-stage training example
   - Launch Workflow → create new workflow
   - Add Preprocess stage: select `image_dataset/` directory, choose bucket family (e.g., L)
   - Add Train stage: select method (e.g., LoKR), configure params, reference preprocess output
   - Run → wait for completion
   - **Where training artifacts live**: `runs/latest/{stage_id}/output/*.safetensors`; `runs/latest` is a junction/symlink to the most recent run
4. Single-stage usage in detail
   - Preprocess stage: data source, bucket_families selection, min_pixels threshold
   - Train stage: method selector, schema-driven parameter form, dataset reference
   - Output structure
5. Multi-stage usage in detail
   - Multi-stage orchestration: topological sort, `depends_on`
   - Multiple Preprocess stages: different bucket_families per stage (e.g., S1 for base, S2 for refinement); different data sources
   - Multiple Train stages:
     - `stop_epoch`: interrupts training at specified epoch, ensures checkpoint is saved at that epoch
     - Checkpoint continuation: next Train stage auto-references previous stage's `safetensors_path` as `network_weights`
     - `dim_from_weights`: LoRA auto-infers rank from weights; LyCORIS (lokr/loha/locon) does not auto-infer
     - Placeholder references: `${train_1.safetensors_path}` cross-stage output
   - Typical flow: Preprocess S1 → Train S1 (stop at epoch 6) → Preprocess S2 → Train S2 (from S1 checkpoint, S1+S2 caches)
6. Log viewer
   - Three tabs: system log, script output, run history
   - TQDM progress bar parsing in script output
   - Stage-level filtering
   - Finding latest training artifacts from history: `runs/latest` junction + "Open directory" button in history records
   - Search and highlight
7. Settings
   - Language: auto-detect from browser, supports zh-CN / en / ja, manual switch
   - Model settings: DiT / Qwen3 TE / VAE paths (global defaults + per-workflow override)
   - Hardware: `mixed_precision` (bf16), `attn_mode` (flex)
   - Override priority: infra defaults → infra config → stage config → auto-derived

**Source material:** `workflow/` module (all files), `workflow/schemas/`, `workflow/web/`

---

## README.md Modification

Insert after line 13 (end of "Four things" list), before the `---` separator on line 16:

```markdown

## What's new

| Feature | Description | Guide |
|---------|-------------|-------|
| **LoKR** | Low-rank Kronecker product adaptation — structured high-rank with adaptive parameter count | [docs/guidelines/lokr.md](docs/guidelines/lokr.md) |
| **LoHA** | Low-rank Hadamard product adaptation — effective rank r² with only 2× LoRA parameters | [docs/guidelines/loha.md](docs/guidelines/loha.md) |
| **CAME optimizer** | Factorized optimizer replacing full-matrix second moments — significant memory savings | [docs/guidelines/came.md](docs/guidelines/came.md) |
| **Bucket Families** | Resolution bucketing by area → AR matching to token-count groups for compile performance | [docs/guidelines/bucket-families.md](docs/guidelines/bucket-families.md) |
| **Workflow engine** | WebUI + CLI multi-stage training pipeline with real-time progress and schema-driven forms | [docs/guidelines/workflow.md](docs/guidelines/workflow.md) |

```

No other changes to README.md. The existing install instructions remain untouched.

---

## Constraints

- All documents in English
- All documents in `docs/guidelines/`
- No modification to existing install instructions in README.md
- Workflow install section adds a new section (does not modify existing Setup section)
- Documents are user-facing usage guides (not internal design docs)
- Cross-references between bucket-families.md and workflow.md (multi-stage strategy)
- Cross-references to existing `docs/methods/lycoris-variants.md` for mathematical details
