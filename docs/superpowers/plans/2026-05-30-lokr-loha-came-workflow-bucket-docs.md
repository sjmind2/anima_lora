# Usage Guides: LoKR / LoHA / CAME / Workflow / Bucket Families — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five English usage guide documents under `docs/guidelines/` and update `README.md` with brief descriptions and links.

**Architecture:** Five independent markdown documents, each self-contained, with cross-references between bucket-families.md ↔ workflow.md and to existing docs/methods/lycoris-variants.md. One edit to README.md inserting a "What's new" table.

**Tech Stack:** Markdown documentation only. No code changes.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `docs/guidelines/lokr.md` | LoKR usage guide |
| Create | `docs/guidelines/loha.md` | LoHA usage guide |
| Create | `docs/guidelines/came.md` | CAME optimizer usage guide |
| Create | `docs/guidelines/bucket-families.md` | Bucket Families resolution bucketing guide |
| Create | `docs/guidelines/workflow.md` | Workflow engine usage guide |
| Modify | `README.md` (after line 13) | Insert "What's new" section |

---

### Task 1: Write `docs/guidelines/lokr.md`

**Files:**
- Create: `docs/guidelines/lokr.md`

- [ ] **Step 1: Create the LoKR usage guide**

Write the following content to `docs/guidelines/lokr.md`:

```markdown
# LoKR — Low-Rank Kronecker Product Adaptation

LoKR is a LyCORIS-family adapter that factorizes weight dimensions and composes them via the Kronecker product, producing a structured high-rank approximation whose parameter count depends on the factorization shape rather than a single rank value.

For the mathematical walkthrough (forward formulas, dimension analysis, weight key naming, scalar baking), see [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md). This guide covers usage and configuration.

## Quick Start

### CLI

Set `network_type = "lokr"` in your method TOML config:

```toml
network_type = "lokr"
network_dim  = 8
network_alpha = 8
```

Then train as usual:

```bash
make lora   # or: python train.py --method lokr
```

### GUI

Select **LoKR** from the method dropdown in the Anima GUI. The dedicated config `configs/gui-methods/lokr.toml` pre-fills all variant-specific parameters.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 8 | Rank for low-rank factors |
| `network_alpha` | 8 | LoRA alpha (scale = alpha / dim) |
| `decompose_both` | `true` | Decompose both W1 and W2 into low-rank pairs. Reduces parameter count significantly at small ranks. |
| `lokr_factor` | -1 | Target factor size for dimension factorization. `-1` = auto (recommended). |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |
| `weight_decompose` | `false` | Enable DoRA-style weight decomposition (LoKR only — not available for LoHA/LoCON) |
| `use_scalar` | `false` | Learnable scalar (zero-init) instead of fixed scalar=1 |
| `full_matrix` | `false` | Force full (non-decomposed) matrices |
| `conv_dim` | 4 | Rank for Conv2d layers (**inert on Anima** — DiT has no Conv2d) |
| `conv_alpha` | 4 | Alpha for Conv2d layers (**inert on Anima**) |
| `use_tucker` | `true` | Tucker core for Conv2d (**inert on Anima**) |

## Anima-Specific Notes

### QKV Fusion and `lokr_factor`

Anima-base-v1.0 DiT fuses Q/K/V into a single `qkv_proj` Linear layer with shape `[6144, 2048]`. When saving checkpoints for ComfyUI compatibility, this fused module must be split back into separate `q_proj`/`k_proj`/`v_proj`. The split requires `factorization(6144, factor)` to produce an `out_l` divisible by 3.

**Recommended `lokr_factor` values:**

| `lokr_factor` | Raw `out_l` | Adjusted `out_l` | Checkpoint size |
|---------------|------------|------------------|-----------------|
| **-1** (auto) | auto | auto | ~10 MB |
| **6** | 6 | 6 (no change) | ~13 MB |
| **12** | 12 | 12 (no change) | ~7 MB |
| **24** | 24 | 24 (no change) | ~4 MB |
| 4 | 4 | **3** (adjusted) | ~10 MB |
| 8 | 8 | **6** (adjusted) | ~13 MB |
| 16 | 16 | **12** (adjusted) | ~7 MB |

All values produce correctly-sized checkpoints. Using a factor that is a multiple of 3 (6, 12, 24) avoids the adjustment entirely. Use `lokr_factor = -1` (auto) for the most balanced factorization.

### No Conv2d in Anima DiT

All LoRA target layers in Anima-base-v1.0 DiT are `nn.Linear`. The `conv_dim`, `conv_alpha`, and `use_tucker` parameters are accepted without error but have no effect.

## Inference

### CLI — Static Merge

`inference.py` auto-detects LoKR by inspecting safetensors key prefixes (`lokr_*`). The weight delta is computed using the Kronecker product formula and merged into the base model weights before denoising.

LoKR checkpoints can coexist in the same `--lora_weight` list with regular LoRA files — each file is merged independently.

### ComfyUI

ComfyUI's LyCORIS loader node natively supports LoKR weight formats. The safetensors files produced by this trainer use the same key naming convention as the sd-scripts / LyCORIS ecosystem, so they are directly loadable without conversion.

## Compatibility

| Stacks with | Notes |
|-------------|-------|
| **T-LoRA** | Timestep rank masking applied after `make_weight` on the reconstructed weight |
| **Spectrum** | No interaction — cached steps skip blocks entirely |
| **Modulation guidance** | Orthogonal — touches AdaLN only |
| **ReFT** | Orthogonal side-channel |
| **P-GRAFT** | Cutoff step toggles `network.enabled` |
| **HydraLoRA** | **Not supported** — requires standard BA structure |
| **OrthoLoRA** | **Not supported** — Cayley re-parameterization defined for standard BA only |

## Recommended Configs

### Small datasets (≤ 20 images)

```toml
network_type = "lokr"
network_dim = 8
network_alpha = 8
decompose_both = true
lokr_factor = -1
scale_weight_norms = 1.0
learning_rate = 2e-5
max_train_epochs = 4
```

### Large datasets (100+ images)

```toml
network_type = "lokr"
network_dim = 16
network_alpha = 16
decompose_both = true
lokr_factor = 12
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 8
```
```

- [ ] **Step 2: Verify the file was created**

Run: `ls -la docs/guidelines/lokr.md`

- [ ] **Step 3: Commit**

```bash
git add docs/guidelines/lokr.md
git commit -m "docs: add LoKR usage guide"
```

---

### Task 2: Write `docs/guidelines/loha.md`

**Files:**
- Create: `docs/guidelines/loha.md`

- [ ] **Step 1: Create the LoHA usage guide**

Write the following content to `docs/guidelines/loha.md`:

```markdown
# LoHA — Low-Rank Hadamard Product Adaptation

LoHA is a LyCORIS-family adapter that uses the Hadamard (element-wise) product of two low-rank matrices to achieve effective rank r² with only ~2× the parameters of standard LoRA. Custom autograd functions provide exact gradients for the Hadamard product.

For the mathematical walkthrough (forward formulas, dimension analysis, weight key naming), see [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md). This guide covers usage and configuration.

## Quick Start

### CLI

Set `network_type = "loha"` in your method TOML config:

```toml
network_type = "loha"
network_dim  = 32
network_alpha = 16
```

Then train as usual:

```bash
make lora   # or: python train.py --method loha
```

### GUI

Select **LoHA** from the method dropdown in the Anima GUI. The dedicated config `configs/gui-methods/loha.toml` pre-fills all variant-specific parameters.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 32 | Rank r (effective rank ≈ r²) |
| `network_alpha` | 16 | LoRA alpha. **Scale = alpha / dim** — recommended range 0.1–0.5. |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |
| `conv_dim` | 4 | Rank for Conv2d layers (**inert on Anima** — DiT has no Conv2d) |
| `conv_alpha` | 1 | Alpha for Conv2d layers (**inert on Anima**) |
| `use_tucker` | `true` | Tucker mode for Conv2d (**inert on Anima**) |

### Understanding Scale

The scale factor `s = alpha / dim` controls the magnitude of the weight update ΔW. For LoHA:

- `dim=32, alpha=16` → scale = 0.5 (good starting point)
- `dim=32, alpha=8` → scale = 0.25 (more conservative)
- `dim=16, alpha=8` → scale = 0.5 (same scale, lower effective rank)

Recommended scale range: **0.1–0.5**. Higher scale = stronger adaptation but risk of instability.

## Comparison with LoKR

| Feature | LoHA | LoKR |
|---------|------|------|
| Core operation | Hadamard (element-wise) product | Kronecker product |
| Effective rank | r² | rank(W1) × rank(W2) (adaptive) |
| Params (same r) | ~2× LoRA | Adaptive; can be < LoRA |
| DoRA support | No | Yes (`weight_decompose=true`) |
| Custom autograd | `HadaWeight` / `HadaWeightTucker` | `KronLinearFn` / `KronLinearTwoStageFn` |
| Best for | Higher effective rank from a given r | Flexible parameter count with factor tuning |

## Inference

### CLI — Static Merge

`inference.py` auto-detects LoHA by inspecting safetensors key prefixes (`hada_*`). The weight delta is computed using the Hadamard product formula and merged into the base model weights.

LoHA checkpoints can coexist in the same `--lora_weight` list with regular LoRA or LoKR files.

### ComfyUI

ComfyUI's LyCORIS loader node natively supports LoHA weight formats. Directly loadable without conversion.

## Compatibility

Same mutual exclusivity rules as LoKR:

| Stacks with | Notes |
|-------------|-------|
| **T-LoRA** | Timestep rank masking applied after `make_weight` |
| **Spectrum** | No interaction |
| **Modulation guidance** | Orthogonal |
| **ReFT** | Orthogonal side-channel |
| **P-GRAFT** | Cutoff step toggles `network.enabled` |
| **HydraLoRA** | **Not supported** |
| **OrthoLoRA** | **Not supported** |

## Recommended Configs

### Standard training

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 4
```

### With T-LoRA and timestep masking

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
use_timestep_mask = true
min_rank = 8
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 6
```
```

- [ ] **Step 2: Verify the file was created**

Run: `ls -la docs/guidelines/loha.md`

- [ ] **Step 3: Commit**

```bash
git add docs/guidelines/loha.md
git commit -m "docs: add LoHA usage guide"
```

---

### Task 3: Write `docs/guidelines/came.md`

**Files:**
- Create: `docs/guidelines/came.md`

- [ ] **Step 1: Create the CAME optimizer usage guide**

Write the following content to `docs/guidelines/came.md`:

```markdown
# CAME Optimizer — Confidence-guided Adaptive Matrix Evaluation

CAME is a factorized optimizer that replaces the full-matrix second moment (`exp_avg_sq`) with row and column moments (`exp_avg_sq_row`, `exp_avg_sq_col`), dramatically reducing optimizer state memory for 2D+ parameters. It also adds a residual correction mechanism to compensate for the factorization approximation error.

CAME is the default optimizer in the [Workflow engine](workflow.md) and is particularly well-suited for LyCORIS variant training (LoKR, LoHA) where factorized modules have small but numerous parameter matrices.

## Why CAME

| Advantage | Details |
|-----------|---------|
| **Memory savings** | For a (d_out × d_in) weight, AdamW stores d_out × d_in values for `exp_avg_sq`. CAME stores d_out + d_in values (row + column moments) — a reduction of ~(d_out × d_in) / (d_out + d_in) ≈ min(d_out, d_in) ×. |
| **Factorized-friendly** | LyCORIS modules (LoKR, LoHA) produce many small 2D+ parameter tensors. CAME's row/column factorization is a natural fit. |
| **torch.compile compatible** | Group-stacked batch operations and optional `torch.compile` acceleration for CUDA tensors. |
| **Residual correction** | Extra `exp_avg_res_row`/`exp_avg_res_col` moments correct the factorization approximation error, improving convergence quality. |

## Usage

### CLI / TOML config

Set `optimizer_type` in your method or preset TOML:

```toml
optimizer_type = "CAME"
learning_rate = 1.5e-5
```

Pass optimizer-specific arguments via `optimizer_args`:

```toml
optimizer_type = "CAME"
optimizer_args = ["betas=0.9,0.999,0.9999"]
learning_rate = 1.5e-5
```

### Workflow

CAME is the default optimizer in Workflow schemas (`workflow/schemas/train_common.yaml`). No configuration needed — it's selected automatically unless you change it.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betas` | (0.9, 0.999, 0.9999) | (beta0 for exp_avg, beta1 for second moments, beta2 for residual moments) |
| `eps` | (1e-30, 1e-16) | (eps0 for update stability, eps1 for residual stability) |
| `clip_threshold` | 1.0 | RMS clipping threshold — prevents update explosion |
| `weight_decay` | 0.0 | Weight decay coefficient |
| `lr` | — | Learning rate. **Recommended: 0.5–0.9× of your equivalent AdamW learning rate.** |

### Internal behavior

- **2D+ parameters**: Factorized update with row/column second moments + residual correction. Same-shape parameters are stacked into batch tensors for efficiency.
- **1D parameters** (biases, scalars): Unfactored Adam-style update with full `exp_avg_sq`.

## Comparison with Other Optimizers

| Optimizer | Memory (per param) | Auto LR | Best for | Source |
|-----------|-------------------|---------|----------|--------|
| **AdamW** | 2 × full state | No | General purpose, default | `torch.optim` |
| **CAME** | ~2 × (row+col) state | No | LyCORIS variants, memory-constrained | Built-in (`library/training/came_optimizer.py`) |
| **Adopt_Adv** | Configurable (factored option) | No | Small-batch stability, long training | `adv_optm` package |
| **Prodigy_Adv** | 2 × full state + D-Adaptation state | **Yes** (set `lr=1.0`) | Unknown optimal LR | `adv_optm` package |

For the full advanced optimizer guide (Adopt_Adv, Prodigy_Adv), see [docs/optimizations/adv_optm_guide.md](../optimizations/adv_optm_guide.md).

## Recommended Learning Rates

CAME's factorized second-moment estimate behaves differently from AdamW's full estimate. In practice:

| Scenario | AdamW lr | CAME lr |
|----------|----------|---------|
| LoRA rank 32 | 2e-4 | 1–1.5e-4 |
| LoKR dim 8 | 2e-5 | 1.5e-5 |
| LoHA dim 32 | 1e-4 | 7e-5 |

Rule of thumb: **start with 0.7× of your AdamW learning rate** and adjust.

## Known Limitations

- **1D parameters** use unfactored (Adam-style) updates — no memory savings for biases and scalars.
- **torch.compile group stacking** groups same-shape parameters into batch tensors. When many distinct shapes exist, the grouping overhead may reduce the benefit. This is rarely an issue in practice.
- **Not compatible with** gradient accumulation factors > 1 without care — the per-step state update is correct, but the effective lr scaling may differ from AdamW.
```

- [ ] **Step 2: Verify the file was created**

Run: `ls -la docs/guidelines/came.md`

- [ ] **Step 3: Commit**

```bash
git add docs/guidelines/came.md
git commit -m "docs: add CAME optimizer usage guide"
```

---

### Task 4: Write `docs/guidelines/bucket-families.md`

**Files:**
- Create: `docs/guidelines/bucket-families.md`

- [ ] **Step 1: Create the Bucket Families guide**

Write the following content to `docs/guidelines/bucket-families.md`:

```markdown
# Bucket Families — Resolution Bucketing for Training

Bucket Families is a resolution bucketing system that groups training images by **token count** (number of patches). Each family contains multiple aspect ratios that all produce exactly the same token count, so `torch.compile` traces one block graph per family rather than per resolution.

## How Images Are Matched to Buckets

The matching is a two-step process: **area → family, then aspect ratio → bucket**.

### Step 1: Area Matching to Family

Each family has a **standard pixel area** = `tc × 256` (because each 16×16 patch covers 256 pixels). The system finds the family whose standard area is closest to the image's actual area:

```
best_family = argmin |tc × 256 - image_width × image_height|
```

**Example:** An 800×800 image (area = 640,000):

| Family | Token Count | Standard Area | \|Diff\| |
|--------|:-----------:|:------------:|:-------:|
| S1 | 1024 | 262,144 | 377,856 |
| XS | 1680 | 430,080 | 209,920 |
| **S** | **2160** | **552,960** | **87,040 ← closest** |
| M | 3600 | 921,600 | 281,600 |
| L | 4032 | 1,032,192 | 392,192 |
| S2 | 4096 | 1,048,576 | 408,576 |
| XL | 5040 | 1,290,240 | 650,240 |

→ The 800×800 image is assigned to the **S** family (tc=2160).

### Step 2: Aspect Ratio Matching to Bucket

Within the matched family, the system finds the bucket whose aspect ratio is closest to the image's:

```
best_bucket = argmin |bucket_AR - image_AR|    where AR = width / height
```

If two buckets tie (e.g., a portrait and landscape mirror), the one with the closest area is chosen.

Continuing the example — the S family has these members:

| Bucket (W×H) | AR |
|-------------|-----|
| 384×1440 | 0.27 |
| 432×1280 | 0.34 |
| 480×1152 | 0.42 |
| 576×960 | 0.60 |
| 640×864 | 0.74 |
| 720×768 | 0.94 |
| (+ landscape mirrors) | (> 1.0) |

An 800×800 image has AR = 1.0. The closest bucket is **720×768** (AR=0.94), or its landscape mirror **768×720** (AR=1.07). The system picks whichever AR is closer — in this case **720×768** (|0.94 - 1.0| = 0.06 vs |1.07 - 1.0| = 0.07).

### Resize and Crop

Once the bucket is selected:

1. **Isotropic scale** — the image is scaled (Lanczos interpolation) so it fully covers the bucket dimensions (no letterboxing).
2. **Center crop** — the scaled image is center-cropped to the bucket's exact (W, H).

When the image area is close to `tc × 256`, the crop is minimal. When the area is far from the standard area, more cropping occurs.

## The Seven Families

| Family | Token Count | Standard Area (px) | Member Resolutions (W×H) | Typical Use |
|--------|:-----------:|:------------------:|--------------------------|-------------|
| **S1** | 1024 | 262,144 (0.26 MP) | 256×1024, 512×512 | Fast prototyping, low-VRAM training |
| **XS** | 1680 | 430,080 (0.43 MP) | 336×1280, 384×1120, 448×960, 480×896, 560×768, 640×672 | Light training with moderate AR coverage |
| **S** | 2160 | 552,960 (0.55 MP) | 384×1440, 432×1280, 480×1152, 576×960, 640×864, 720×768 | Balanced speed and quality |
| **M** | 3600 | 921,600 (0.92 MP) | 480×1920, 576×1600, 640×1440, 720×1280, 768×1200, 800×1152, 960×960 | High quality, good AR coverage |
| **L** | 4032 | 1,032,192 (1.03 MP) | 512×2016, 576×1792, 672×1536, 768×1344, 896×1152, 1008×1024 | Default quality, dense AR coverage |
| **S2** | 4096 | 1,048,576 (1.05 MP) | 512×2048, 1024×1024 | High quality square and 1:2 AR |
| **XL** | 5040 | 1,290,240 (1.29 MP) | 640×2016, 672×1920, 720×1792, 768×1680, 896×1440, 960×1344, 1008×1280, 1120×1152 | Maximum quality |

Each family includes landscape mirrors (W and H swapped) automatically, doubling the available aspect ratios (except square buckets).

## Token Count and Performance

The DiT uses 16×16 patches, so **token count = (W ÷ 16) × (H ÷ 16)**. This is the number of patches the model processes per image.

### Why token count matters

When `torch.compile` is enabled, the compiler traces one block graph per distinct token count (via `_native_flatten`). Within a single family, all resolutions share one compiled graph — regardless of aspect ratio.

**Lower token count means:**

| Metric | S1 (1024 tokens) | L (4032 tokens) | Ratio |
|--------|:-----------------:|:----------------:|:-----:|
| Pixels per image | 262,144 | 1,032,192 | 1:3.9 |
| Patches per forward | 1,024 | 4,032 | 1:3.9 |
| VRAM per batch | Lower | Higher | — |
| Steps/second | Faster | Slower | ~2–4× |
| Visual detail preserved | Lower | Higher | — |

**Trade-off:** Low token count families (S1, XS, S) train significantly faster and use less VRAM, but the model sees fewer pixels per image — fine details and textures are lost. High token count families (L, S2, XL) preserve more detail but require more compute and VRAM.

### The zero-padding guarantee

Every bucket in a family **exactly** fills its token count by construction — there is no intra-bucket padding. This means:

- Flash Attention runs without padding masks (no attention leak)
- The compiled graph is bit-exact with the eager forward path
- No wasted computation on pad tokens

## Multi-Stage Training Strategy

A practical approach to balance speed and quality is **two-stage training** using the [Workflow engine](workflow.md):

### Concept

1. **Stage 1 — Low resolution** (e.g., S1 or S family): Train the adapter at low token count. The model learns overall composition, colors, and style at a fraction of the compute cost.
2. **Stage 2 — High resolution** (e.g., L or S2 family): Continue training from the Stage 1 checkpoint at high token count. The model refines details and textures.

### Benefits

- **Reduced total compute** for large datasets — the expensive high-resolution stage only needs a few epochs.
- **Speed-detail balance** — most of the learning happens cheaply at low resolution.
- **Creative effects** — some users prefer the stylized look produced by multi-resolution training.

### Risks

- **Quality degradation** — features learned at low resolution may not translate well to high resolution, potentially causing artifacts or reduced fidelity.
- **Overfitting risk** — if Stage 1 overfits at low resolution, Stage 2 may amplify those artifacts.

### How to set this up in Workflow

See the [Workflow guide — Multi-stage usage](workflow.md#multi-stage-usage) for a complete walkthrough. Briefly:

1. Add **Preprocess** stage with `bucket_families = "S1"` → processes images at low resolution
2. Add **Train** stage with `stop_epoch = 6` → trains and stops at epoch 6
3. Add **Preprocess** stage with `bucket_families = "L"` → processes images at high resolution
4. Add **Train** stage → automatically continues from the Stage 1 checkpoint, using both S1 and L caches

## CLI Usage

### Resize with a specific family

```bash
python scripts/preprocess/resize_images.py \
  --src image_dataset/ \
  --dst post_image_dataset/resized/ \
  --bucket_families "L" \
  --tree
```

### Multiple families

```bash
python scripts/preprocess/resize_images.py \
  --src image_dataset/ \
  --dst post_image_dataset/resized/ \
  --bucket_families "S,M,L" \
  --tree
```

### Dataset distribution analysis

Use the Workflow UI's "Analyze dataset" button on the Preprocess stage to see how your images distribute across families — both with all families available (natural distribution) and with only your selected families.
```

- [ ] **Step 2: Verify the file was created**

Run: `ls -la docs/guidelines/bucket-families.md`

- [ ] **Step 3: Commit**

```bash
git add docs/guidelines/bucket-families.md
git commit -m "docs: add Bucket Families resolution bucketing guide"
```

---

### Task 5: Write `docs/guidelines/workflow.md`

**Files:**
- Create: `docs/guidelines/workflow.md`

- [ ] **Step 1: Create the Workflow usage guide**

Write the following content to `docs/guidelines/workflow.md`:

```markdown
# Workflow Engine — Automated Multi-Stage Training

The Workflow engine is a WebUI + CLI automated training pipeline built on aiohttp (backend) and Vue 3 CDN (frontend). It supports configurable multi-stage training workflows with schema-driven dynamic forms, real-time progress feedback via SSE, and cross-stage checkpoint continuation.

## Installation

### Python Dependencies

All Python dependencies are included in `pyproject.toml`. Install with:

```bash
uv sync
```

Key dependencies:
- `aiohttp >= 3.13.5` — HTTP server and REST API
- `pywebview >= 5.0` — Desktop window mode (optional, falls back to browser)

### Node.js (Development Only)

The Workflow frontend uses Vue 3 via CDN (no build step required for production use). **Node.js is only needed if you want to modify the frontend JavaScript.**

Install Node.js from [nodejs.org](https://nodejs.org/) (LTS recommended) or via package manager:

```bash
# Windows (winget)
winget install OpenJS.NodeJS.LTS

# macOS (Homebrew)
brew install node

# Linux (nvm - recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install --lts
```

For frontend development, you may want a local dev server with hot reload:

```bash
cd workflow/web
npx serve .    # or: python -m http.server 3000
```

### pywebview System Dependency

On Windows, pywebview requires **Microsoft Edge WebView2 Runtime**, which is pre-installed on Windows 10 (1903+) and Windows 11. If missing, download it from [Microsoft](https://developer.microsoft.com/en-us/microsoft-edge/webview2/).

On Linux, pywebview requires `python3-gi` or `python3-pyqt5` — see [pywebview docs](https://pywebview.flowrl.com/guide/installation.html).

### Launching

```bash
# Desktop window mode (default)
python -m workflow

# Browser mode (no pywebview needed)
python -m workflow --no-gui

# Custom port and workflow root
python -m workflow --port 8765 --workflows-root /path/to/workflows
```

## Quick Start: Single-Stage Training

This example walks through creating a basic one-stage LoKR training workflow.

### 1. Launch the Workflow UI

```bash
python -m workflow
```

A desktop window opens at `http://localhost:8765`.

### 2. Create a New Workflow

Click **"New Workflow"** and give it a name (e.g., `my_first_training`).

### 3. Add a Preprocess Stage

1. Click **"Add Stage"** → select **Preprocess**
2. Set **Source directory** to your `image_dataset/` folder
3. Select **Bucket family** — start with `L` (1.03 MP, good balance of quality and speed)
4. Leave **Min pixels** at the default (500,000)

The preprocess stage will resize your images to fit the selected bucket family, then cache VAE latents and text embeddings.

### 4. Add a Train Stage

1. Click **"Add Stage"** → select **Train**
2. Select **Method** — e.g., **LoKR**
3. Configure parameters in the schema-driven form (network_dim, learning_rate, max_train_epochs, etc.)
4. The **Dataset** field automatically references the upstream Preprocess stage's output

### 5. Run

Click **"Run"**. The workflow executes stages in order:

1. **Preprocess** — resizes images, caches VAE latents and text embeddings
2. **Train** — trains the LoKR adapter

### 6. Find Your Training Artifacts

Training outputs are organized under the workflow directory:

```
.anima_workflow/my_first_training/
  runs/
    20260530-120000/          ← timestamped run directory
      preprocess_1/
        post_image_dataset/   ← resized images and caches
      train_1/
        output/
          *.safetensors       ← your trained adapter
        command.txt           ← exact command that was run
        config.toml           ← resolved config
      status.json             ← run status snapshot
      run.log                 ← full log
    latest → 20260530-120000/ ← junction link to latest run
```

**Three ways to find your latest adapter:**

1. **`runs/latest/train_1/output/`** — the `latest` junction always points to the most recent run
2. **History tab** — click the "Open directory" button on any completed run
3. **System log** — shows the safetensors path when training completes

## Single-Stage Usage in Detail

### Preprocess Stage

| Setting | Description |
|---------|-------------|
| **Source directory** | Path to your raw training images (with `.txt` caption sidecars) |
| **Bucket families** | Which resolution family to use. See [Bucket Families guide](bucket-families.md) for details. |
| **Min pixels** | Images below this pixel count are skipped (default: 500,000) |

The preprocess stage runs three sub-steps in order:
1. **Resize** — scales and crops images to fit the selected bucket family
2. **VAE cache** — encodes images to latent space
3. **TE cache** — encodes text captions to embeddings

### Train Stage

The train stage presents a schema-driven form that changes based on the selected method:

- **Method selector** — dropdown to switch between LoRA, LoKR, LoHA, etc.
- **Common parameters** — learning rate, epochs, batch size, optimizer
- **Method-specific parameters** — e.g., `lokr_factor` for LoKR, `network_dim` for LoRA

The form is generated from `workflow/schemas/train_{method}.yaml` and `workflow/schemas/train_common.yaml`.

## Multi-Stage Usage

Multi-stage workflows enable advanced training strategies like [low-resolution pre-training followed by high-resolution refinement](bucket-families.md#multi-stage-training-strategy).

### How Multi-Stage Orchestration Works

Stages are executed in **topological order** based on `depends_on` declarations. The scheduler detects circular dependencies and reports an error.

Each stage's outputs are available to subsequent stages via:
- **Automatic references** — the system auto-fills `network_weights` and `datasets` from upstream outputs
- **Placeholder syntax** — `${stage_id.output_key}` in config values, resolved at runtime

### Multiple Preprocess Stages

Each Preprocess stage can use different settings:

| Setting | Preprocess 1 | Preprocess 2 |
|---------|-------------|-------------|
| **Bucket families** | `S1` (low resolution, 0.26 MP) | `L` (high resolution, 1.03 MP) |
| **Source directory** | `image_dataset/` | `image_dataset/` (same or different) |

This produces two sets of cached data at different resolutions, each in its own subdirectory.

### Multiple Train Stages

#### `stop_epoch` — Interrupt and Save

Set `stop_epoch` on a Train stage to stop training at a specific epoch and ensure a checkpoint is saved:

```
stop_epoch = 6
```

This sets `max_train_epochs` and `save_every_n_epochs` to the specified value, so training stops immediately after saving the epoch-6 checkpoint.

#### Checkpoint Continuation

When a Train stage runs after another Train stage, it automatically:

1. Finds the upstream stage's `safetensors_path` output
2. Sets `--network_weights` to that path
3. For LoRA: sets `--dim_from_weights` to auto-infer rank from the checkpoint
4. For LyCORIS (lokr/loha/locon): sets `dim_from_weights = false` (dimensions must match config)

#### Typical Multi-Stage Flow

```
Preprocess S1 → Train S1 (stop at epoch 6) → Preprocess L → Train L (from S1 checkpoint)
```

1. **Preprocess S1**: Resize + cache at S1 family (0.26 MP)
2. **Train S1**: Train LoKR adapter, stop at epoch 6
3. **Preprocess L**: Resize + cache at L family (1.03 MP)
4. **Train L**: Continue from S1's epoch-6 checkpoint, using both S1 and L caches

The second Train stage references the first's output via placeholder: `${train_1.safetensors_path}` → resolved to the actual path.

## Log Viewer

The bottom panel has three tabs:

### System Log

Shows workflow-level events: stage start/end, checkpoint saves, errors. Updated in real-time via SSE (Server-Sent Events).

### Script Output

Shows subprocess stdout with:
- **TQDM progress bars** — parsed and displayed as visual progress bars with step count, elapsed time, ETA, and metrics (loss, lr)
- **Stage filtering** — filter output by stage using the dropdown
- **Auto-scroll** — automatically scrolls to latest output; pause/resume with the scroll lock button
- **Buffer limit** — 500 lines per stage; oldest lines are trimmed when exceeded

### Run History

Lists all previous runs in reverse chronological order. Each entry shows:
- **Timestamp** and **duration**
- **Status**: ok / stopped / error / running
- **Stage chain** with color-coded status indicators
- **Actions**: "View log" and "Open directory"

**To find your latest training artifact from history:**
1. Open the **History** tab
2. The most recent run is at the top
3. Click **"Open directory"** to open the run folder
4. Navigate to `{train_stage_id}/output/` to find the `.safetensors` file

Alternatively, `runs/latest` is always a junction/symlink to the most recent run directory.

### Search and Highlight

The log viewer supports text search with highlight matching across all visible log lines.

## Settings

### Language

The UI automatically detects your browser language and supports three languages:
- **English** (en)
- **中文** (zh-CN)
- **日本語** (ja)

To switch manually, use the language selector in the top-right corner. Your preference is saved in `localStorage`.

All schema labels, field descriptions, help texts, and choice labels are translated via the i18n overlay system.

### Model Settings

Configure model paths in the **Settings** dialog:

| Setting | Default | Description |
|---------|---------|-------------|
| **DiT model** | `models/diffusion_models/anima-base-v1.0.safetensors` | Base model path |
| **Qwen3 text encoder** | `models/text_encoders/qwen_3_06b_base.safetensors` | Text encoder path |
| **VAE** | `models/vae/qwen_image_vae.safetensors` | VAE path |

Paths resolve against the repository root (`ANIMA_HOME`). Set `ANIMA_DIT`, `ANIMA_VAE`, or `ANIMA_TEXT_ENCODER` environment variables to override.

### Hardware Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Mixed precision** | `bf16` | Training precision |
| **Attention mode** | `flex` | Attention implementation |

### Override Priority

Settings are applied in this order (later overrides earlier):

1. **Infrastructure defaults** — resolved from `library.env.resolve_under_home()`
2. **Infrastructure config** — per-workflow settings stored in `workflow.yaml`
3. **Stage config** — per-stage TOML overrides
4. **Auto-derived** — `network_weights`, `datasets`, etc. automatically filled from upstream outputs

Global settings (workflows root, etc.) are stored in `.anima_workflow_config.json` at the project root.
```

- [ ] **Step 2: Verify the file was created**

Run: `ls -la docs/guidelines/workflow.md`

- [ ] **Step 3: Commit**

```bash
git add docs/guidelines/workflow.md
git commit -m "docs: add Workflow engine usage guide"
```

---

### Task 6: Update `README.md`

**Files:**
- Modify: `README.md` (insert after line 13, before the `---` on line 16)

- [ ] **Step 1: Insert "What's new" section into README.md**

In `README.md`, search for the block:

```markdown
4. **A broad experimental surface** — SPD, ChimeraHydra, Soft Tokens, Turbo distillation, ReFT, IP-Adapter, EasyControl, DirectEdit, embedding inversion.

> **At-a-glance diagrams** for every method (DiT internals, LoRA, OrthoLoRA, T-LoRA, HydraLoRA, ReFT, Spectrum, modulation, compile optimizations) live in [`docs/structure_images/`](docs/structure_images/) — paired with prose walkthroughs in [`docs/structure/`](docs/structure/).

---
```

Replace with:

```markdown
4. **A broad experimental surface** — SPD, ChimeraHydra, Soft Tokens, Turbo distillation, ReFT, IP-Adapter, EasyControl, DirectEdit, embedding inversion.

> **At-a-glance diagrams** for every method (DiT internals, LoRA, OrthoLoRA, T-LoRA, HydraLoRA, ReFT, Spectrum, modulation, compile optimizations) live in [`docs/structure_images/`](docs/structure_images/) — paired with prose walkthroughs in [`docs/structure/`](docs/structure/).

## What's new

| Feature | Description | Guide |
|---------|-------------|-------|
| **LoKR** | Low-rank Kronecker product adaptation — structured high-rank with adaptive parameter count | [docs/guidelines/lokr.md](docs/guidelines/lokr.md) |
| **LoHA** | Low-rank Hadamard product adaptation — effective rank r² with only 2× LoRA parameters | [docs/guidelines/loha.md](docs/guidelines/loha.md) |
| **CAME optimizer** | Factorized optimizer replacing full-matrix second moments — significant memory savings | [docs/guidelines/came.md](docs/guidelines/came.md) |
| **Bucket Families** | Resolution bucketing by area → AR matching to token-count groups for compile performance | [docs/guidelines/bucket-families.md](docs/guidelines/bucket-families.md) |
| **Workflow engine** | WebUI + CLI multi-stage training pipeline with real-time progress and schema-driven forms | [docs/guidelines/workflow.md](docs/guidelines/workflow.md) |

---
```

- [ ] **Step 2: Verify the edit**

Run: `head -30 README.md` and confirm the "What's new" table appears between the `> **At-a-glance diagrams**` block and the `---` separator.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add What's new section to README with links to usage guides"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Task |
|-----------------|------|
| `docs/guidelines/lokr.md` | Task 1 |
| `docs/guidelines/loha.md` | Task 2 |
| `docs/guidelines/came.md` | Task 3 |
| `docs/guidelines/bucket-families.md` | Task 4 |
| `docs/guidelines/workflow.md` | Task 5 |
| README.md "What's new" section | Task 6 |
| Workflow install (Node.js, pywebview) | Task 5 Step 1 (Installation section) |
| Workflow quick start example | Task 5 Step 1 (Quick Start section) |
| Workflow single-stage usage | Task 5 Step 1 (Single-Stage Usage section) |
| Workflow multi-stage (stop_epoch, checkpoint continuation) | Task 5 Step 1 (Multi-Stage Usage section) |
| Workflow log viewer and history | Task 5 Step 1 (Log Viewer section) |
| Workflow settings (language, model, overrides) | Task 5 Step 1 (Settings section) |
| Bucket families two-step matching algorithm | Task 4 Step 1 (How Images Are Matched section) |
| Bucket families performance trade-offs | Task 4 Step 1 (Token Count and Performance section) |
| Bucket families multi-stage strategy with link to workflow | Task 4 Step 1 (Multi-Stage Training Strategy section) |
| No modification to existing install instructions | Task 6 only inserts new section |

### Placeholder Scan

No TBD, TODO, or placeholder patterns found.

### Cross-Reference Consistency

- `bucket-families.md` links to `workflow.md#multi-stage-usage` ✅
- `workflow.md` links to `bucket-families.md#multi-stage-training-strategy` ✅
- `lokr.md` and `loha.md` link to `../methods/lycoris-variants.md` ✅
- `came.md` links to `../optimizations/adv_optm_guide.md` ✅
- README.md links are consistent with file paths ✅
