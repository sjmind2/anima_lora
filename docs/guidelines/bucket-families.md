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
