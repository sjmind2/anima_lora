# Preprocess Tree Mode: Multi-Subset Directory Preservation

Single-pass preprocessing that discovers sub-directories as independent subsets and preserves the directory tree through all pipeline stages (resize → cache latents → cache text embeddings → masks).

## Design Motivation

Traditional `--recursive` mode flattens all images into a single output directory, requiring globally unique filenames. When training with multiple subsets (e.g. different characters, poses, or styles), this causes two problems:

1. **Name collisions** — `char_a/photo.png` and `char_b/photo.png` overwrite each other when flattened.
2. **Model reload overhead** — each subset requires a separate invocation of `resize_images.py`, `cache_latents.py`, and `cache_text_embeddings.py`, reloading the VAE and text encoders every time. With 10 subsets, models are loaded 30 times instead of 3.

Tree mode solves both by treating each direct sub-directory as an **independent namespace** with its own `.resized/`, `.lora/`, and `.masks/` output folders. All subsets are processed in a single pass, so each model is loaded exactly once.

## Directory Structure (Stages 0–4)

### Stage 0: Source Data

```
origin_dataset/mychar/                    ← source_image_dir
  img_root1.png
  img_root2.png
  4_a/                                    ← subset (num_repeats=4)
    img_a1.png
    img_a2.png
  5_b/                                    ← subset (num_repeats=5)
    img_b1.png
  6_c/                                    ← subset (num_repeats=6)
    img_c1.png
    img_c2.png
```

### Stage 1: Resized Output (`.resized/` per subset)

```
post_image_dataset/mychar/
  .resized/                               ← root subset
    img_root1.png
    img_root2.png
  4_a/
    .resized/
      img_a1.png
      img_a2.png
  5_b/
    .resized/
      img_b1.png
  6_c/
    .resized/
      img_c1.png
      img_c2.png
```

### Stage 2: Cached Latents (`.lora/` per subset)

```
post_image_dataset/mychar/
  .resized/
    img_root1.png
    img_root2.png
  .lora/                                   ← root subset latents
    img_root1_1024x1024_anima.npz
    img_root2_1024x1024_anima.npz
  4_a/
    .resized/
      img_a1.png
      img_a2.png
    .lora/                                 ← 4_a latents
      img_a1_1024x1024_anima.npz
      img_a2_1024x1024_anima.npz
  5_b/
    .resized/
      img_b1.png
    .lora/                                 ← 5_b latents
      img_b1_1024x1024_anima.npz
  6_c/
    .resized/
      img_c1.png
      img_c2.png
    .lora/                                 ← 6_c latents
      img_c1_1024x1024_anima.npz
      img_c2_1024x1024_anima.npz
```

### Stage 3: Cached Text Embeddings (inside `.lora/`)

```
post_image_dataset/mychar/
  .resized/
    img_root1.png
    img_root2.png
  .lora/
    img_root1_1024x1024_anima.npz
    img_root2_1024x1024_anima.npz
    img_root1_anima_te.safetensors        ← root subset TE cache
    img_root2_anima_te.safetensors
  4_a/
    .resized/
      img_a1.png
      img_a2.png
    .lora/
      img_a1_1024x1024_anima.npz
      img_a2_1024x1024_anima.npz
      img_a1_anima_te.safetensors
      img_a2_anima_te.safetensors
  5_b/
    .resized/
      img_b1.png
    .lora/
      img_b1_1024x1024_anima.npz
      img_b1_anima_te.safetensors
  6_c/
    .resized/
      img_c1.png
      img_c2.png
    .lora/
      img_c1_1024x1024_anima.npz
      img_c2_1024x1024_anima.npz
      img_c1_anima_te.safetensors
      img_c2_anima_te.safetensors
```

### Stage 4: Masks (`.masks/` per subset)

```
post_image_dataset/mychar/
  .resized/
    img_root1.png
    img_root2.png
  .lora/
    img_root1_1024x1024_anima.npz
    img_root2_1024x1024_anima.npz
    img_root1_anima_te.safetensors
    img_root2_anima_te.safetensors
  .masks/                                  ← root subset masks
    img_root1_mask.png
    img_root2_mask.png
  4_a/
    .resized/
      img_a1.png
      img_a2.png
    .lora/
      img_a1_1024x1024_anima.npz
      img_a2_1024x1024_anima.npz
      img_a1_anima_te.safetensors
      img_a2_anima_te.safetensors
    .masks/                                ← 4_a masks
      img_a1_mask.png
      img_a2_mask.png
  5_b/
    .resized/
      img_b1.png
    .lora/
      img_b1_1024x1024_anima.npz
      img_b1_anima_te.safetensors
    .masks/                                ← 5_b masks
      img_b1_mask.png
  6_c/
    .resized/
      img_c1.png
      img_c2.png
    .lora/
      img_c1_1024x1024_anima.npz
      img_c2_1024x1024_anima.npz
      img_c1_anima_te.safetensors
      img_c2_anima_te.safetensors
    .masks/                                ← 6_c masks
      img_c1_mask.png
      img_c2_mask.png
```

No global `post_image_dataset/masks/` directory is created. Each subset's masks live alongside its images and caches, making the subset directory fully self-contained.

## CLI Usage

```bash
# Resize with tree mode
python scripts/preprocess/resize_images.py \
  --src origin_dataset/mychar \
  --dst post_image_dataset/mychar \
  --tree

# Cache latents with tree mode
python scripts/preprocess/cache_latents.py \
  --dir post_image_dataset/mychar \
  --cache_dir post_image_dataset/mychar \
  --tree

# Cache text embeddings with tree mode
python scripts/preprocess/cache_text_embeddings.py \
  --dir origin_dataset/mychar \
  --cache_dir post_image_dataset/mychar \
  --tree

# Generate masks (auto-detects tree structure)
make mask
```

## Comparison: Tree vs Recursive Mode

| Aspect | `--recursive` | `--tree` |
|--------|---------------|----------|
| **Input structure** | Flattens all sub-directories into one | Each sub-directory = independent subset |
| **Filename scope** | Global uniqueness required across all sub-dirs | Uniqueness only within each subset |
| **Same-name files** | `char_a/photo.png` and `char_b/photo.png` collide | No collision — separate `.resized/` per subset |
| **Output layout** | Single flat output directory | `{subset}/.resized/`, `{subset}/.lora/`, `{subset}/.masks/` |
| **Model loads** | One per invocation (N subsets = N loads) | Single load processes all subsets |
| **Mask location** | Global `post_image_dataset/masks/` | Per-subset `{subset}/.masks/` |
| **Portability** | Subset data scattered across directories | Copy one subset folder = all data included |
| **Hidden directories** | Processed normally | Skipped (`.*` prefix) |
| **Root images** | Mixed into flat output | Separate `.resized/` at root level |
| **Backward compat** | Original mode, unchanged | Opt-in flag, no effect when omitted |

## Mask Self-Containment

Tree mode extends the self-containment principle to masks: each subset directory holds its own `.masks/` folder alongside `.resized/` and `.lora/`. This means:

- **Portability** — copy `post_image_dataset/mychar/4_a/` to another project and all data (images, latent caches, TE caches, masks) travel together.
- **Independence** — subsets have no implicit cross-references; any subset can be added or removed without affecting others.
- **Consistency** — `.resized/`, `.lora/`, and `.masks/` all follow the same directory pattern within each subset.

See [mask-self-containment.md](mask-self-containment.md) for the full design rationale and implementation details.

## Implementation Notes

- `library/preprocess/images.py` — `resize_to_buckets(tree=True)` scans `src` for direct sub-directories, processes each subset independently, and outputs to `{dst}/{subset}/.resized/`.
- `scripts/preprocess/resize_images.py` — argparse wrapper exposing `--tree` flag.
- `scripts/tasks/masking.py` — `_detect_subset_dirs()` auto-discovers subsets by scanning for `.resized/` directories; `_cmd_mask_tree()` runs SAM + MIT + merge per subset.
- Hidden directories (`.*` prefix) are always skipped during subset discovery.
- Root-level images (if any) are treated as a special `__root__` subset with output at `{dst}/.resized/`.
