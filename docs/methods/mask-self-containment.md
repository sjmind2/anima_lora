# Mask Self-Containment: Per-Subset Mask Directories

Design principle and implementation for distributing masks into each subset directory rather than a single global directory.

## Design Principle

Each subset directory is a **self-contained data unit**. All data required for training — images, latent caches, text-encoder caches, and masks — live within the subset's own directory tree:

```
post_image_dataset/mychar/4_a/
  .resized/img_a1.png                    ← images
  .resized/img_a2.png
  .lora/img_a1_1024x1024_anima.npz       ← VAE latent cache
  .lora/img_a2_1024x1024_anima.npz
  .lora/img_a1_anima_te.safetensors      ← text encoder cache
  .lora/img_a2_anima_te.safetensors
  .masks/img_a1_mask.png                 ← masks
  .masks/img_a2_mask.png
```

This yields three properties:

1. **Portability** — copy one subset folder to another project and all its data travels together. No external mask directory to remember.
2. **Independence** — subsets have no implicit cross-references. Any subset can be added, removed, or re-generated without affecting others.
3. **Consistency** — `.resized/`, `.lora/`, and `.masks/` all follow the same directory pattern within each subset.

## SAM → MIT → Merge Execution Flow (Per-Subset)

Mask generation in tree mode runs the full SAM + MIT → merge pipeline **independently for each subset**. Intermediates are written to a `tempfile.TemporaryDirectory` (never persisted under the project root), and only the merged result lands in the subset's `.masks/` directory.

### Per-Subset Pipeline

```
For each subset (root, 4_a, 5_b, 6_c):

  Step 1: SAM mask generation
    Input:  {subset}/.resized/ (e.g. img_a1.png, img_a2.png)
    Output: tempfile/sam_{subset}/ (img_a1_mask.png, img_a2_mask.png)

  Step 2: MIT mask generation
    Input:  {subset}/.resized/ (e.g. img_a1.png, img_a2.png)
    Output: tempfile/mit_{subset}/ (img_a1_mask.png, img_a2_mask.png)

  Step 3: Merge (SAM + MIT → final)
    Input:  tempfile/sam_{subset}/ + tempfile/mit_{subset}/
    Output: {subset}/.masks/ (img_a1_mask.png, img_a2_mask.png)
    Logic:  pixel-wise minimum of both sources (union merge)
```

### Discovery and Orchestration

`_detect_subset_dirs()` in `scripts/tasks/masking.py` scans `post_image_dataset/` for directories containing `.resized/` sub-directories:

```python
def _detect_subset_dirs(parent_dir: Path) -> list[tuple[str, Path]]:
    # Returns (name, resized_dir) for each subset
    # Root subset: parent_dir/.resized/
    # Child subsets: parent_dir/{child}/.resized/
```

`cmd_mask()` checks whether any subset directories exist. If so, it delegates to `_cmd_mask_tree()` for per-subset processing. If no tree structure is detected, it falls back to the original global mode (single `post_image_dataset/masks/` output).

Either backend (SAM or MIT) can be disabled via environment variables:

```bash
RUN_SAM_MASK=0   # skip SAM
RUN_MIT_MASK=0   # skip MIT
```

When only one runs, the merge step still fires but is a no-op for single-source inputs.

## Mask Discovery Path Priority

When a subset's `mask_dir` is not explicitly configured, `_resolve_default_mask_dir()` in `library/datasets/subsets.py` searches a priority-ordered candidate list:

| Priority | Candidate Path | When Used |
|----------|---------------|-----------|
| 1 (highest) | `{image_dir}/../.masks/` | Tree mode — subset-local masks |
| 2 | `post_image_dataset/masks/` | Global mask output (non-tree mode) |
| 3 | `masks/merged/` | Legacy layout |
| 4 | `masks/sam/` | Legacy single-backend |
| 5 | `masks/mit/` | Legacy single-backend |

For a subset with `image_dir = "post_image_dataset/mychar/4_a/.resized"`, the resolution is:

1. `Path(image_dir).parent / ".masks"` → `post_image_dataset/mychar/4_a/.masks` — **exists** → return this path.
2. (Fallback chain not reached.)

The first existing directory wins. This means tree-mode subsets automatically discover their own `.masks/`, while legacy projects without tree structure continue to find the global `post_image_dataset/masks/`.

### GUI Mask Preview

`_resolve_mask_path()` in `gui/tabs/image_tab.py` mirrors this discovery logic for the image viewer's mask overlay:

1. If the image is inside a `.resized/` directory, check `parent.parent / ".masks" / {stem}_mask.png`.
2. Otherwise, check `parent / ".masks" / {stem}_mask.png`.
3. Fall back to global paths: `post_image_dataset/masks/` and `masks/merged/`.

## Backward Compatibility

The design is fully backward-compatible with existing non-tree-mode workflows:

- **`--tree` is opt-in** — omitting the flag preserves the original behavior exactly.
- **Global `post_image_dataset/masks/` remains a fallback** — old datasets that were masked with the global layout continue to work without re-running mask generation.
- **`cmd_mask_clean()` cleans both locations** — removes the global `post_image_dataset/masks/` directory as well as any per-subset `.masks/` directories found under `post_image_dataset/`.
- **Training mask loading is transparent** — `DreamBoothDataset` and `load_mask_from_dir()` only depend on `subset.mask_dir` being correct. The discovery logic in `_resolve_default_mask_dir()` ensures this is always set appropriately regardless of layout.
- **`mask_dir` explicit config wins** — if a TOML config specifies `mask_dir` explicitly, it is used directly without auto-discovery, preserving manual overrides.

## Files Involved

| File | Role |
|------|------|
| `scripts/tasks/masking.py` | Orchestration: `_detect_subset_dirs()`, `_cmd_mask_tree()`, `cmd_mask()`, `cmd_mask_clean()` |
| `library/datasets/subsets.py` | Mask directory discovery: `_resolve_default_mask_dir()` |
| `gui/tabs/image_tab.py` | GUI mask preview: `_resolve_mask_path()` |
| `library/datasets/dreambooth.py` | Mask path binding during dataset enumeration |
| `library/datasets/base.py` | Mask preloading (`_preload_alpha_masks()`) and runtime access |
| `library/datasets/image_utils.py` | `load_mask_from_dir()` — actual PNG decode |
| `library/training/losses.py` | `apply_masked_loss()` — masked loss computation |
