# Preprocess Bucket Family: Detailed Descriptions + Dataset Distribution Stats

Date: 2026-05-29

## Problem

Two issues in the workflow frontend's preprocess stage:

1. **Vague option labels**: The `bucket_families` field shows labels like "XS — 336~640 竖/横" which don't reveal the exact resolutions. Each family has a fixed set of resolutions (from `BUCKET_FAMILIES` in `library/datasets/buckets.py`), but the user can't see them without reading source code.

2. **No dataset distribution analysis**: Users cannot evaluate how their dataset images distribute across bucket families before selecting which families to enable. The GUI has a "Stats" button (`scan_images_for_bucket_stats` in `gui/__init__.py`) but the workflow frontend has nothing.

## Design

### Part 1: Detailed resolution display per family

**Schema change** (`workflow/schemas/preprocess.yaml`):

Add `choice_details` to the `bucket_families` field, containing each family's token count and full resolution list:

```yaml
choice_details:
  XL:
    tc: 5040
    resolutions: ["640×2016", "672×1920", "720×1792", "768×1680", "896×1440", "960×1344", "1008×1280", "1120×1152"]
  L:
    tc: 4032
    resolutions: ["512×2016", "576×1792", "672×1536", "768×1344", "896×1152", "1008×1024"]
  M:
    tc: 3600
    resolutions: ["480×1920", "576×1600", "640×1440", "720×1280", "768×1200", "800×1152", "960×960"]
  S:
    tc: 2160
    resolutions: ["384×1440", "432×1280", "480×1152", "576×960", "640×864", "720×768"]
  XS:
    tc: 1680
    resolutions: ["336×1280", "384×1120", "448×960", "480×896", "560×768", "640×672"]
  S1:
    tc: 1024
    resolutions: ["256×1024", "512×512"]
  S2:
    tc: 4096
    resolutions: ["512×2048", "1024×1024"]
```

**Frontend change** (`workflow/web/js/components/FieldRenderer.js`):

When a `list[str]` field has `choice_details`, each option button renders an additional gray text line below it showing the token count and all resolutions. Style matches the GUI's `color: gray; font-size: 10px;`.

Visual example:
```
┌─────────────────────────────────────────┐
│ ✓ XL                                    │  ← selectable button
│   TC=5040 · 640×2016 672×1920 ...       │  ← gray detail line
│   原始: 5张 · 缩放后: 5张               │  ← stats line (after analysis)
└─────────────────────────────────────────┘
```

### Part 2: Dataset distribution statistics

#### New API endpoint

`POST /api/dataset/bucket-stats`

Request:
```json
{
    "source_dir": "image_dataset/my_chars",
    "enabled_families": ["M", "L"]
}
```

Response:
```json
{
    "total_images": 42,
    "families": {
        "XL":  { "original": 5,  "resized": 0 },
        "L":   { "original": 12, "resized": 18 },
        "M":   { "original": 10, "resized": 24 },
        "S":   { "original": 8,  "resized": 0 },
        "XS":  { "original": 4,  "resized": 0 },
        "S1":  { "original": 2,  "resized": 0 },
        "S2":  { "original": 1,  "resized": 0 }
    }
}
```

Semantics:
- **original**: How images distribute when ALL families are available (natural area-based matching). Each image maps to the family whose `tc * 256` is closest to the image's pixel area.
- **resized**: How images distribute when ONLY `enabled_families` are available. Images that would naturally map to unselected families get reassigned to the nearest selected family.

For example, if 5 images naturally distribute 1-per-family across 5 families but the user only selects XL, then XL gets `resized: 5` and all others get `resized: 0`.

#### Core logic

New function `scan_dataset_bucket_distribution(source_dir, enabled_families)` in `library/datasets/buckets.py`:

1. Iterate all images in `source_dir`, read dimensions via PIL
2. Pass 1: match each image to nearest family among ALL families → `original` counts
3. Pass 2: match each image to nearest family among `enabled_families` only → `resized` counts
4. Return structured result

The existing `scan_images_for_bucket_stats()` in `gui/__init__.py` will be updated to import from this shared location.

#### Frontend interaction

- An "分析数据集" button appears at the top-right of the bucket_families field area
- Button is disabled when `source_image_dir` is empty or doesn't exist
- On click: POST to `/api/dataset/bucket-stats` with `source_dir` and current `enabled_families`
- Response populates inline stats rows below each option button
- Stats rows show "原始: N张 · 缩放后: M张" format
- Families not in `enabled_families` show "缩放后: 0张" in a dimmer style

### Files to change

| File | Change |
|------|--------|
| `workflow/schemas/preprocess.yaml` | Add `choice_details` with tc + resolutions per family |
| `workflow/web/js/components/FieldRenderer.js` | Render detail lines + stats rows + analysis button for `list[str]` with `choice_details` |
| `workflow/web/js/api.js` | Add `analyzeBucketStats(sourceDir, enabledFamilies)` method |
| `workflow/app.py` | Add `POST /api/dataset/bucket-stats` route handler |
| `library/datasets/buckets.py` | Add `scan_dataset_bucket_distribution()` function |
| `gui/__init__.py` | Refactor `scan_images_for_bucket_stats()` to import from `library/datasets/buckets.py` |

### Error handling

- `source_dir` doesn't exist → return `{"error": "Directory not found"}`
- `source_dir` has no images → return `{"total_images": 0, "families": {...}}` with all zeros
- `enabled_families` empty → treat same as all families selected (both passes identical)
- PIL read failures → skip image silently (same as GUI behavior)
