# Workflow Subset Repeat Fix

## Problem

When training datasets are organized in `{repeat}_{caption_tag}` directories (e.g. `10_hina_(uniform)_(rune_factory)_face/`, `4_iroha_(bikini)_(rune_factory)/`), the workflow system ignores the repeat prefix and trains every subset with `num_repeats=1`.

This means a directory like `10_hina_...` (intended to repeat 10x per epoch) only trains once per epoch, causing under-training for subsets that need more repetitions.

## Root Cause

Three locations cause the bug:

1. **`workflow/stages/preprocess.py`** — `discover_subsets()` creates `SubsetInfo` objects with `num_repeats=1` regardless of the directory name prefix (lines 96, 113).

2. **`workflow/scheduler.py`** — When serializing `SubsetInfo` to `stage_outputs`, the scheduler drops `num_repeats`, only keeping `name`, `image_dir`, `cache_dir` (line 190).

3. **`workflow/stages/train.py`** — `_write_dataset_toml()` writes `entry["num_repeats"] = 1` into the generated `dataset_config.toml` (line 166).

The generated `dataset_config.toml` therefore always has `num_repeats = 1` for every subset, even when the source directory name encodes a different repeat count.

## Solution

Parse the `{repeat}_{tag}` naming convention from directory names in both locations, using the same algorithm as the existing `extract_dreambooth_params` in `library/config/loader.py`:

```
name.split("_") → first token coerced to int → repeat count
fallback to 1 if not a positive integer
```

### Changes

#### 1. Add helper function in `workflow/stages/preprocess.py`

```python
def _parse_num_repeats(name: str) -> int:
    """Extract repeat count from a '{repeat}_{tag}' directory name."""
    tokens = name.split("_")
    try:
        n = int(tokens[0])
        return n if n >= 1 else 1
    except (ValueError, IndexError):
        return 1
```

#### 2. Use in `discover_subsets()` (preprocess.py)

Replace both `num_repeats=1` with `num_repeats=_parse_num_repeats(dataset_dir.name)` (line 96) and `num_repeats=_parse_num_repeats(subset_dir.name)` (line 113).

#### 3. Use in `_write_dataset_toml()` (train.py)

The `SubsetInfo` model already carries `num_repeats`. After step 2, the correct value flows through `stage_outputs["subsets"]`. Change line 166 from:

```python
entry["num_repeats"] = 1
```

to:

```python
entry["num_repeats"] = s.get("num_repeats", 1)
```

Also fix the fallback path (line 171) similarly.

### Backward Compatibility

* Directories without a leading integer prefix (e.g. `my_characters/`) return `num_repeats=1`, same as current behavior.

* No config file changes required.

* No changes to the core Anima training pipeline (`library/`, `train.py`).

## Files Modified

| File                            | Change                                                        |
| ------------------------------- | ------------------------------------------------------------- |
| `workflow/stages/preprocess.py` | Add `_parse_num_repeats()`, use in `discover_subsets()`       |
| `workflow/scheduler.py`         | Include `num_repeats` when serializing `SubsetInfo` to dict   |
| `workflow/stages/train.py`      | Use `num_repeats` from subset info in `_write_dataset_toml()` |

## Verification

* Run an existing workflow with `{repeat}_{tag}` directories and check the generated `dataset_config.toml` — `num_repeats` should match the directory prefix.

* Run with directories that have no numeric prefix — `num_repeats` should be `1`.

* Training log should show correct repeat-multiplied image counts.

