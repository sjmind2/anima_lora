# Shared Cache for Workflow Preprocess Stages

**Date**: 2026-06-08
**Status**: Approved

## Problem

Every workflow run creates a completely fresh preprocess output under its own `<run_dir>/<stage_id>/post_image_dataset/`. When the same dataset and configuration are used across multiple runs (e.g. iterating on training hyperparameters), this wastes time and disk space re-computing identical VAE latents and text embeddings.

## Scope

- Shared cache is **per-workflow**: `{workflows_root}/{workflow_name}/shared_cache/`, not cross-workflow.
- Applies only to **preprocess** stages. Train stages consume whatever path the upstream preprocess provides.
- Configuration comparison is **per-stage, whole-pipeline** (resize + VAE + TE treated as one unit). If config changes, the entire `post_image_dataset/` is deleted and re-computed.

## Design

### §1 Directory Structure

**New shared cache path**:

```
{workflows_root}/{workflow_name}/shared_cache/{stage_id}/
    config.toml              ← full resolved config snapshot (human-readable)
    .config_hash             ← SHA256 of canonical JSON for comparison
    post_image_dataset/      ← identical structure to current run-dir output
```

**Non-shared (existing behavior)** remains at:

```
{workflows_root}/{workflow_name}/runs/{timestamp}/{stage_id}/post_image_dataset/
```

### §2 Configuration Hash

Hash is computed over the **resolved** config dict (after placeholder substitution), not the raw user config. This ensures that infrastructure path changes (e.g. model path resolution) are detected.

```python
import json, hashlib

canonical = json.dumps(resolved_config, sort_keys=True, ensure_ascii=False)
hash_val = hashlib.sha256(canonical.encode()).hexdigest()
```

`json.dumps(sort_keys=True)` guarantees deterministic key ordering regardless of how the dict was constructed (merge order, placeholder resolution order, etc.). The TOML file is saved alongside for human inspection but is **not** used for hash comparison.

### §3 PreprocessExecutor Flow

When `shared_cache=True` (default):

1. Compute `config_hash` from resolved config.
2. `shared_dir = wf_dir / "shared_cache" / stage_id`
3. `post_image_dataset = shared_dir / "post_image_dataset"`
4. If `post_image_dataset` exists:
   - Read `shared_dir / ".config_hash"`
   - **Hash matches**: log "Shared cache hit, skipping preprocess ({stage_id})". Call `discover_subsets()` on the existing directory. Return `StageResult` with `dataset_dir` pointing to `shared_dir / "post_image_dataset"`.
   - **Hash differs**: delete `post_image_dataset/` (keep `config.toml` and `.config_hash` at top level — they will be overwritten). Execute full preprocess pipeline. Write new `config.toml` and `.config_hash`.
5. If `post_image_dataset` does not exist: execute full preprocess. Write `config.toml` and `.config_hash`.

When `shared_cache=False`:

- Identical to current behavior: `stage_dir = run_dir / stage_id`, always compute fresh.

### §4 Train Node — No Changes Required

The train node receives dataset paths via `stage_outputs[stage_id]["dataset_dir"]`. Whether this points to `shared_cache/{stage_id}/post_image_dataset` or `runs/{timestamp}/{stage_id}/post_image_dataset` is transparent — the train node and `_write_dataset_toml()` work with either path unchanged.

### §5 UI / Schema

- `workflow/schemas/preprocess.yaml`: add `shared_cache` boolean field at the top of the schema, default `true`, with i18n label.
- `FieldRenderer` already renders boolean fields as toggle switches — no JS changes needed.
- i18n keys added to `workflow/i18n/locales/{en,ja,zh-CN}.json`.

### §6 Edge Cases

- **Partial cache** (e.g. resize completed but VAE cache failed mid-run): `post_image_dataset/` exists but is incomplete. The hash file will be missing or stale. On next run, hash mismatch triggers full re-compute, deleting the partial `post_image_dataset/`.
- **Manual cache deletion**: User deletes `shared_cache/{stage_id}/post_image_dataset/` but leaves `.config_hash`. Hash file exists but `post_image_dataset` does not → falls through to "does not exist" branch, re-computes.
- **Multiple preprocess stages**: Each stage has its own `shared_cache/{stage_id}/` — independent.

## Files to Modify

| File | Change |
|------|--------|
| `workflow/schemas/preprocess.yaml` | Add `shared_cache` boolean field |
| `workflow/stages/preprocess.py` | Add shared cache logic to `PreprocessExecutor.execute()` |
| `workflow/i18n/locales/en.json` | Add label for `shared_cache` |
| `workflow/i18n/locales/ja.json` | Add label for `shared_cache` |
| `workflow/i18n/locales/zh-CN.json` | Add label for `shared_cache` |

No changes to `train.py`, `scheduler.py`, `models.py`, or frontend JS.
