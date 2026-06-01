# Layer-Targeted LoRA Training

**Date:** 2026-06-01
**Status:** Design
**Scope:** CLI, GUI, Workflow frontend

## Problem Statement

Anima's LoRA training currently applies adapters uniformly to all eligible DiT layers
(self-attention, cross-attention, MLP, and — if `_DEFAULT_EXCLUDE` is overridden — AdaLN
modulation). Anima's original author documents a per-goal layer-targeting matrix in the
trainer presets:

| Goal              | self_attn | cross_attn | mlp  | adaln |
|-------------------|-----------|------------|------|-------|
| Style + poses     | true      | false      | true | false |
| Style only        | false     | false      | true | false |
| Character         | false     | true       | true | false |
| Slider/concept    | false     | true       | false| false |
| Full (default)    | true      | true       | true | false |

We need to expose this four-axis control across all training interfaces (CLI, GUI
config editor, workflow frontend), with defaults matching the "Full" row
(true/true/true/false).

## Design Decisions

1. **Layer-type classification by path name.** Layer types are identified by
   substring matching on the module's `original_name` path. This is consistent with
   the existing layer classification in `library/anima/training.py::get_anima_param_groups()`
   (used for per-layer LR grouping) and the existing `router_targets` regex mechanism.

2. **Explicit filter step in `create_modules()`.** A new filter step is added to the
   LoRA module creation loop, after the existing `exclude_patterns`/`include_patterns`/
   `layer_start`/`layer_end` filters. The four flags are **hard switches** — they take
   precedence over `include_patterns` for matching layer types.

3. **`_DEFAULT_EXCLUDE` modification.** The `_modulation` alternative is removed from
   `_DEFAULT_EXCLUDE`. The `train_adaln=false` default takes over the responsibility of
   excluding AdaLN modulation layers. This is semantically equivalent for the default
   configuration.

4. **Static filtering, no runtime checks.** Filtering happens before `apply_to()` and
   `compile_blocks()`. The four values are static for the duration of a training task
   (read from the config chain at startup). Different layer-targeting choices produce
   different compile graphs, but each training task compiles exactly once — no
   recompilation is triggered.

5. **Mixed CLI bool approach.** The four new params use `type=lambda x: x.lower() in
   ("true", "1", "yes")`, `default=None` in argparse. This accepts `--train_self_attn
   true/false` format and is consistent with the existing per-layer LR params
   (`--self_attn_lr`, etc.). Existing `store_true` params are unaffected.

6. **Workflow bool whitelist.** The workflow's `_build_train_cmd()` uses a small
   whitelist set `_BOOL_VALUE_KEYS` to always pass `--key true/false` for the four new
   params, while keeping the existing `store_true` flag-passing behavior for all other
   bool params.

## Layer-Type Classification

### Path → Layer-Type Mapping

| Layer type  | Match rule                        | Example `original_name`                            |
|-------------|-----------------------------------|----------------------------------------------------|
| `self_attn` | `".self_attn."` in path           | `blocks.0.self_attn.qkv_proj`                      |
| `cross_attn`| `".cross_attn."` in path          | `blocks.0.cross_attn.kv_proj`                      |
| `mlp`       | `".mlp."` in path                 | `blocks.0.mlp.layer1`                              |
| `adaln`     | `"adaln_modulation_"` in path     | `blocks.0.adaln_modulation_self_attn.1`            |
| `None`      | (none of the above)               | `patch_embed.proj`, `time_embed.timestep_embedder.linear.0` |

### Boundary Disambiguation

The trailing dot in `.self_attn.` / `.cross_attn.` / `.mlp.` prevents false matches on
`adaln_modulation_self_attn` / `adaln_modulation_cross_attn` / `adaln_modulation_mlp`
paths, where the prefix uses underscores (`_self_attn`, `_mlp`) rather than dot-separated
components.

Verified examples:
- `blocks.0.adaln_modulation_self_attn.1` → `.self_attn.` not found → classified as `adaln` ✓
- `blocks.0.adaln_modulation_mlp.1` → `.mlp.` not found (only `_mlp.`) → classified as `adaln` ✓
- `blocks.0.self_attn.qkv_proj` → `.self_attn.` found → classified as `self_attn` ✓

### Helper Function

```python
# In networks/lora_anima/network.py (or config.py)

def _classify_layer(original_name: str) -> str | None:
    """Classify a module path as self_attn / cross_attn / mlp / adaln / None."""
    if ".self_attn." in original_name:
        return "self_attn"
    if ".cross_attn." in original_name:
        return "cross_attn"
    if ".mlp." in original_name:
        return "mlp"
    if "adaln_modulation_" in original_name:
        return "adaln"
    return None
```

### Non-Block Modules

Modules under `PatchEmbed`, `TimestepEmbedding`, and `FinalLayer` do not match any of
the four layer types (`_classify_layer()` returns `None`). They are unaffected by the
four flags and continue to be controlled by `_DEFAULT_EXCLUDE` and user-provided
`exclude_patterns`/`include_patterns`.

## Configuration Flow

### Three-Layer Merge Chain

The four params follow the standard config chain:
```
base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args
```
Method settings override preset settings on overlap (same as existing params).

### `configs/base.toml` (new entries)

```toml
train_self_attn = true
train_cross_attn = true
train_mlp = true
train_adaln = false
```

### `library/config/schema.py`

Register four new bool keys in the config schema validation table.

### `networks/lora_anima/config.py`

Add four fields to `LoRANetworkCfg`:

```python
train_self_attn: bool = True
train_cross_attn: bool = True
train_mlp: bool = True
train_adaln: bool = False
```

Parse from kwargs in `LoRANetworkCfg.from_kwargs()` (alongside `use_ortho`,
`use_moe_style`, etc.).

### `_DEFAULT_EXCLUDE` Change

**Before:**
```python
_DEFAULT_EXCLUDE = (
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)
```

**After:**
```python
_DEFAULT_EXCLUDE = (
    r".*(_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)
```

The `_modulation` alternative is removed. `train_adaln=false` (default) takes over.

### Method Overrides

`configs/methods/*.toml` and `configs/gui-methods/*.toml` can override any subset:

```toml
# Example: "Style only" preset in a hypothetical methods/style_only.toml
train_self_attn = false
train_cross_attn = false
train_mlp = true
train_adaln = false
```

## CLI Surface

### New Arguments

Added in `library/anima/training.py::add_anima_training_arguments()`, following the
existing `--mod_lr` argument:

```python
parser.add_argument(
    "--train_self_attn",
    type=lambda x: x.lower() in ("true", "1", "yes"),
    default=None,
    help="Attach LoRA to self-attention projections. None=config default.",
)
parser.add_argument(
    "--train_cross_attn",
    type=lambda x: x.lower() in ("true", "1", "yes"),
    default=None,
    help="Attach LoRA to cross-attention projections. None=config default.",
)
parser.add_argument(
    "--train_mlp",
    type=lambda x: x.lower() in ("true", "1", "yes"),
    default=None,
    help="Attach LoRA to MLP layers. None=config default.",
)
parser.add_argument(
    "--train_adaln",
    type=lambda x: x.lower() in ("true", "1", "yes"),
    default=None,
    help="Attach LoRA to AdaLN modulation projections. None=config default.",
)
```

`default=None` means the config chain value is used when the flag is absent.

### Usage Examples

```bash
# Style + poses (override from CLI)
python train.py --method lora --train_cross_attn false

# Character-focused
python train.py --method lora --train_self_attn false --train_mlp true --train_cross_attn true

# AdaLN experimental
python train.py --method lora --train_adaln true
```

## GUI Integration

### `gui/__init__.py` — `_GROUPS` Dictionary

Add the four keys to the **Architecture** group (alongside `use_ortho`,
`use_moe_style`, etc.):

```python
# In _GROUPS["Architecture"]["fields"] (conceptual — actual structure is a flat list):
"train_self_attn",
"train_cross_attn",
"train_mlp",
"train_adaln",
```

The existing `_widget()` function automatically renders bool keys as `QCheckBox`. No
custom widget code needed.

### Behavior

- GUI reads the merged config (base.toml → preset → gui-methods/<variant>.toml).
- Defaults from base.toml are shown for new variants.
- Existing variants inherit the defaults unless their TOML is edited.
- Users toggle checkboxes; on save, the values are written to the variant TOML.

## Workflow Frontend Integration

### `workflow/schemas/train_common.yaml`

Add four bool fields to the `training` group (alongside `learning_rate`,
`max_train_epochs`, etc.):

```yaml
- key: train_self_attn
  type: bool
  required: false
  layer: common
  default: true
  label: "训练 Self-Attn"
  help: "对 self-attention 投影层附加 LoRA（空间构图/姿势/布局）"
- key: train_cross_attn
  type: bool
  required: false
  layer: common
  default: true
  label: "训练 Cross-Attn"
  help: "对 cross-attention 投影层附加 LoRA（文本-图像绑定）"
- key: train_mlp
  type: bool
  required: false
  layer: common
  default: true
  label: "训练 MLP"
  help: "对 MLP 层附加 LoRA（视觉特征/纹理/色彩/渲染风格）"
- key: train_adaln
  type: bool
  required: false
  layer: common
  default: false
  label: "训练 AdaLN"
  help: "对 AdaLN 调制投影附加 LoRA（时间步条件）— 默认冻结以保持稳定性"
```

### `workflow/stages/train.py` — Bool Passing Change

Add a whitelist set:

```python
_BOOL_VALUE_KEYS = {"train_self_attn", "train_cross_attn", "train_mlp", "train_adaln"}
```

Modify `_build_train_cmd()` bool handling:

```python
elif isinstance(value, bool):
    if key in _BOOL_VALUE_KEYS:
        cmd.append(f"--{key}")
        cmd.append("true" if value else "false")
    elif value:
        cmd.append(f"--{key}")
```

This ensures the workflow can pass both `true` and `false` for the four new params,
overriding any base.toml defaults. Other bool params retain the existing `store_true`
flag-only behavior.

### i18n Labels

The workflow i18n structure places field labels under
`train_common.field.<key>` and help text under `train_common.help.<key>`. Add four
entries to each locale file (`zh-CN.json`, `en.json`, `ja.json`):

**`workflow/i18n/locales/zh-CN.json`** (in `train_common.field`):
```json
"train_self_attn": "训练 Self-Attn",
"train_cross_attn": "训练 Cross-Attn",
"train_mlp": "训练 MLP",
"train_adaln": "训练 AdaLN"
```

And in `train_common.help`:
```json
"train_self_attn": "对 self-attention 投影层附加 LoRA（空间构图/姿势/布局）",
"train_cross_attn": "对 cross-attention 投影层附加 LoRA（文本-图像绑定）",
"train_mlp": "对 MLP 层附加 LoRA（视觉特征/纹理/色彩/渲染风格）",
"train_adaln": "对 AdaLN 调制投影附加 LoRA（时间步条件）— 默认冻结以保持稳定性"
```

Equivalent entries in `en.json` and `ja.json` follow the same key structure with
translated strings.

## Filtering Implementation

### Location

In `networks/lora_anima/network.py::create_modules()`, inside the candidate module
loop, **after** the existing `exclude_patterns`/`include_patterns`/`layer_start`/
`layer_end` filters and **before** LoRA module instantiation.

### Filter Step

```python
# After existing exclude/include/layer_start_end checks:

layer_kind = _classify_layer(original_name)
if layer_kind == "self_attn" and not cfg.train_self_attn:
    skipped_by_target["self_attn"] += 1
    continue
if layer_kind == "cross_attn" and not cfg.train_cross_attn:
    skipped_by_target["cross_attn"] += 1
    continue
if layer_kind == "mlp" and not cfg.train_mlp:
    skipped_by_target["mlp"] += 1
    continue
if layer_kind == "adaln" and not cfg.train_adaln:
    skipped_by_target["adaln"] += 1
    continue
# layer_kind is None → not a targetable layer, pass through
```

### Logging

After `create_modules()` completes, emit an INFO-level summary:

```python
logger.info(
    "Layer targeting: self_attn=%s (%d attached), cross_attn=%s (%d attached), "
    "mlp=%s (%d attached), adaln=%s (%d attached)",
    cfg.train_self_attn, attached["self_attn"],
    cfg.train_cross_attn, attached["cross_attn"],
    cfg.train_mlp, attached["mlp"],
    cfg.train_adaln, attached["adaln"],
)
if any(skipped_by_target.values()):
    logger.info(
        "Skipped by layer targeting: %s",
        ", ".join(f"{k}={v}" for k, v in skipped_by_target.items() if v > 0),
    )
```

## torch.compile Compatibility

### Why No Recompilation

1. **Filtering is static.** The four config values are read from the config chain at
   startup and stored in `LoRANetworkCfg`. They do not change during training.

2. **Filtering happens before `apply_to()`.** By the time `compile_blocks()` runs
   (in `train.py`, after `network.apply_to()` + `load_weights()`), the set of LoRA
   modules attached to each Block is fixed. Each Block's `_forward` method has a
   deterministic set of LoRA forward calls.

3. **Different configs → different graphs, but one compile per task.** A training run
   with `train_self_attn=false` produces a different `Block._forward` graph than the
   default, but Dynamo compiles it exactly once. The Dynamo cache-size budget formula
   (`2 * n + 8` where `n` = token-count families) is unaffected — it keys on token
   count, not on LoRA module count.

4. **No dynamic dispatch.** LoRA modules use static boolean attributes
   (`use_custom_down_autograd`, `enabled`, `_fused`) that Dynamo sees as static Python
   branches. The layer-targeting flags do not introduce any new dynamic dispatch.

### What Could Go Wrong (And Why It Won't)

- **Hot-reloading config mid-training?** Not supported. The four values are read once
  at startup. No mechanism exists to change them during training.
- **Mixing compiled/uncompiled blocks?** `compile_blocks()` compiles all blocks
  uniformly. A block with no LoRA modules (e.g., if all four flags are false for that
  block's layers) still gets compiled — it just has fewer LoRA forward calls in its
  graph.

## Test Strategy

### New File: `tests/test_layer_targeting.py`

#### Test 1: Default Config Equivalence

```python
def test_default_config_matches_baseline():
    """train_self_attn=true, train_cross_attn=true, train_mlp=true, train_adaln=false
    produces the same LoRA module set as the pre-change default behavior
    (when _DEFAULT_EXCLUDE contained _modulation)."""
```

#### Test 2: Per-Flag Isolation

```python
@pytest.mark.parametrize("flag,kind,path_fragments", [
    ("train_self_attn", "self_attn", ["self_attn.qkv_proj", "self_attn.output_proj"]),
    ("train_cross_attn", "cross_attn",
     ["cross_attn.q_proj", "cross_attn.kv_proj", "cross_attn.output_proj"]),
    ("train_mlp", "mlp", ["mlp.layer1", "mlp.layer2"]),
    ("train_adaln", "adaln",
     ["adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp"]),
])
def test_disabling_flag_removes_only_that_kind(flag, kind, path_fragments):
    """Setting one flag to false removes only that layer type's modules."""
```

#### Test 3: All-False Minimal Set

```python
def test_all_false_produces_no_block_loras():
    """All four flags false → no LoRA modules on any Block Linear.
    (PatchEmbed Conv2d may still get one, depending on _DEFAULT_EXCLUDE.)"""
```

#### Test 4: AdaLN Enabled

```python
def test_adaln_enabled_includes_modulation_layers():
    """train_adaln=true → adaln_modulation_* Linear modules are included."""
```

#### Test 5: Non-Block Modules Unaffected

```python
def test_non_block_modules_unaffected_by_flags():
    """PatchEmbed Conv2d, TimestepEmbedding Linear are not affected by the four flags.
    Their inclusion is controlled solely by _DEFAULT_EXCLUDE and exclude_patterns."""
```

#### Test 6: Classification Unit Tests

```python
@pytest.mark.parametrize("name,expected", [
    ("blocks.0.self_attn.qkv_proj", "self_attn"),
    ("blocks.0.self_attn.output_proj", "self_attn"),
    ("blocks.0.cross_attn.q_proj", "cross_attn"),
    ("blocks.0.cross_attn.kv_proj", "cross_attn"),
    ("blocks.0.cross_attn.output_proj", "cross_attn"),
    ("blocks.0.mlp.layer1", "mlp"),
    ("blocks.0.mlp.layer2", "mlp"),
    ("blocks.0.adaln_modulation_self_attn.1", "adaln"),
    ("blocks.0.adaln_modulation_cross_attn.1", "adaln"),
    ("blocks.0.adaln_modulation_mlp.1", "adaln"),
    ("patch_embed.proj", None),
    ("time_embed.timestep_embedder.linear.0", None),
    ("final_layer.linear", None),
])
def test_classify_layer(name, expected):
    assert _classify_layer(name) == expected
```

#### Test 7: Config Chain Override

```python
def test_method_overrides_base():
    """Method TOML's train_* values override base.toml's."""
    # Use load_method_preset() and check provenance
```

#### Test 8: Workflow Bool Passing

```python
def test_workflow_bool_value_keys_always_passed():
    """_build_train_cmd() passes --key true/false for keys in _BOOL_VALUE_KEYS,
    even when the value is False."""
```

## Files Changed

| File                                        | Change                                                         | Risk   |
|---------------------------------------------|----------------------------------------------------------------|--------|
| `configs/base.toml`                         | Add 4 default entries                                          | Low    |
| `library/config/schema.py`                  | Register 4 bool keys                                           | Low    |
| `networks/lora_anima/config.py`             | Add 4 fields to `LoRANetworkCfg`; parse in `from_kwargs()`; remove `_modulation` from `_DEFAULT_EXCLUDE` | Medium |
| `networks/lora_anima/network.py`            | Add `_classify_layer()`; add filter step + logging in `create_modules()` | Medium |
| `library/anima/training.py`                 | Add 4 argparse arguments in `add_anima_training_arguments()`   | Low    |
| `workflow/schemas/train_common.yaml`        | Add 4 bool fields to `training` group                           | Low    |
| `workflow/stages/train.py`                  | Add `_BOOL_VALUE_KEYS` set; modify `_build_train_cmd()` bool handling | Medium |
| `gui/__init__.py`                           | Add 4 keys to `_GROUPS["Architecture"]`                        | Low    |
| `tests/test_layer_targeting.py`             | New test file (8 test functions)                               | Low    |
| `workflow/i18n/locales/{en,ja,zh-CN}.json`  | Add i18n labels for 4 new fields                                | Low    |

## Out of Scope

- **ComfyUI trainer node (`custom_nodes/comfyui-anima-trainer/`)**: Not modified. The
  node will inherit the new defaults from `base.toml` through its existing
  `build_training_namespace()` path, but the node UI will not expose the four
  checkboxes.
- **Text encoder layer targeting**: `llm_adapter` is explicitly out of scope per the
  original Anima author's recommendation. The existing `llm_adapter_lr` /
  `cache_llm_adapter_outputs` controls remain unchanged.
- **Per-layer LR interaction**: If `train_self_attn=false`, no LoRA modules are
  attached to self-attention layers, so `self_attn_lr` has no LoRA params to apply to.
  This is silently ignored (no error). The existing per-layer LR mechanism
  (`get_anima_param_groups()`) operates on DiT base params, which are frozen during
  LoRA training, so this interaction is benign.
- **`networks/lycoris`/`lycoris_utils.py`**: LyCORIS methods (LoHA/LoKR/LoCon) use the
  same `networks.lora_anima` network module entry point with `network_type` in
  `network_args`. The filtering in `create_modules()` applies uniformly. No separate
  LyCORIS-side change needed.
