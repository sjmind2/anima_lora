# Output-Layer Gating for LoRA Saves — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)
**Depends on:** `2026-06-01-layer-targeted-lora-training-design.md` (uses the same
`_classify_layer` helper and parallel CLI/GUI/Workflow surface)
**Follow-up:** Fixes a leftover from the prior spec — `train_*` flags were never
added to `SHARED_KWARG_FLAGS`, so non-default values did not propagate through
the TOML → CLI → `create_network()` chain. This spec adds all eight flags.

---

## 1. Motivation

The layer-targeted training feature (2026-06-01) controls which DiT layer
**families** receive LoRA adapters during training. Symmetrically, users want
to control which families are **written to the saved `.safetensors`** file.
Typical use cases:

- Train all four families (including adaln) but ship a slim LoRA containing
  only `self_attn` + `cross_attn` for an inference runtime that does not yet
  understand adaln projections.
- Train a "Full" preset but ablate by shipping multiple variants
  (`-attn-only`, `-mlp-only`) from a single training run.
- Debug: train with `train_adaln=true` for one cycle, save a no-adaln variant
  to compare inference quality without retraining.

This is purely a **save-time** filter. It never affects:
- The set of parameters updated during backward (governed by `train_*`)
- The set of parameters regularized by `apply_max_norm_regularization`
- Intermediate training checkpoints (they go through the same `save_weights`
  path, but the filter only mutates the temporary `state_dict` dict, not the
  live `nn.Parameter` objects — see §5 Safety)

---

## 2. Requirements

| # | Requirement |
|---|---|
| R1 | Add four boolean knobs: `output_self_attn`, `output_cross_attn`, `output_mlp`, `output_adaln` |
| R2 | Defaults: `True, True, True, False` — byte-identical to current saves for the default `train_*` defaults |
| R3 | Surface in CLI, GUI, and Workflow (mirrors `train_*` placement) |
| R4 | Do not modify anything under `custom_nodes/` |
| R5 | `output_*` is **independent** of `train_*` (no cross-validation, no implicit coupling) |
| R6 | Save-time filtering must not pollute subsequent epochs when `save_every_n_epochs` / `save_every_n_steps` is active |
| R7 | A saved file with output gating must self-describe via safetensors metadata, so that loading it for resume training emits a clear warning |
| R8 | Fix prior leftover: add `train_self_attn`/`train_cross_attn`/`train_mlp`/`train_adaln` to `SHARED_KWARG_FLAGS` so non-default values propagate correctly |
| R9 | Tests must cover: default equivalence, each `output_*=False`, independence from `train_*`, workflow bool forwarding, `SHARED_KWARG_FLAGS` membership, metadata emission, and load warning |

---

## 3. Non-Goals

- **No** runtime "selective load" that silently initializes missing layers.
  Missing layers will keep their fresh init; the warning is informational,
  not blocking.
- **No** changes to `custom_nodes/` (per project policy).
- **No** changes to `apply_max_norm_regularization`, `pre_calculation`,
  `backup_weights`, or any other path that reads `state_dict()` for in-training
  purposes — the filter lives exclusively inside `save_weights()`.
- **No** changes to `.pt` checkpoint paths (those don't carry metadata).

---

## 4. Architecture

```
CLI / TOML / GUI / Workflow
        │
        ▼
SHARED_KWARG_FLAGS (8 new keys: 4 train_* + 4 output_*)
        │
        ▼
factory.create_network() → LoRANetworkCfg.from_kwargs()
        │
        ▼
LoRANetwork.cfg.output_*
        │
        ▼ (at save time)
save_weights():
  state_dict = self.state_dict()      # fresh OrderedDict
  filter by output_* via _classify_layer()
  write metadata: ss_output_gated, ss_output_gated_layers
  save_network_weights(state_dict, ...)

load_weights():
  if metadata["ss_output_gated"] == "true":
      logger.warning(...)
  load_state_dict(weights_sd, False)   # strict=False keeps backward compat
```

The filter reuses the `_classify_layer` helper from the prior spec — single
source of truth for layer-family classification.

---

## 5. Safety: Why Filtering `state_dict()` Doesn't Poison Training

PyTorch's `nn.Module.state_dict()` returns a **new** `OrderedDict` whose
values are tensor references. Deleting a key from this dict removes the
reference from the dict only; the underlying `nn.Parameter` remains attached
to its `LoRAModule`, the module remains in `self.unet_loras`, and the
optimizer state is unchanged.

This is the same pattern the codebase already uses for `repa_head.*`
(see [network.py:3261-3262](file:///o:/loratool/anima_lora_fork/networks/lora_anima/network.py#L3261-L3262))
and for `LohaModule.scalar` rescaling
([network.py:3264-3276](file:///o:/loratool/anima_lora_fork/networks/lora_anima/network.py#L3264-L3276)).
Both predate this change and have never caused training drift.

The intermediate-checkpoint path (`save_every_n_epochs` /
`save_every_n_steps`) routes through `checkpoints.py:414` which calls the
**same** `save_weights()` method — same dictionary-local semantics apply.

---

## 6. Component Changes

### 6.1 `networks/lora_anima/config.py`

Add four fields to `LoRANetworkCfg` immediately after `train_adaln`:

```python
# Layer-type output gating: which DiT layer families are written to the
# saved .safetensors. Independent of train_* (you may train a family but
# strip it at save time). Defaults preserve pre-feature save behavior.
output_self_attn: bool = True
output_cross_attn: bool = True
output_mlp: bool = True
output_adaln: bool = False
```

In `from_kwargs()` parse with the same `_as_bool(kwargs.get(...))` idiom used
for `train_*`:

```python
output_self_attn = _as_bool(kwargs.get("output_self_attn", True))
output_cross_attn = _as_bool(kwargs.get("output_cross_attn", True))
output_mlp = _as_bool(kwargs.get("output_mlp", True))
output_adaln = _as_bool(kwargs.get("output_adaln", False))
```

### 6.2 `networks/lora_anima/network.py::save_weights`

Insert after `state_dict = self.state_dict()` (currently
[network.py:3260](file:///o:/loratool/anima_lora_fork/networks/lora_anima/network.py#L3260)),
before the LyCORIS scalar rescaling block:

```python
# ── Per-layer-type output gating ──────────────────────────────
# Independent of train_*: a family may be trained but stripped at save,
# or (harmlessly) marked for output despite not being trained — the latter
# simply has no matching keys in state_dict.
_output_cfg = {
    "self_attn": self.cfg.output_self_attn,
    "cross_attn": self.cfg.output_cross_attn,
    "mlp":       self.cfg.output_mlp,
    "adaln":     self.cfg.output_adaln,
}
_removed_counts = {k: 0 for k in _output_cfg}
for _lora in self.text_encoder_loras + self.unet_loras:
    # lora_name is the dotted module path with '.' collapsed to '_';
    # prepend '.' so _classify_layer's dot-boundary matchers still work.
    _kind = _classify_layer("." + _lora.lora_name)
    if _kind is None or _output_cfg.get(_kind, True):
        continue
    _prefix = _lora.lora_name + "."
    for _key in [k for k in state_dict if k.startswith(_prefix)]:
        del state_dict[_key]
        _removed_counts[_kind] += 1
if any(_removed_counts.values()):
    logger.info(
        "[Save] output gating removed modules — "
        f"self_attn={_removed_counts['self_attn']}, "
        f"cross_attn={_removed_counts['cross_attn']}, "
        f"mlp={_removed_counts['mlp']}, "
        f"adaln={_removed_counts['adaln']}"
    )
```

**Metadata** (inside the existing metadata block, before
`save_network_weights` is called):

```python
if any(_removed_counts.values()):
    metadata["ss_output_gated"] = "true"
    metadata["ss_output_gated_layers"] = ",".join(
        k for k, n in _removed_counts.items() if n > 0
    )
```

**Important:** metadata is keyed off `_removed_counts` (the number of
modules actually deleted from `state_dict`), **not** off the raw
`output_*` flags. A family that was never trained produces zero
removals and is omitted from metadata. This keeps default-config saves
metadata-silent: `train_adaln=false` → no adaln modules in
`self.unet_loras` → `_removed_counts["adaln"] == 0` → no metadata
key written.

### 6.3 `networks/lora_anima/network.py::load_weights`

Add metadata check at the top of the safetensors branch
([network.py:2744-2750](file:///o:/loratool/anima_lora_fork/networks/lora_anima/network.py#L2744-L2750)):

```python
def load_weights(self, file):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        from safetensors import safe_open

        weights_sd = load_file(file)
        try:
            with safe_open(file, framework="pt") as f:
                _md = dict(f.metadata() or {})
        except Exception:
            _md = {}
        if _md.get("ss_output_gated") == "true":
            _missing = _md.get("ss_output_gated_layers", "")
            logger.warning(
                f"[Load] {os.path.basename(file)} was saved with output "
                f"gating disabled for: [{_missing}]. These layers will "
                "retain their initialization; resuming training from this "
                "file will NOT recover the original weights. Use a full "
                "checkpoint for resume."
            )
    else:
        weights_sd = torch.load(file, map_location="cpu")
    # ... existing _stack_lora_ups / _refuse_* / _reabsorb_baked_inv_scale
```

Warning only — `load_state_dict(weights_sd, False)` keeps `strict=False`,
preserving backward compatibility with all existing checkpoints.

### 6.4 `networks/__init__.py::SHARED_KWARG_FLAGS`

Append eight keys to the tuple at
[`networks/__init__.py:74`](file:///o:/loratool/anima_lora_fork/networks/__init__.py#L74),
in the "Core network targeting / knobs" group:

```python
# Layer-type targeting (training-time and save-time gating).
# Training-time: which DiT layer families receive adapters at all.
# Save-time: which trained families are written to the .safetensors.
"train_self_attn", "train_cross_attn", "train_mlp", "train_adaln",
"output_self_attn", "output_cross_attn", "output_mlp", "output_adaln",
```

This simultaneously fixes the prior leftover (R8).

### 6.5 `library/anima/training.py`

Append four argparse arguments after the existing `--train_*` group, using
the same `_bool_from_str` parser (consistent with the `train_*` flags):

```python
parser.add_argument("--output_self_attn",  type=_bool_from_str, default=True, ...)
parser.add_argument("--output_cross_attn",  type=_bool_from_str, default=True, ...)
parser.add_argument("--output_mlp",         type=_bool_from_str, default=True, ...)
parser.add_argument("--output_adaln",       type=_bool_from_str, default=False, ...)
```

### 6.6 `configs/base.toml`

Append four defaults in the same section as `train_*`:

```toml
output_self_attn = true
output_cross_attn = true
output_mlp        = true
output_adaln      = false
```

### 6.7 `gui/__init__.py`

Append four keys to `_GROUPS["Architecture"]` immediately after the
`train_*` entries.

### 6.8 Workflow (frontend + i18n)

- **`workflow/schemas/train_common.yaml`**: add four boolean fields to the
  `training` group, mirroring the `train_*` shape.
- **`workflow/stages/train.py`**: append the four keys to the
  `_BOOL_VALUE_KEYS` set so they are always forwarded as
  `--output_X true/false` regardless of value (consistent with how
  `train_*` is treated). Existing `store_true` flags are unaffected.
- **`workflow/i18n/locales/{zh-CN,en,ja}.json`**: add `field` and `help`
  entries for each new key, mirroring the `train_*` translations.

---

## 7. Default Equivalence Proof

With defaults `train_*=True,True,True,False` and `output_*=True,True,True,False`:

1. `train_*` filters at module-creation time → adaln modules never enter
   `self.unet_loras`
2. `output_*` iterates over `self.unet_loras` and removes nothing
   (self_attn/cross_attn/mlp all have `output_*=True`)
3. The LyCORIS scalar rescaling block sees the same dict as before
4. `save_network_weights` writes the same bytes as before
5. Metadata is **silent**: `train_adaln=false` → no adaln modules in
   `self.unet_loras` → `_removed_counts["adaln"] == 0` → no
   `ss_output_gated` key written. Default saves are byte-identical
   (modulo pre-existing metadata) to current main.

If the user explicitly sets both `train_adaln=true` and `output_adaln=true`,
adaln modules are trained and saved — metadata stays silent, full set
preserved. If they set `train_adaln=true` but leave `output_adaln=false`,
the trained adaln modules are stripped at save and metadata records the
fact; any subsequent `load_weights()` will warn.

---

## 8. Testing

New tests appended to `tests/test_layer_targeting.py` (the existing file
for the parallel `train_*` feature):

| # | Test | Verifies |
|---|------|----------|
| T1 | `_classify_layer` recognizes all four families when called as `_classify_layer("." + lora_name)` | Helper reuse works for save-time input shape |
| T2 | Default `output_*` config does not remove any trained modules | R2 default equivalence |
| T3 | `output_self_attn=False` removes all `self_attn` keys from state_dict, leaves others | Per-family filter works |
| T4 | Same for `output_cross_attn=False` | "
| T5 | Same for `output_mlp=False` | "
| T6 | Same for `output_adaln=False` (default case) | "
| T7 | `train_adaln=true` + `output_adaln=false`: adaln modules exist in network but not in saved dict | R5 independence |
| T8 | `train_adaln=false` + `output_adaln=true`: no error, no metadata (nothing to remove) | R5 independence reverse direction |
| T9 | `SHARED_KWARG_FLAGS` contains all 8 keys | R8 propagation fix |
| T10 | Workflow `_BOOL_VALUE_KEYS` contains all 4 `output_*` | R3 surfacing |
| T11 | Default-config save does NOT emit `ss_output_gated` metadata (no trained adaln modules → no removals) | R7 metadata cleanliness + R2 default equivalence |
| T12 | `train_adaln=true` + `output_adaln=false` save emits `ss_output_gated=true, ss_output_gated_layers=adaln` | R7 metadata when actually gated |
| T13 | `load_weights` of a file with `ss_output_gated=true` emits exactly one warning containing the gated layer list | R7 load warning |
| T14 | `load_weights` of a file without metadata emits no warning | Backward compat |

Tests use a tiny in-memory `LoRANetwork` fixture (already present in the
file). For T13/T14, write a temporary `.safetensors` with controlled
metadata using `safetensors.torch.save_file`.

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `lora_name` format differs from expectation, `_classify_layer("." + name)` returns `None` for known layers | Medium | T1 covers this; if it fails we switch to direct `_classify_layer(lora_name)` after inspecting a sample key |
| User resumes training from a gated file and silently loses layers | Low (warning is shown) | T13 guarantees the warning; future work could add `--strict_resume` flag |
| `output_X=true` but `train_X=false` confuses users | Low | R5 explicitly accepts this; documentation will state it's a no-op |
| Metadata bloat | Negligible | Only two short keys, only when gating is active |

---

## 10. Open Questions

None at spec approval time. Implementation may discover a `lora_name`
format quirk; T1 will surface it.

---

## 11. Rollout

Single PR. No migrations, no breaking changes, no on-disk format changes.
Default behavior is byte-equivalent to current main (modulo the new
metadata keys, which are additive and ignored by older loaders).
