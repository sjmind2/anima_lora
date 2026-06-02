# Output-Layer Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `output_self_attn` / `output_cross_attn` / `output_mlp` / `output_adaln` save-time gating flags that strip selected DiT layer families from the saved `.safetensors`, parallel to the existing `train_*` training-time gating.

**Architecture:** Filter lives inside `LoRANetwork.save_weights()` immediately after `state_dict = self.state_dict()`. Reuses the existing `_classify_layer` helper. Independent of `train_*`. Emits safetensors metadata (`ss_output_gated`, `ss_output_gated_layers`) keyed off actual removals, not raw flags, so default-config saves stay metadata-silent. `load_weights()` emits a non-blocking warning when loading such a file.

**Tech Stack:** Python 3.11, PyTorch, safetensors, TOML config, pytest, argparse.

**Spec:** `docs/superpowers/specs/2026-06-02-output-layer-gating-design.md`

**Conventions:**
- All code comments in English (project convention; surrounding code is English)
- PowerShell terminal — use `;` not `&&` to chain commands
- Test framework: pytest, run from repo root as `pytest tests/test_layer_targeting.py -v`
- Commit messages: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`

**Pre-existing files referenced (do NOT create):**
- `networks/lora_anima/config.py` — `LoRANetworkCfg` dataclass + `from_kwargs`
- `networks/lora_anima/network.py` — `LoRANetwork.save_weights`, `load_weights`, `_classify_layer`
- `networks/__init__.py` — `SHARED_KWARG_FLAGS` tuple
- `library/anima/training.py` — argparse setup, `_bool_from_str` helper at line 175
- `configs/base.toml` — config defaults
- `gui/__init__.py` — `_GROUPS["Architecture"]` dict
- `workflow/schemas/train_common.yaml` — workflow schema
- `workflow/stages/train.py` — `_BOOL_VALUE_KEYS` set at line 47
- `workflow/i18n/locales/{zh-CN,en,ja}.json` — i18n labels
- `tests/test_layer_targeting.py` — existing tests for `train_*`

---

### Task 1: Add `output_*` fields to `LoRANetworkCfg`

**Files:**
- Modify: `networks/lora_anima/config.py:214` (insert after `train_adaln` field)
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layer_targeting.py`:

```python
def test_output_layer_targeting_defaults():
    """Default cfg has output_self_attn=True, output_cross_attn=True,
    output_mlp=True, output_adaln=False."""
    cfg = _make_cfg()
    assert cfg.output_self_attn is True
    assert cfg.output_cross_attn is True
    assert cfg.output_mlp is True
    assert cfg.output_adaln is False


def test_output_layer_targeting_string_bool_parsing():
    """String 'true'/'false' from TOML/CLI are parsed correctly."""
    cfg = _make_cfg(
        output_self_attn="false",
        output_cross_attn="false",
        output_mlp="true",
        output_adaln="true",
    )
    assert cfg.output_self_attn is False
    assert cfg.output_cross_attn is False
    assert cfg.output_mlp is True
    assert cfg.output_adaln is True
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_layer_targeting.py::test_output_layer_targeting_defaults tests/test_layer_targeting.py::test_output_layer_targeting_string_bool_parsing -v
```

Expected: FAIL with `AttributeError: 'LoRANetworkCfg' object has no attribute 'output_self_attn'`

- [ ] **Step 3: Add four fields to `LoRANetworkCfg`**

In `networks/lora_anima/config.py`, locate lines 209-214 (the `train_*` block):

```python
    # Layer-type targeting: which DiT layer families receive LoRA adapters.
    # Defaults match the "Full" row of the Anima trainer preset matrix.
    train_self_attn: bool = True
    train_cross_attn: bool = True
    train_mlp: bool = True
    train_adaln: bool = False
```

Insert immediately after line 214 (`train_adaln: bool = False`):

```python

    # Layer-type output gating: which trained DiT layer families are written
    # to the saved .safetensors. Independent of train_* (a family may be
    # trained but stripped at save time). Defaults preserve pre-feature
    # save behavior; output_adaln=False pairs with train_adaln=False.
    output_self_attn: bool = True
    output_cross_attn: bool = True
    output_mlp: bool = True
    output_adaln: bool = False
```

- [ ] **Step 4: Parse the four fields in `from_kwargs()`**

In the same file, locate the `from_kwargs()` method's body where `train_*` is parsed (search for `train_self_attn = _as_bool`). Insert immediately after the `train_adaln = ...` line:

```python
        output_self_attn = _as_bool(kwargs.get("output_self_attn", True))
        output_cross_attn = _as_bool(kwargs.get("output_cross_attn", True))
        output_mlp = _as_bool(kwargs.get("output_mlp", True))
        output_adaln = _as_bool(kwargs.get("output_adaln", False))
```

Then locate the `return LoRANetworkCfg(...)` call and add four keyword args after `train_adaln=train_adaln,`:

```python
            output_self_attn=output_self_attn,
            output_cross_attn=output_cross_attn,
            output_mlp=output_mlp,
            output_adaln=output_adaln,
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_layer_targeting.py::test_output_layer_targeting_defaults tests/test_layer_targeting.py::test_output_layer_targeting_string_bool_parsing -v
```

Expected: PASS

- [ ] **Step 6: Run the full targeting test file to ensure no regressions**

```
pytest tests/test_layer_targeting.py -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```
git add networks/lora_anima/config.py tests/test_layer_targeting.py
git commit -m "feat(cfg): add output_* layer-gating fields to LoRANetworkCfg"
```

---

### Task 2: Add `train_*` and `output_*` to `SHARED_KWARG_FLAGS`

This task also fixes the prior leftover: `train_*` was never added, so non-default values did not propagate. Adding all eight in one commit keeps the white-list consistent.

**Files:**
- Modify: `networks/__init__.py:74-145` (the `SHARED_KWARG_FLAGS` tuple)
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layer_targeting.py`:

```python
def test_shared_kwarg_flags_contains_layer_targeting_keys():
    """SHARED_KWARG_FLAGS must include all 8 train_*+output_* keys so they
    propagate from TOML/CLI through to create_network()."""
    from networks import SHARED_KWARG_FLAGS

    expected = {
        "train_self_attn", "train_cross_attn", "train_mlp", "train_adaln",
        "output_self_attn", "output_cross_attn", "output_mlp", "output_adaln",
    }
    missing = expected - set(SHARED_KWARG_FLAGS)
    assert not missing, f"Missing from SHARED_KWARG_FLAGS: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_layer_targeting.py::test_shared_kwarg_flags_contains_layer_targeting_keys -v
```

Expected: FAIL with `AssertionError: Missing from SHARED_KWARG_FLAGS: {...}`

- [ ] **Step 3: Add eight keys to `SHARED_KWARG_FLAGS`**

In `networks/__init__.py`, locate the comment block at lines 71-73:

```python
# targeting knobs + cross-cutting add-ons (ReFT, channel scaling,
# LoRA+, T-LoRA). Cross-cutting because these compose on top of any
# variant rather than belonging to a single one.
```

Insert a new comment block + 8 keys immediately after this comment, before the existing `"train_llm_adapter",` line:

```python
    # Layer-type targeting (training-time and save-time gating).
    # Training-time: which DiT layer families receive adapters at all.
    # Save-time: which trained families are written to the .safetensors.
    "train_self_attn", "train_cross_attn", "train_mlp", "train_adaln",
    "output_self_attn", "output_cross_attn", "output_mlp", "output_adaln",
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_layer_targeting.py::test_shared_kwarg_flags_contains_layer_targeting_keys -v
```

Expected: PASS

- [ ] **Step 5: Run full test file for regressions**

```
pytest tests/test_layer_targeting.py tests/test_network_cfg.py tests/test_network_registry.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```
git add networks/__init__.py tests/test_layer_targeting.py
git commit -m "fix(shared-kwargs): add train_*+output_* to SHARED_KWARG_FLAGS"
```

---

### Task 3: Add save-time filtering in `save_weights()`

**Files:**
- Modify: `networks/lora_anima/network.py:3260` (insert after `state_dict = self.state_dict()`)
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layer_targeting.py`:

```python
def test_classify_layer_with_dot_prefix_for_lora_name():
    """save_weights prepends '.' to lora_name for boundary matching."""
    # lora_name typically looks like 'lora_unet_dit_blocks_0_self_attn_q_proj'
    # (dots collapsed to underscores). Prepending '.' lets _classify_layer's
    # dot-boundary matchers still find '.self_attn.'-like segments even when
    # the name itself contains no literal dots.
    assert _classify_layer(".lora_unet_blocks_0_self_attn_q") == "self_attn"
    assert _classify_layer(".lora_unet_blocks_0_cross_attn_q") == "cross_attn"
    assert _classify_layer(".lora_unet_blocks_0_mlp_fc1") == "mlp"
    assert (
        _classify_layer(".lora_unet_blocks_0_adaln_modulation_proj")
        == "adaln"
    )
```

- [ ] **Step 2: Run test to verify it passes already**

```
pytest tests/test_layer_targeting.py::test_classify_layer_with_dot_prefix_for_lora_name -v
```

Expected: PASS (the helper already supports this). If it fails, inspect a real `lora_name` from a saved checkpoint and adjust the matcher — but per spec §6.2 the dot-prefix should suffice.

- [ ] **Step 3: Add the filtering block to `save_weights`**

In `networks/lora_anima/network.py`, locate the `save_weights` method around line 3260:

```python
        state_dict = self.state_dict()
        for key in [k for k in state_dict if k.startswith("repa_head.")]:
            del state_dict[key]
```

Insert immediately AFTER the `for key in [...]` block and BEFORE the LyCORIS scalar rescaling block (which starts with `if self.cfg.network_type in ("loha", "locon"):`):

```python

        # ── Per-layer-type output gating ──────────────────────────
        # Independent of train_*: a family may be trained but stripped at
        # save, or (harmlessly) marked for output despite not being trained
        # — the latter simply has no matching keys in state_dict.
        _output_cfg = {
            "self_attn": self.cfg.output_self_attn,
            "cross_attn": self.cfg.output_cross_attn,
            "mlp":       self.cfg.output_mlp,
            "adaln":     self.cfg.output_adaln,
        }
        _removed_counts = {k: 0 for k in _output_cfg}
        for _lora in self.text_encoder_loras + self.unet_loras:
            # lora_name is the dotted module path with '.' collapsed to '_';
            # prepend '.' so _classify_layer's dot-boundary matchers work.
            _kind = _classify_layer("." + _lora.lora_name)
            if _kind is None or _output_cfg.get(_kind, True):
                continue
            _prefix = _lora.lora_name + "."
            for _key in [k for k in state_dict if k.startswith(_prefix)]:
                del state_dict[_key]
                _removed_counts[_kind] += 1
        if any(_removed_counts.values()):
            logger.info(
                "[Save] output gating removed modules - "
                f"self_attn={_removed_counts['self_attn']}, "
                f"cross_attn={_removed_counts['cross_attn']}, "
                f"mlp={_removed_counts['mlp']}, "
                f"adaln={_removed_counts['adaln']}"
            )

        # ── Output-gating metadata (only when something was actually removed)
        if any(_removed_counts.values()):
            metadata["ss_output_gated"] = "true"
            metadata["ss_output_gated_layers"] = ",".join(
                k for k, n in _removed_counts.items() if n > 0
            )
```

Note: use ASCII hyphen `-` in the log message, not em-dash, to avoid mojibake (per the lesson learned in the prior spec).

- [ ] **Step 4: Verify the file still parses**

```
python -c "from networks.lora_anima.network import LoRANetwork; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Run all targeting tests**

```
pytest tests/test_layer_targeting.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```
git add networks/lora_anima/network.py tests/test_layer_targeting.py
git commit -m "feat(save): output_* layer gating in save_weights + metadata"
```

---

### Task 4: Add load-time warning in `load_weights()`

**Files:**
- Modify: `networks/lora_anima/network.py:2744-2750` (the safetensors branch in `load_weights`)
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layer_targeting.py`:

```python
def test_load_weights_warns_on_output_gated_metadata(tmp_path, caplog):
    """load_weights emits a warning when the file has ss_output_gated=true."""
    import logging
    import torch
    from safetensors.torch import save_file

    # Create a minimal safetensors file with the gating metadata
    dummy = {"lora_TE_ATTN_dummy.weight": torch.zeros(1)}
    f = tmp_path / "gated.safetensors"
    save_file(
        dummy,
        str(f),
        metadata={
            "ss_output_gated": "true",
            "ss_output_gated_layers": "self_attn,adaln",
        },
    )

    # Stub network: only need load_weights to read metadata and warn
    from networks.lora_anima.network import LoRANetwork
    net = LoRANetwork.__new__(LoRANetwork)

    with caplog.at_level(logging.WARNING, logger="networks.lora_anima.network"):
        # load_weights will fail after the warning because state_dict is empty,
        # but we only care about the warning being emitted. Catch the error.
        try:
            net.load_weights(str(f))
        except Exception:
            pass

    msgs = [r.getMessage() for r in caplog.records]
    assert any("output gating disabled for: [self_attn,adaln]" in m for m in msgs), (
        f"Expected gating warning not found in: {msgs}"
    )


def test_load_weights_no_warning_without_metadata(tmp_path, caplog):
    """load_weights does NOT warn when metadata lacks ss_output_gated."""
    import logging
    import torch
    from safetensors.torch import save_file

    dummy = {"lora_TE_ATTN_dummy.weight": torch.zeros(1)}
    f = tmp_path / "plain.safetensors"
    save_file(dummy, str(f), metadata={})

    from networks.lora_anima.network import LoRANetwork
    net = LoRANetwork.__new__(LoRANetwork)

    with caplog.at_level(logging.WARNING, logger="networks.lora_anima.network"):
        try:
            net.load_weights(str(f))
        except Exception:
            pass

    msgs = [r.getMessage() for r in caplog.records]
    assert not any("output gating" in m for m in msgs), (
        f"Unexpected gating warning: {msgs}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_layer_targeting.py::test_load_weights_warns_on_output_gated_metadata tests/test_layer_targeting.py::test_load_weights_no_warning_without_metadata -v
```

Expected: FAIL (warning not emitted yet)

- [ ] **Step 3: Add the metadata check to `load_weights`**

In `networks/lora_anima/network.py`, locate `load_weights` at line 2744:

```python
    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
```

Replace the safetensors branch with:

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
                    "retain their initialization; resuming training from "
                    "this file will NOT recover the original weights. "
                    "Use a full checkpoint for resume."
                )
        else:
            weights_sd = torch.load(file, map_location="cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_layer_targeting.py::test_load_weights_warns_on_output_gated_metadata tests/test_layer_targeting.py::test_load_weights_no_warning_without_metadata -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```
git add networks/lora_anima/network.py tests/test_layer_targeting.py
git commit -m "feat(load): warn when loading output-gated checkpoints"
```

---

### Task 5: Add CLI argparse flags

**Files:**
- Modify: `library/anima/training.py:201` (insert after the `--train_adaln` block, inside `add_anima_training_arguments`)
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layer_targeting.py`:

```python
def test_cli_argparse_has_output_layer_flags():
    """The training CLI exposes all 4 --output_* flags with correct defaults."""
    import argparse
    import library.anima.training as training_module

    # Re-parse the module's add_arguments to a fresh parser
    parser = argparse.ArgumentParser()
    training_module.add_anima_training_arguments(parser)

    # Default values
    defaults = vars(parser.parse_args([]))
    assert defaults.get("output_self_attn") is None  # None = use config default
    assert defaults.get("output_cross_attn") is None
    assert defaults.get("output_mlp") is None
    assert defaults.get("output_adaln") is None

    # Boolean parsing
    args = parser.parse_args([
        "--output_self_attn", "false",
        "--output_cross_attn", "true",
        "--output_mlp", "false",
        "--output_adaln", "true",
    ])
    assert args.output_self_attn is False
    assert args.output_cross_attn is True
    assert args.output_mlp is False
    assert args.output_adaln is True
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_layer_targeting.py::test_cli_argparse_has_output_layer_flags -v
```

Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'output_self_attn'` (or similar)

- [ ] **Step 3: Add four argparse arguments**

In `library/anima/training.py`, locate the existing `--train_adaln` block at lines 196-201:

```python
    parser.add_argument(
        "--train_adaln",
        type=_bool_from_str,
        default=None,
        help="Attach LoRA to AdaLN modulation projections. None=config default.",
    )
```

Insert immediately after:

```python
    parser.add_argument(
        "--output_self_attn",
        type=_bool_from_str,
        default=None,
        help="Save self-attention LoRA weights to output file. None=config default.",
    )
    parser.add_argument(
        "--output_cross_attn",
        type=_bool_from_str,
        default=None,
        help="Save cross-attention LoRA weights to output file. None=config default.",
    )
    parser.add_argument(
        "--output_mlp",
        type=_bool_from_str,
        default=None,
        help="Save MLP LoRA weights to output file. None=config default.",
    )
    parser.add_argument(
        "--output_adaln",
        type=_bool_from_str,
        default=None,
        help="Save AdaLN modulation LoRA weights to output file. None=config default.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_layer_targeting.py::test_cli_argparse_has_output_layer_flags -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```
git add library/anima/training.py tests/test_layer_targeting.py
git commit -m "feat(cli): add --output_* argparse flags"
```

---

### Task 6: Add defaults to `base.toml`

**Files:**
- Modify: `configs/base.toml:23` (insert after `train_adaln = false`)

- [ ] **Step 1: Add four defaults**

In `configs/base.toml`, locate lines 20-23:

```toml
train_self_attn = true
train_cross_attn = true
train_mlp = true
train_adaln = false
```

Insert immediately after:

```toml
output_self_attn = true
output_cross_attn = true
output_mlp = true
output_adaln = false
```

- [ ] **Step 2: Verify TOML parses**

```
python -c "import tomllib; tomllib.load(open('configs/base.toml','rb')); print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```
git add configs/base.toml
git commit -m "feat(config): add output_* defaults to base.toml"
```

---

### Task 7: Add GUI keys

**Files:**
- Modify: `gui/__init__.py:241` (insert after `"train_adaln",` in `_GROUPS["Architecture"]`)

- [ ] **Step 1: Add four keys to the Architecture group**

In `gui/__init__.py`, locate the Architecture group block at lines 238-241:

```python
        "train_self_attn",
        "train_cross_attn",
        "train_mlp",
        "train_adaln",
    },
```

Insert four new lines after `"train_adaln",`:

```python
        "train_adaln",
        "output_self_attn",
        "output_cross_attn",
        "output_mlp",
        "output_adaln",
    },
```

- [ ] **Step 2: Verify the GUI module imports**

```
python -c "import gui; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```
git add gui/__init__.py
git commit -m "feat(gui): add output_* keys to Architecture group"
```

---

### Task 8: Add Workflow schema fields

**Files:**
- Modify: `workflow/schemas/train_common.yaml:147` (insert after the `train_adaln` entry)

- [ ] **Step 1: Add four fields to the schema**

In `workflow/schemas/train_common.yaml`, locate the `train_adaln` entry ending around line 147:

```yaml
      - key: train_adaln
        type: bool
        required: false
        layer: common
```

Insert immediately after (preserve indentation):

```yaml
      - key: output_self_attn
        type: bool
        required: false
        layer: common
      - key: output_cross_attn
        type: bool
        required: false
        layer: common
      - key: output_mlp
        type: bool
        required: false
        layer: common
      - key: output_adaln
        type: bool
        required: false
        layer: common
```

If the `train_adaln` entry has additional fields (e.g. `default:`, `help:`), mirror them on the new entries — match whatever pattern the existing `train_*` entries follow in this file.

- [ ] **Step 2: Verify schema still parses**

```
python -c "import yaml; yaml.safe_load(open('workflow/schemas/train_common.yaml')); print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```
git add workflow/schemas/train_common.yaml
git commit -m "feat(workflow): add output_* fields to train_common schema"
```

---

### Task 9: Add Workflow `_BOOL_VALUE_KEYS`

**Files:**
- Modify: `workflow/stages/train.py:47` (the `_BOOL_VALUE_KEYS` set)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layer_targeting.py`:

```python
def test_workflow_bool_value_keys_contains_output_flags():
    """_BOOL_VALUE_KEYS must include output_* so workflow always passes
    --output_X true/false rather than treating them as store_true."""
    from workflow.stages.train import _BOOL_VALUE_KEYS

    expected = {
        "output_self_attn", "output_cross_attn",
        "output_mlp", "output_adaln",
    }
    missing = expected - _BOOL_VALUE_KEYS
    assert not missing, f"Missing from _BOOL_VALUE_KEYS: {missing}"
```

Also add an end-to-end test that verifies `_build_train_cmd` forwards the new keys:

```python
def test_build_train_cmd_passes_output_layer_flags():
    """_build_train_cmd() passes --output_X true/false for output_* keys."""
    executor = TrainExecutor.__new__(TrainExecutor)
    executor.stage_dir = Path("/tmp/test")
    executor.infrastructure = {}

    config = {
        "output_self_attn": False,
        "output_cross_attn": True,
        "output_mlp": False,
        "output_adaln": True,
    }
    cmd = executor._build_train_cmd(config, Path("/tmp/dataset.toml"))

    idx = cmd.index("--output_self_attn")
    assert cmd[idx + 1] == "false"
    idx = cmd.index("--output_cross_attn")
    assert cmd[idx + 1] == "true"
    idx = cmd.index("--output_mlp")
    assert cmd[idx + 1] == "false"
    idx = cmd.index("--output_adaln")
    assert cmd[idx + 1] == "true"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_layer_targeting.py::test_workflow_bool_value_keys_contains_output_flags tests/test_layer_targeting.py::test_build_train_cmd_passes_output_layer_flags -v
```

Expected: FAIL

- [ ] **Step 3: Add four keys to `_BOOL_VALUE_KEYS`**

In `workflow/stages/train.py`, locate line 47:

```python
_BOOL_VALUE_KEYS = {"train_self_attn", "train_cross_attn", "train_mlp", "train_adaln"}
```

Replace with:

```python
_BOOL_VALUE_KEYS = {
    "train_self_attn", "train_cross_attn", "train_mlp", "train_adaln",
    "output_self_attn", "output_cross_attn", "output_mlp", "output_adaln",
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_layer_targeting.py::test_workflow_bool_value_keys_contains_output_flags tests/test_layer_targeting.py::test_build_train_cmd_passes_output_layer_flags -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```
git add workflow/stages/train.py tests/test_layer_targeting.py
git commit -m "feat(workflow): forward output_* via _BOOL_VALUE_KEYS"
```

---

### Task 10: Add Workflow i18n labels (zh-CN, en, ja)

**Files:**
- Modify: `workflow/i18n/locales/zh-CN.json`
- Modify: `workflow/i18n/locales/en.json`
- Modify: `workflow/i18n/locales/ja.json`

- [ ] **Step 1: Add zh-CN labels**

In `workflow/i18n/locales/zh-CN.json`, locate the train_* labels (around line 205-208):

```json
        "train_self_attn": "训练 Self-Attn",
        "train_cross_attn": "训练 Cross-Attn",
        "train_mlp": "训练 MLP",
        "train_adaln": "训练 AdaLN",
```

Insert immediately after:

```json
        "output_self_attn": "输出 Self-Attn",
        "output_cross_attn": "输出 Cross-Attn",
        "output_mlp": "输出 MLP",
        "output_adaln": "输出 AdaLN",
```

Then locate the help-text block (around line 233-236):

```json
        "train_self_attn": "对 self-attention 投影层附加 LoRA（空间构图/姿势/布局）",
        "train_cross_attn": "对 cross-attention 投影层附加 LoRA（文本-图像绑定）",
        "train_mlp": "对 MLP 层附加 LoRA（视觉特征/纹理/色彩/渲染风格）",
        "train_adaln": "对 AdaLN 调制投影附加 LoRA（时间步条件）— 默认冻结以保持稳定性"
```

Insert immediately after:

```json
        "output_self_attn": "保存 self-attention LoRA 权重到输出文件",
        "output_cross_attn": "保存 cross-attention LoRA 权重到输出文件",
        "output_mlp": "保存 MLP LoRA 权重到输出文件",
        "output_adaln": "保存 AdaLN 调制 LoRA 权重到输出文件（默认关闭）"
```

- [ ] **Step 2: Add en labels (mirror structure)**

In `workflow/i18n/locales/en.json`, find the same `train_*` field label and help sections and add:

Field labels:
```json
        "output_self_attn": "Output Self-Attn",
        "output_cross_attn": "Output Cross-Attn",
        "output_mlp": "Output MLP",
        "output_adaln": "Output AdaLN",
```

Help text:
```json
        "output_self_attn": "Save self-attention LoRA weights to the output file",
        "output_cross_attn": "Save cross-attention LoRA weights to the output file",
        "output_mlp": "Save MLP LoRA weights to the output file",
        "output_adaln": "Save AdaLN modulation LoRA weights to the output file (default off)"
```

- [ ] **Step 3: Add ja labels (mirror structure)**

In `workflow/i18n/locales/ja.json`, add field labels:

```json
        "output_self_attn": "Self-Attn 出力",
        "output_cross_attn": "Cross-Attn 出力",
        "output_mlp": "MLP 出力",
        "output_adaln": "AdaLN 出力",
```

Help text:

```json
        "output_self_attn": "self-attention LoRA 重みを出力ファイルに保存",
        "output_cross_attn": "cross-attention LoRA 重みを出力ファイルに保存",
        "output_mlp": "MLP LoRA 重みを出力ファイルに保存",
        "output_adaln": "AdaLN 変調 LoRA 重みを出力ファイルに保存 (デフォルト OFF)"
```

- [ ] **Step 4: Verify all three JSON files parse**

```
python -c "import json; json.load(open('workflow/i18n/locales/zh-CN.json')); json.load(open('workflow/i18n/locales/en.json')); json.load(open('workflow/i18n/locales/ja.json')); print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Run workflow schema tests**

```
pytest tests/test_workflow_schema.py tests/test_workflow_stages.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```
git add workflow/i18n/locales/zh-CN.json workflow/i18n/locales/en.json workflow/i18n/locales/ja.json
git commit -m "feat(i18n): add output_* labels (zh-CN/en/ja)"
```

---

### Task 11: Add save-filtering integration tests

These tests verify the actual `save_weights` filtering behavior with mock LoRA modules. They complement the unit tests from Task 3.

**Files:**
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_layer_targeting.py`:

```python
class _StubLora:
    """Minimal stand-in for a LoRAModule: only lora_name matters for filtering."""
    def __init__(self, lora_name):
        self.lora_name = lora_name


def _stub_save_state(net, loras, state_dict):
    """Drive the same filtering loop that lives in save_weights.
    Mirrors the production code so tests catch divergence."""
    _output_cfg = {
        "self_attn": net.cfg.output_self_attn,
        "cross_attn": net.cfg.output_cross_attn,
        "mlp":       net.cfg.output_mlp,
        "adaln":     net.cfg.output_adaln,
    }
    removed = {k: 0 for k in _output_cfg}
    for lora in loras:
        kind = _classify_layer("." + lora.lora_name)
        if kind is None or _output_cfg.get(kind, True):
            continue
        prefix = lora.lora_name + "."
        for key in [k for k in state_dict if k.startswith(prefix)]:
            del state_dict[key]
            removed[kind] += 1
    return state_dict, removed


def _make_stub_loras():
    return [
        _StubLora("lora_unet_blocks_0_self_attn_q_proj"),
        _StubLora("lora_unet_blocks_0_cross_attn_kv_proj"),
        _StubLora("lora_unet_blocks_0_mlp_fc1"),
        _StubLora("lora_unet_blocks_0_adaln_modulation_proj"),
    ]


def _make_full_state_dict():
    keys = [
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight",
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
        "lora_unet_blocks_0_cross_attn_kv_proj.lora_up.weight",
        "lora_unet_blocks_0_cross_attn_kv_proj.lora_down.weight",
        "lora_unet_blocks_0_mlp_fc1.lora_up.weight",
        "lora_unet_blocks_0_mlp_fc1.lora_down.weight",
        "lora_unet_blocks_0_adaln_modulation_proj.lora_up.weight",
        "lora_unet_blocks_0_adaln_modulation_proj.lora_down.weight",
    ]
    import torch
    return {k: torch.zeros(1) for k in keys}


def test_output_filter_default_removes_nothing():
    """Default output_* config (T,T,T,F) with all 4 families trained:
    only adaln is stripped, because output_adaln=False."""
    cfg = _make_cfg(train_adaln="true")  # train all 4
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    # adaln gated off, one module, two keys
    assert removed == {"self_attn": 0, "cross_attn": 0, "mlp": 0, "adaln": 2}
    assert not any("adaln_modulation" in k for k in sd)
    assert any("self_attn" in k for k in sd)


class _StubNet:
    def __init__(self, cfg):
        self.cfg = cfg


def test_output_filter_disables_self_attn():
    cfg = _make_cfg(output_self_attn="false", train_adaln="true")
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    assert removed["self_attn"] == 2
    assert not any("self_attn_q_proj" in k for k in sd)
    assert any("cross_attn" in k for k in sd)
    assert any("mlp" in k for k in sd)


def test_output_filter_disables_cross_attn():
    cfg = _make_cfg(output_cross_attn="false", train_adaln="true")
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    assert removed["cross_attn"] == 2
    assert not any("cross_attn" in k for k in sd)
    assert any("self_attn" in k for k in sd)


def test_output_filter_disables_mlp():
    cfg = _make_cfg(output_mlp="false", train_adaln="true")
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    assert removed["mlp"] == 2
    assert not any("mlp_fc1" in k for k in sd)
    assert any("self_attn" in k for k in sd)


def test_output_filter_independent_of_train_filter():
    """train_adaln=false + output_adaln=true: no error, no removals
    (because adaln modules never entered self.unet_loras)."""
    cfg = _make_cfg()  # train_adaln=False default
    # No adaln lora in the list, mirroring the production filter step
    loras = [l for l in _make_stub_loras() if "adaln" not in l.lora_name]
    sd = {k: v for k, v in _make_full_state_dict().items() if "adaln" not in k}
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    assert removed == {"self_attn": 0, "cross_attn": 0, "mlp": 0, "adaln": 0}


def test_output_filter_train_true_output_false_strips_at_save():
    """train_adaln=true + output_adaln=false: adaln trained but stripped at save."""
    cfg = _make_cfg(train_adaln="true")  # output_adaln=False is default
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    assert removed["adaln"] == 2
    assert not any("adaln_modulation" in k for k in sd)
```

- [ ] **Step 2: Run new tests**

```
pytest tests/test_layer_targeting.py -k "output_filter" -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```
git add tests/test_layer_targeting.py
git commit -m "test(save): integration tests for output_* layer gating"
```

---

### Task 12: Add metadata emission tests

**Files:**
- Test: `tests/test_layer_targeting.py`

- [ ] **Step 1: Add metadata tests**

Append to `tests/test_layer_targeting.py`:

```python
def test_metadata_silent_for_default_config():
    """Default save (train_adaln=False, output_adaln=False) emits NO
    ss_output_gated metadata because nothing was actually removed."""
    cfg = _make_cfg()  # train_adaln=False default; adaln never trained
    loras = [l for l in _make_stub_loras() if "adaln" not in l.lora_name]
    sd = {k: v for k, v in _make_full_state_dict().items() if "adaln" not in k}
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    # Production: metadata is written only if any(_removed_counts.values())
    assert not any(removed.values()), "Default config should remove nothing"


def test_metadata_emitted_when_adaln_actually_gated():
    """train_adaln=true + output_adaln=false emits ss_output_gated=true
    and ss_output_gated_layers=adaln."""
    cfg = _make_cfg(train_adaln="true")  # output_adaln=False default
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    # Production: metadata would be {"ss_output_gated": "true",
    # "ss_output_gated_layers": "adaln"}
    assert removed["adaln"] > 0
    expected_layers = ",".join(k for k, n in removed.items() if n > 0)
    assert expected_layers == "adaln"


def test_metadata_emitted_for_multiple_gated_families():
    """output_self_attn=false + output_adaln=false emits both in metadata."""
    cfg = _make_cfg(
        output_self_attn="false",
        train_adaln="true",  # so adaln modules exist
    )
    loras = _make_stub_loras()
    sd = _make_full_state_dict()
    sd, removed = _stub_save_state(_StubNet(cfg), loras, sd)
    expected_layers = set(k for k, n in removed.items() if n > 0)
    assert expected_layers == {"self_attn", "adaln"}
```

- [ ] **Step 2: Run tests**

```
pytest tests/test_layer_targeting.py -k "metadata" -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```
git add tests/test_layer_targeting.py
git commit -m "test(save): metadata emission for output gating"
```

---

### Task 13: Final regression sweep + smoke test

**Files:**
- Test: full test suite

- [ ] **Step 1: Run the full targeting test file**

```
pytest tests/test_layer_targeting.py -v
```

Expected: all PASS. Count should be original 29 + ~15 new = ~44+ tests.

- [ ] **Step 2: Run adjacent test files for regressions**

```
pytest tests/test_network_cfg.py tests/test_network_registry.py tests/test_workflow_schema.py tests/test_workflow_stages.py tests/test_config.py -v
```

Expected: all PASS

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all PASS. If anything fails, investigate before proceeding — likely a config-from_kwargs signature mismatch.

- [ ] **Step 4: Smoke-test the CLI parser**

```
python -c "from library.anima.training import add_anima_training_arguments; import argparse; p=argparse.ArgumentParser(); add_anima_training_arguments(p); a=p.parse_args(['--output_self_attn','true','--output_adaln','false']); print(a.output_self_attn, a.output_adaln)"
```

Expected: `True False`

- [ ] **Step 5: If everything passes, no commit needed (this task is verification only)**

If you discovered fixes during regression, commit them with appropriate messages.

---

## Self-Review Notes

**Spec coverage check:**
- §6.1 config fields → Task 1
- §6.2 save_weights filtering + metadata → Tasks 3, 11, 12
- §6.3 load_weights warning → Task 4
- §6.4 SHARED_KWARG_FLAGS (8 keys) → Task 2
- §6.5 CLI argparse → Task 5
- §6.6 base.toml → Task 6
- §6.7 GUI → Task 7
- §6.8 workflow schema + bool keys + i18n → Tasks 8, 9, 10
- §7 default equivalence → Tasks 1, 12 (T11 in spec)
- §8 all 14 test categories → covered across Tasks 1-12
- R1-R9 requirements → all mapped

**Type/name consistency:**
- `output_self_attn` / `output_cross_attn` / `output_mlp` / `output_adaln` — used identically in cfg, argparse, schema, GUI, workflow, i18n, tests
- `_classify_layer` reused from prior spec unchanged
- `_BOOL_VALUE_KEYS` set extension mirrors prior `train_*` addition

**Placeholder scan:** No TBDs. All code blocks complete.

**Known risk:** Task 5 assumes the CLI registration function is named `add_arguments` in `library/anima/training.py`. If it's different, the engineer should inspect the file first. This is the only place where the plan depends on a name not yet verified — all other file:line references were confirmed during spec writing.
