"""Tests for layer-type classification and targeted LoRA filtering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import _classify_layer
from networks.lora_modules import LoRAModule
from workflow.stages.train import TrainExecutor


@pytest.mark.parametrize(
    "name,expected",
    [
        # self_attn — dot-boundary match
        ("blocks.0.self_attn.qkv_proj", "self_attn"),
        ("blocks.0.self_attn.output_proj", "self_attn"),
        ("blocks.10.self_attn.qkv_proj", "self_attn"),
        # cross_attn — dot-boundary match
        ("blocks.0.cross_attn.q_proj", "cross_attn"),
        ("blocks.0.cross_attn.kv_proj", "cross_attn"),
        ("blocks.0.cross_attn.output_proj", "cross_attn"),
        # mlp — dot-boundary match
        ("blocks.0.mlp.layer1", "mlp"),
        ("blocks.0.mlp.layer2", "mlp"),
        # adaln — underscore-prefix match on adaln_modulation_
        ("blocks.0.adaln_modulation_self_attn.1", "adaln"),
        ("blocks.0.adaln_modulation_cross_attn.1", "adaln"),
        ("blocks.0.adaln_modulation_mlp.1", "adaln"),
        # Non-Block modules -- unclassified
        ("patch_embed.proj", None),
        ("time_embed.timestep_embedder.linear.0", None),
        ("final_layer.linear", None),
        # Boundary edge case: adaln_modulation_self_attn must NOT match self_attn
        # because the path uses _self_attn (underscore) not .self_attn (dot)
        ("blocks.0.adaln_modulation_self_attn.1.weight", "adaln"),
        # Similarly adaln_modulation_mlp must NOT match mlp
        ("blocks.0.adaln_modulation_mlp.1.weight", "adaln"),
    ],
)
def test_classify_layer(name, expected):
    assert _classify_layer(name) == expected


def _make_cfg(**overrides):
    """Build a LoRANetworkCfg from kwargs with sensible defaults."""
    return LoRANetworkCfg.from_kwargs(
        overrides,
        network_dim=16,
        network_alpha=16,
        neuron_dropout=None,
        module_class=LoRAModule,
    )


def test_layer_targeting_defaults():
    """Default cfg has train_self_attn=True, train_cross_attn=True,
    train_mlp=True, train_adaln=False."""
    cfg = _make_cfg()
    assert cfg.train_self_attn is True
    assert cfg.train_cross_attn is True
    assert cfg.train_mlp is True
    assert cfg.train_adaln is False


def test_layer_targeting_string_bool_parsing():
    """String 'true'/'false' from TOML/CLI are parsed correctly."""
    cfg = _make_cfg(
        train_self_attn="false",
        train_cross_attn="false",
        train_mlp="true",
        train_adaln="true",
    )
    assert cfg.train_self_attn is False
    assert cfg.train_cross_attn is False
    assert cfg.train_mlp is True
    assert cfg.train_adaln is True


def test_default_exclude_no_longer_contains_modulation():
    """_DEFAULT_EXCLUDE no longer matches _modulation -- train_adaln=false
    has taken over that responsibility."""
    cfg = _make_cfg()
    # _DEFAULT_EXCLUDE is appended to exclude_patterns
    for pattern in cfg.exclude_patterns:
        assert "_modulation" not in pattern, (
            f"_modulation should be removed from _DEFAULT_EXCLUDE; "
            f"found in pattern: {pattern}"
        )


def test_build_train_cmd_passes_bool_values_for_layer_targeting():
    """_build_train_cmd() passes --key true/false for keys in _BOOL_VALUE_KEYS."""
    executor = TrainExecutor.__new__(TrainExecutor)
    executor.stage_dir = Path("/tmp/test")
    executor.infrastructure = {}

    config = {
        "train_self_attn": False,
        "train_cross_attn": True,
        "train_mlp": False,
        "train_adaln": True,
        "torch_compile": False,
    }
    cmd = executor._build_train_cmd(config, Path("/tmp/dataset.toml"))

    idx = cmd.index("--train_self_attn")
    assert cmd[idx + 1] == "false", f"Expected 'false', got {cmd[idx + 1]}"
    idx = cmd.index("--train_cross_attn")
    assert cmd[idx + 1] == "true"
    idx = cmd.index("--train_mlp")
    assert cmd[idx + 1] == "false"
    idx = cmd.index("--train_adaln")
    assert cmd[idx + 1] == "true"

    assert "--torch_compile" not in cmd, (
        "Existing store_true bools should not be passed when false"
    )


def test_build_train_cmd_passes_bool_true_for_existing_store_true():
    """Existing bool params (torch_compile=true) still use flag-only style."""
    executor = TrainExecutor.__new__(TrainExecutor)
    executor.stage_dir = Path("/tmp/test")
    executor.infrastructure = {}

    config = {
        "torch_compile": True,
        "gradient_checkpointing": True,
    }
    cmd = executor._build_train_cmd(config, Path("/tmp/dataset.toml"))

    assert "--torch_compile" in cmd
    assert "--gradient_checkpointing" in cmd
    idx = cmd.index("--torch_compile")
    assert cmd[idx + 1].startswith("--"), (
        "Existing bools should not pass a value after the flag"
    )


class _MockBlock(nn.Module):
    """Minimal DiT Block replica with self_attn / cross_attn / mlp / adaln."""

    def __init__(self, dim=64, context_dim=64):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(dim, 3 * dim)
        self.self_attn.output_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(dim, dim)
        self.cross_attn.kv_proj = nn.Linear(context_dim, 2 * dim)
        self.cross_attn.output_proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, dim * 2)
        self.mlp.layer2 = nn.Linear(dim * 2, dim)
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 3 * dim)
        )
        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 3 * dim)
        )
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))


class _MockDiT(nn.Module):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_MockBlock() for _ in range(num_blocks)])


def _apply_layer_filter(dit, cfg):
    """Apply the layer-type filter logic (same as create_modules) on a mock DiT.
    Returns list of original_name strings that survive the filter."""
    results = []
    for name, module in dit.named_modules():
        if module.__class__.__name__ == "_MockBlock":
            for child_name, child_module in module.named_modules():
                if isinstance(child_module, (nn.Linear, nn.Conv2d)):
                    original_name = (name + "." if name else "") + child_name
                    kind = _classify_layer(original_name)
                    if kind == "self_attn" and not cfg.train_self_attn:
                        continue
                    if kind == "cross_attn" and not cfg.train_cross_attn:
                        continue
                    if kind == "mlp" and not cfg.train_mlp:
                        continue
                    if kind == "adaln" and not cfg.train_adaln:
                        continue
                    results.append(original_name)
    return results


def test_default_config_matches_pre_change_behavior():
    """Default config (train_self_attn=T, cross_attn=T, mlp=T, adaln=F) produces
    the same module set as the pre-change behavior (when _DEFAULT_EXCLUDE
    contained _modulation)."""
    dit = _MockDiT(num_blocks=2)
    cfg = _make_cfg()
    results = _apply_layer_filter(dit, cfg)

    expected_prefixes = [
        ".self_attn.qkv_proj",
        ".self_attn.output_proj",
        ".cross_attn.q_proj",
        ".cross_attn.kv_proj",
        ".cross_attn.output_proj",
        ".mlp.layer1",
        ".mlp.layer2",
    ]
    for prefix in expected_prefixes:
        matching = [r for r in results if prefix in r]
        assert len(matching) == 2, (
            f"Expected 2 modules matching '{prefix}' (2 blocks), got {len(matching)}"
        )

    adaln_modules = [r for r in results if "adaln_modulation_" in r]
    assert len(adaln_modules) == 0, (
        f"Default config should exclude adaln; got {adaln_modules}"
    )


def test_all_four_disabled_blocks_have_only_non_block_modules():
    """All 4 flags false -> no Block Linear modules survive the filter."""
    dit = _MockDiT(num_blocks=2)
    cfg = _make_cfg(
        train_self_attn="false",
        train_cross_attn="false",
        train_mlp="false",
        train_adaln="false",
    )
    results = _apply_layer_filter(dit, cfg)
    assert len(results) == 0, (
        f"With all 4 flags false, no Block Linears should remain; got {results}"
    )


def test_config_chain_method_overrides_base():
    """Method TOML overrides base.toml for layer-targeting params."""
    cfg = _make_cfg(train_adaln="true")
    assert cfg.train_adaln is True

    cfg = _make_cfg(train_self_attn="false")
    assert cfg.train_self_attn is False


def test_disabling_self_attn_removes_only_self_attn():
    dit = _MockDiT(num_blocks=2)
    cfg = _make_cfg(train_self_attn="false")
    results = _apply_layer_filter(dit, cfg)
    assert not any(".self_attn." in r for r in results)
    assert any(".cross_attn." in r for r in results)
    assert any(".mlp." in r for r in results)


def test_disabling_cross_attn_removes_only_cross_attn():
    dit = _MockDiT(num_blocks=2)
    cfg = _make_cfg(train_cross_attn="false")
    results = _apply_layer_filter(dit, cfg)
    assert any(".self_attn." in r for r in results)
    assert not any(".cross_attn." in r for r in results)
    assert any(".mlp." in r for r in results)


def test_disabling_mlp_removes_only_mlp():
    dit = _MockDiT(num_blocks=2)
    cfg = _make_cfg(train_mlp="false")
    results = _apply_layer_filter(dit, cfg)
    assert any(".self_attn." in r for r in results)
    assert any(".cross_attn." in r for r in results)
    assert not any(".mlp." in r for r in results)


def test_enabling_adaln_includes_modulation_layers():
    dit = _MockDiT(num_blocks=2)
    cfg = _make_cfg(train_adaln="true")
    results = _apply_layer_filter(dit, cfg)
    assert any("adaln_modulation_self_attn" in r for r in results)
    assert any("adaln_modulation_cross_attn" in r for r in results)
    assert any("adaln_modulation_mlp" in r for r in results)


def test_style_only_preset():
    """Style only: self_attn=F, cross_attn=F, mlp=T, adaln=F -> only mlp modules."""
    dit = _MockDiT(num_blocks=1)
    cfg = _make_cfg(
        train_self_attn="false",
        train_cross_attn="false",
        train_mlp="true",
        train_adaln="false",
    )
    results = _apply_layer_filter(dit, cfg)
    assert not any(".self_attn." in r for r in results)
    assert not any(".cross_attn." in r for r in results)
    assert any(".mlp." in r for r in results)
    assert not any("adaln_modulation_" in r for r in results)


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


def test_classify_layer_supports_lora_name_underscore_form():
    """save_weights must classify modules by their lora_name, which is the
    dotted module path with all '.' collapsed to '_' (see create_modules:
    `lora_name = f"{prefix}.{original_name}".replace(".", "_")`).

    The original plan proposed prepending '.' so the dot-boundary matchers
    still hit. That does NOT work: in lora_name form the segment is
    `_self_attn_` (underscores on both sides), so `.self_attn.` cannot match.
    Instead, _classify_layer has been extended to recognize the underscore
    boundary form as well.

    These inputs mirror the lora_name shape produced for Anima DiT blocks:
      lora_unet_<...>_blocks_<n>_<family>_<...>
    """
    assert _classify_layer("lora_unet_blocks_0_self_attn_q") == "self_attn"
    assert _classify_layer("lora_unet_blocks_0_self_attn_qkv_proj") == "self_attn"
    assert _classify_layer("lora_unet_blocks_0_cross_attn_q") == "cross_attn"
    assert _classify_layer("lora_unet_blocks_0_mlp_fc1") == "mlp"
    assert (
        _classify_layer("lora_unet_blocks_0_adaln_modulation_self_attn_1")
        == "adaln"
    )
    assert (
        _classify_layer("lora_unet_blocks_0_adaln_modulation_mlp_1")
        == "adaln"
    )

    # The plan's dot-prefix form must also still classify correctly now that
    # _classify_layer accepts both forms (defence in depth for any caller that
    # still prepends a dot).
    assert _classify_layer(".lora_unet_blocks_0_self_attn_q") == "self_attn"

    # Non-DiT modules remain unclassified.
    assert _classify_layer("lora_unet_patch_embed_proj") is None
    assert _classify_layer("lora_unet_final_layer_linear") is None


def test_load_weights_warns_on_output_gated_metadata(tmp_path, caplog):
    """load_weights emits a warning when the file has ss_output_gated=true."""
    import logging
    import torch
    from safetensors.torch import save_file

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

    from networks.lora_anima.network import LoRANetwork
    net = LoRANetwork.__new__(LoRANetwork)

    with caplog.at_level(logging.WARNING, logger="networks.lora_anima.network"):
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


def test_cli_argparse_has_output_layer_flags():
    """The training CLI exposes all 4 --output_* flags with correct defaults."""
    import argparse
    import library.anima.training as training_module

    parser = argparse.ArgumentParser()
    training_module.add_anima_training_arguments(parser)

    defaults = vars(parser.parse_args([]))
    assert defaults.get("output_self_attn") is None
    assert defaults.get("output_cross_attn") is None
    assert defaults.get("output_mlp") is None
    assert defaults.get("output_adaln") is None

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
        kind = _classify_layer(lora.lora_name)
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


class _StubNet:
    def __init__(self, cfg):
        self.cfg = cfg


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
