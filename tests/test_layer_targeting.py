"""Tests for layer-type classification and targeted LoRA filtering."""

from __future__ import annotations

import pytest

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import _classify_layer
from networks.lora_modules import LoRAModule


@pytest.mark.parametrize("name,expected", [
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
])
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


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workflow.stages.train import TrainExecutor


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
    assert cmd[idx + 1] == "false", f"Expected 'false', got {cmd[idx+1]}"
    idx = cmd.index("--train_cross_attn")
    assert cmd[idx + 1] == "true"
    idx = cmd.index("--train_mlp")
    assert cmd[idx + 1] == "false"
    idx = cmd.index("--train_adaln")
    assert cmd[idx + 1] == "true"

    assert "--torch_compile" not in cmd, \
        "Existing store_true bools should not be passed when false"


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
    assert cmd[idx + 1].startswith("--"), \
        "Existing bools should not pass a value after the flag"
