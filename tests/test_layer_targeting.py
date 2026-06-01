"""Tests for layer-type classification and targeted LoRA filtering."""

from __future__ import annotations

import pytest

from networks.lora_anima.network import _classify_layer


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
    # Non-Block modules — unclassified
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
