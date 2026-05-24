from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_adapter_module():
    here = Path(__file__).resolve().parent.parent
    node_dir = here / "custom_nodes" / "comfyui-hydralora"

    pkg_name = "_anima_lycoris_test_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(node_dir)]
        sys.modules[pkg_name] = pkg

    adapter_name = f"{pkg_name}.adapter"
    if adapter_name in sys.modules:
        return sys.modules[adapter_name]

    spec = importlib.util.spec_from_file_location(
        adapter_name,
        node_dir / "adapter.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[adapter_name] = mod
    spec.loader.exec_module(mod)
    return mod


_adapter = _load_adapter_module()
_parse_lycoris = _adapter._parse_lycoris
_extract_lora_sd = _adapter._extract_lora_sd


def _t(*shape):
    return torch.randn(*shape)


class TestParseLycoris:
    def test_parse_loha_keys(self):
        sd = {
            "lora_unet_blocks_0_attn_q.hada_w1_a": _t(8),
            "lora_unet_blocks_0_attn_q.hada_w1_b": _t(8, 4),
            "lora_unet_blocks_0_attn_q.hada_w2_a": _t(8),
            "lora_unet_blocks_0_attn_q.hada_w2_b": _t(8, 4),
            "lora_unet_blocks_0_attn_q.alpha": _t(1),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "loha" in result
        prefix = "lora_unet_blocks_0_attn_q"
        assert prefix in result["loha"]
        mod = result["loha"][prefix]
        assert "hada_w1_a" in mod
        assert "hada_w1_b" in mod
        assert "hada_w2_a" in mod
        assert "hada_w2_b" in mod
        assert "alpha" in mod

    def test_parse_loha_with_tucker(self):
        sd = {
            "mod_a.hada_w1_a": _t(4),
            "mod_a.hada_w1_b": _t(4, 2),
            "mod_a.hada_w2_a": _t(4),
            "mod_a.hada_w2_b": _t(4, 2),
            "mod_a.hada_t1": _t(3, 3),
            "mod_a.hada_t2": _t(3, 3),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        mod = result["loha"]["mod_a"]
        assert "hada_t1" in mod
        assert "hada_t2" in mod

    def test_parse_lokr_keys(self):
        sd = {
            "lora_unet_blocks_1_ffn.w1.lokr_w1": _t(16, 4),
            "lora_unet_blocks_1_ffn.w1.lokr_w2": _t(8, 16),
            "lora_unet_blocks_1_ffn.w1.alpha": _t(1),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "lokr" in result
        prefix = "lora_unet_blocks_1_ffn.w1"
        assert prefix in result["lokr"]
        mod = result["lokr"][prefix]
        assert "lokr_w1" in mod
        assert "lokr_w2" in mod
        assert "alpha" in mod

    def test_parse_lokr_factored(self):
        sd = {
            "mod_b.lokr_w1_a": _t(8, 2),
            "mod_b.lokr_w1_b": _t(2, 4),
            "mod_b.lokr_w2_a": _t(6, 3),
            "mod_b.lokr_w2_b": _t(3, 8),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "lokr" in result
        mod = result["lokr"]["mod_b"]
        assert "lokr_w1_a" in mod
        assert "lokr_w1_b" in mod
        assert "lokr_w2_a" in mod
        assert "lokr_w2_b" in mod

    def test_parse_lokr_with_tucker(self):
        sd = {
            "mod_c.lokr_w1": _t(4, 2),
            "mod_c.lokr_t2": _t(3, 3),
            "mod_c.lokr_w2_a": _t(8, 3),
            "mod_c.lokr_w2_b": _t(3, 6),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        mod = result["lokr"]["mod_c"]
        assert "lokr_t2" in mod
        assert "lokr_w2_a" in mod
        assert "lokr_w2_b" in mod

    def test_parse_lokr_dora_scale(self):
        sd = {
            "mod_d.lokr_w1": _t(4, 2),
            "mod_d.lokr_w2": _t(6, 4),
            "mod_d.dora_scale": _t(6),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        mod = result["lokr"]["mod_d"]
        assert "dora_scale" in mod

    def test_parse_locon_tucker_keys(self):
        sd = {
            "lora_unet_blocks_2_proj.lora_mid.weight": _t(4, 4, 1),
            "lora_unet_blocks_2_proj.lora_down.weight": _t(4, 32),
            "lora_unet_blocks_2_proj.lora_up.weight": _t(64, 4),
            "lora_unet_blocks_2_proj.alpha": _t(1),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "locon_tucker" in result
        prefix = "lora_unet_blocks_2_proj"
        assert prefix in result["locon_tucker"]
        mod = result["locon_tucker"][prefix]
        assert "lora_mid.weight" in mod
        assert "lora_down.weight" in mod
        assert "lora_up.weight" in mod
        assert "alpha" in mod

    def test_parse_locon_tucker_with_inv_scale(self):
        sd = {
            "mod_e.lora_mid.weight": _t(2, 2, 1),
            "mod_e.lora_down.weight": _t(2, 8),
            "mod_e.lora_up.weight": _t(16, 2),
            "mod_e.inv_scale": _t(8),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        mod = result["locon_tucker"]["mod_e"]
        assert "inv_scale" in mod

    def test_parse_no_lycoris_returns_none(self):
        sd = {
            "lora_unet_blocks_0_attn_q.lora_down.weight": _t(4, 32),
            "lora_unet_blocks_0_attn_q.lora_up.weight": _t(64, 4),
            "lora_unet_blocks_0_attn_q.alpha": _t(1),
        }
        result = _parse_lycoris(sd)
        assert result is None

    def test_empty_sd_returns_none(self):
        result = _parse_lycoris({})
        assert result is None

    def test_mixed_lora_and_lycoris(self):
        sd = {
            "mod_plain.lora_down.weight": _t(4, 16),
            "mod_plain.lora_up.weight": _t(32, 4),
            "mod_plain.alpha": _t(1),
            "mod_lycoris.hada_w1_a": _t(8),
            "mod_lycoris.hada_w1_b": _t(8, 4),
            "mod_lycoris.hada_w2_a": _t(8),
            "mod_lycoris.hada_w2_b": _t(8, 4),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "loha" in result
        assert "mod_lycoris" in result["loha"]
        assert "mod_plain" not in result.get("loha", {})

    def test_multiple_modules_grouped(self):
        sd = {
            "mod_x.hada_w1_a": _t(4),
            "mod_x.hada_w1_b": _t(4, 2),
            "mod_x.hada_w2_a": _t(4),
            "mod_x.hada_w2_b": _t(4, 2),
            "mod_y.hada_w1_a": _t(6),
            "mod_y.hada_w1_b": _t(6, 3),
            "mod_y.hada_w2_a": _t(6),
            "mod_y.hada_w2_b": _t(6, 3),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "mod_x" in result["loha"]
        assert "mod_y" in result["loha"]
        assert len(result["loha"]) == 2

    def test_mixed_loha_lokr_locon_in_same_sd(self):
        sd = {
            "m_loha.hada_w1_a": _t(4),
            "m_loha.hada_w1_b": _t(4, 2),
            "m_loha.hada_w2_a": _t(4),
            "m_loha.hada_w2_b": _t(4, 2),
            "m_lokr.lokr_w1": _t(8, 2),
            "m_lokr.lokr_w2": _t(4, 8),
            "m_locon.lora_mid.weight": _t(2, 2, 1),
            "m_locon.lora_down.weight": _t(2, 8),
            "m_locon.lora_up.weight": _t(16, 2),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "loha" in result
        assert "lokr" in result
        assert "locon_tucker" in result
        assert "m_loha" in result["loha"]
        assert "m_lokr" in result["lokr"]
        assert "m_locon" in result["locon_tucker"]

    def test_alpha_shared_across_variants(self):
        sd = {
            "shared_mod.hada_w1_a": _t(4),
            "shared_mod.hada_w1_b": _t(4, 2),
            "shared_mod.hada_w2_a": _t(4),
            "shared_mod.hada_w2_b": _t(4, 2),
            "shared_mod.alpha": _t(1),
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "alpha" in result["loha"]["shared_mod"]

    def test_alpha_only_matched_to_existing_prefix(self):
        sd = {
            "orphan_mod.alpha": _t(1),
        }
        result = _parse_lycoris(sd)
        assert result is None

    def test_loha_suffix_stripped_correctly(self):
        val_a = _t(4)
        val_b = _t(4, 2)
        sd = {
            "prefix.sub.hada_w1_a": val_a,
            "prefix.sub.hada_w1_b": val_b,
        }
        result = _parse_lycoris(sd)
        assert result is not None
        assert "prefix.sub" in result["loha"]
        assert result["loha"]["prefix.sub"]["hada_w1_a"] is val_a
        assert result["loha"]["prefix.sub"]["hada_w1_b"] is val_b


class TestExtractLoraSdExclusion:
    def test_lycoris_keys_excluded_from_lora_sd(self):
        lora_up = _t(8, 4)
        sd = {
            "mod_a.lora_down.weight": _t(4, 16),
            "mod_a.lora_up.weight": lora_up,
            "mod_a.alpha": _t(1),
            "mod_a.hada_w1_a": _t(8),
            "mod_a.hada_w1_b": _t(8, 4),
            "mod_a.hada_w2_a": _t(8),
            "mod_a.hada_w2_b": _t(8, 4),
        }
        lycoris_prefixes = {"mod_a"}
        result = _extract_lora_sd(sd, exclude_prefixes=lycoris_prefixes)
        assert result is None

    def test_lycoris_prefix_excludes_all_keys_under_prefix(self):
        sd = {
            "lycoris_mod.lora_down.weight": _t(4, 16),
            "lycoris_mod.alpha": _t(1),
            "lycoris_mod.hada_w1_a": _t(8),
            "lycoris_mod.hada_w1_b": _t(8, 4),
            "normal_mod.lora_down.weight": _t(4, 16),
            "normal_mod.lora_up.weight": _t(32, 4),
            "normal_mod.alpha": _t(1),
        }
        result = _extract_lora_sd(sd, exclude_prefixes={"lycoris_mod"})
        assert result is not None
        assert "normal_mod.lora_down.weight" in result
        assert "normal_mod.lora_up.weight" in result
        assert "normal_mod.alpha" in result
        for key in result:
            assert not key.startswith("lycoris_mod.")

    def test_no_exclude_prefixes_passes_all_lora(self):
        sd = {
            "mod_a.lora_down.weight": _t(4, 16),
            "mod_a.lora_up.weight": _t(32, 4),
            "mod_a.alpha": _t(1),
        }
        result = _extract_lora_sd(sd)
        assert result is not None
        assert len(result) == 3

    def test_empty_exclude_keeps_all(self):
        sd = {
            "mod_a.lora_down.weight": _t(4, 16),
            "mod_a.lora_up.weight": _t(32, 4),
        }
        result = _extract_lora_sd(sd, exclude_prefixes=set())
        assert result is not None
        assert len(result) == 2

    def test_returns_none_when_no_lora_up(self):
        sd = {
            "mod_a.lora_down.weight": _t(4, 16),
            "mod_a.alpha": _t(1),
        }
        result = _extract_lora_sd(sd)
        assert result is None

    def test_reft_keys_excluded(self):
        sd = {
            "mod_a.lora_down.weight": _t(4, 16),
            "mod_a.lora_up.weight": _t(32, 4),
            "reft_unet_blocks_0.rotate_layer.weight": _t(8, 32),
        }
        result = _extract_lora_sd(sd)
        assert result is not None
        assert not any(k.startswith("reft_") for k in result)

    def test_multiple_lycoris_prefixes_excluded(self):
        sd = {
            "loha_mod.lora_down.weight": _t(4, 16),
            "loha_mod.alpha": _t(1),
            "loha_mod.hada_w1_a": _t(8),
            "loha_mod.hada_w1_b": _t(8, 4),
            "lokr_mod.lokr_w1": _t(8, 2),
            "lokr_mod.lokr_w2": _t(4, 8),
            "plain.lora_down.weight": _t(4, 16),
            "plain.lora_up.weight": _t(32, 4),
            "plain.alpha": _t(1),
        }
        result = _extract_lora_sd(sd, exclude_prefixes={"loha_mod", "lokr_mod"})
        assert result is not None
        assert all(not k.startswith(("loha_mod.", "lokr_mod.")) for k in result)
        assert "plain.lora_down.weight" in result
        assert "plain.lora_up.weight" in result
