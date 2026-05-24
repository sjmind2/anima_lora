from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from networks.lora_anima.factory import create_network_from_weights


def _write_safetensors(tmp_path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> Path:
    p = tmp_path / "ckpt.safetensors"
    save_file(tensors, str(p), metadata=metadata)
    return p


class TestFactoryLohaLoad:
    def test_loha_detection_no_nameerror(self, tmp_path: Path):
        prefix = "lora_unet_blocks_0_self_attn_q_proj"
        tensors = {
            f"{prefix}.hada_w1_a": torch.randn(4),
            f"{prefix}.hada_w1_b": torch.randn(8),
            f"{prefix}.hada_w2_a": torch.randn(4),
            f"{prefix}.hada_w2_b": torch.randn(8),
            f"{prefix}.alpha": torch.tensor(4.0),
        }
        path = _write_safetensors(tmp_path, tensors)
        with pytest.raises(Exception) as exc_info:
            create_network_from_weights(1.0, str(path), None, None, None)
        assert not isinstance(exc_info.value, NameError), (
            f"NameError raised during LOHA key scanning — the lora_dim bug is back: {exc_info.value}"
        )

    def test_lokr_detection_no_error(self, tmp_path: Path):
        prefix = "lora_unet_blocks_0_self_attn_q_proj"
        tensors = {
            f"{prefix}.lokr_w1": torch.randn(8, 4),
            f"{prefix}.lokr_w2": torch.randn(4, 8),
            f"{prefix}.alpha": torch.tensor(4.0),
        }
        path = _write_safetensors(tmp_path, tensors)
        with pytest.raises(Exception) as exc_info:
            create_network_from_weights(1.0, str(path), None, None, None)
        assert not isinstance(exc_info.value, NameError), (
            f"NameError raised during LOKR key scanning: {exc_info.value}"
        )

    def test_locon_detection_via_metadata(self, tmp_path: Path):
        prefix = "lora_unet_blocks_0_self_attn_q_proj"
        tensors = {
            f"{prefix}.lora_down.weight": torch.randn(4, 8),
            f"{prefix}.lora_up.weight": torch.randn(8, 4),
            f"{prefix}.alpha": torch.tensor(4.0),
        }
        path = _write_safetensors(tmp_path, tensors, metadata={"ss_network_type": "locon"})
        with pytest.raises(Exception) as exc_info:
            create_network_from_weights(1.0, str(path), None, None, None)
        assert not isinstance(exc_info.value, NameError), (
            f"NameError raised during LOCON key scanning: {exc_info.value}"
        )

    def test_loha_2d_hada_w1_a_sets_modules_dim(self, tmp_path: Path):
        prefix = "lora_unet_blocks_0_self_attn_q_proj"
        tensors = {
            f"{prefix}.hada_w1_a": torch.randn(8, 4),
            f"{prefix}.hada_w1_b": torch.randn(12),
            f"{prefix}.hada_w2_a": torch.randn(8, 4),
            f"{prefix}.hada_w2_b": torch.randn(12),
            f"{prefix}.alpha": torch.tensor(4.0),
        }
        path = _write_safetensors(tmp_path, tensors)
        with pytest.raises(Exception) as exc_info:
            create_network_from_weights(1.0, str(path), None, None, None)
        assert not isinstance(exc_info.value, NameError), (
            f"NameError raised with 2-D hada_w1_a — lora_dim bug regression: {exc_info.value}"
        )

    def test_loha_1d_hada_w1_a_no_nameerror(self, tmp_path: Path):
        prefix = "lora_unet_blocks_0_self_attn_q_proj"
        tensors = {
            f"{prefix}.hada_w1_a": torch.randn(4),
            f"{prefix}.hada_w1_b": torch.randn(12),
            f"{prefix}.alpha": torch.tensor(4.0),
        }
        path = _write_safetensors(tmp_path, tensors)
        with pytest.raises(Exception) as exc_info:
            create_network_from_weights(1.0, str(path), None, None, None)
        assert not isinstance(exc_info.value, NameError), (
            f"NameError raised with 1-D hada_w1_a: {exc_info.value}"
        )

    def test_lokr_w1_a_variant_no_error(self, tmp_path: Path):
        prefix = "lora_unet_blocks_0_self_attn_q_proj"
        tensors = {
            f"{prefix}.lokr_w1_a": torch.randn(8, 4),
            f"{prefix}.lokr_w1_b": torch.randn(12, 4),
            f"{prefix}.lokr_w2": torch.randn(4, 8),
            f"{prefix}.alpha": torch.tensor(4.0),
        }
        path = _write_safetensors(tmp_path, tensors)
        with pytest.raises(Exception) as exc_info:
            create_network_from_weights(1.0, str(path), None, None, None)
        assert not isinstance(exc_info.value, NameError), (
            f"NameError raised during LOKR w1_a variant scanning: {exc_info.value}"
        )
