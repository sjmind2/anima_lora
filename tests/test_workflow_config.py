import pytest
import tempfile
from pathlib import Path
from workflow.config import (
    load_workflow_yaml,
    save_workflow_yaml,
    load_stage_toml,
    save_stage_toml,
    resolve_placeholders,
)


class TestWorkflowYaml:
    def test_load_and_save(self, tmp_path):
        wf_data = {
            "name": "test",
            "stages": [
                {"id": "p1", "type": "preprocess", "config_file": "p1.toml", "depends_on": []},
            ],
        }
        wf_file = tmp_path / "workflow.yaml"
        save_workflow_yaml(wf_data, wf_file)
        loaded = load_workflow_yaml(wf_file)
        assert loaded["name"] == "test"
        assert len(loaded["stages"]) == 1

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_workflow_yaml(tmp_path / "missing.yaml")


class TestStageToml:
    def test_load_and_save(self, tmp_path):
        toml_data = {
            "network_type": "lokr",
            "network_dim": 16,
            "learning_rate": 0.0004,
        }
        toml_file = tmp_path / "train.toml"
        save_stage_toml(toml_data, toml_file)
        loaded = load_stage_toml(toml_file)
        assert loaded["network_type"] == "lokr"
        assert loaded["network_dim"] == 16


class TestPlaceholderResolution:
    def test_resolve_stage_output(self):
        stage_outputs = {
            "preprocess_s1": {
                "dataset_dir": "/runs/20260528/preprocess_s1/post_image_dataset",
            },
            "train_s1": {
                "safetensors_path": "/runs/20260528/train_s1/output/anima_lokr.safetensors",
            },
        }
        text = "${preprocess_s1.dataset_dir}/hanechan/.resized"
        result = resolve_placeholders(text, stage_outputs)
        assert result == "/runs/20260528/preprocess_s1/post_image_dataset/hanechan/.resized"

    def test_resolve_nested_placeholder(self):
        stage_outputs = {
            "train_s1": {"safetensors_path": "/path/to/model.safetensors"},
        }
        toml_data = {"network_weights": "${train_s1.safetensors_path}"}
        result = resolve_placeholders(toml_data, stage_outputs)
        assert result["network_weights"] == "/path/to/model.safetensors"

    def test_resolve_dict_recursively(self):
        stage_outputs = {"p1": {"dataset_dir": "/data"}}
        toml_data = {
            "key1": "${p1.dataset_dir}/a",
            "sections": {"key2": "${p1.dataset_dir}/b"},
        }
        result = resolve_placeholders(toml_data, stage_outputs)
        assert result["key1"] == "/data/a"
        assert result["sections"]["key2"] == "/data/b"

    def test_unresolved_raises(self):
        with pytest.raises(ValueError, match="unresolved"):
            resolve_placeholders("${missing.foo}", {})
