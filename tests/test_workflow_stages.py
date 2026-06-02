import pytest
from pathlib import Path
from workflow.stages.base import StageBase, StageResult
from workflow.stages.preprocess import PreprocessExecutor
from workflow.stages.train import TrainExecutor


class TestPreprocessExecutor:
    def test_build_resize_command(self, tmp_path):
        config = {
            "source_image_dir": "O:/LoRATraining/hanechan",
            "bucket_families": ["S1"],
        }
        stage_dir = tmp_path / "preprocess_s1"
        stage_dir.mkdir()
        infra = {"pretrained_model_name_or_path": "", "vae": "", "qwen3": ""}
        executor = PreprocessExecutor("preprocess_s1", config, stage_dir, infra)
        cmd = executor._build_resize_cmd()
        assert "resize_images.py" in cmd[0] or "resize_images.py" in str(cmd)
        assert "--bucket_families" in cmd

    def test_discover_subsets_after_run(self, tmp_path):
        stage_dir = tmp_path / "preprocess_s1"
        post_dir = stage_dir / "post_image_dataset" / "hanechan" / "1_subset_a"
        resized = post_dir / ".resized"
        lora = post_dir / ".lora"
        resized.mkdir(parents=True)
        lora.mkdir(parents=True)
        (resized / "img.png").write_bytes(b"fake")
        (lora / "img_anima_te.safetensors").write_bytes(b"fake")
        config = {"source_image_dir": "O:/LoRATraining/hanechan"}
        executor = PreprocessExecutor("preprocess_s1", config, stage_dir, {})
        subsets = executor.discover_subsets()
        assert len(subsets) == 1
        assert subsets[0].name == "1_subset_a"
        assert subsets[0].num_repeats == 1


class TestTrainExecutor:
    def test_build_train_cmd_with_stop_epoch(self, tmp_path):
        config = {
            "network_type": "lokr",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0004,
            "lr_scheduler": "cosine",
            "max_train_epochs": 10,
            "stop_epoch": 6,
            "optimizer_type": "CAME",
        }
        stage_dir = tmp_path / "train_s1"
        stage_dir.mkdir()
        infra = {
            "pretrained_model_name_or_path": "/dit",
            "vae": "/vae",
            "qwen3": "/te",
            "mixed_precision": "bf16",
        }
        executor = TrainExecutor("train_s1", config, stage_dir, infra)
        resolved_config = executor.prepare_config({})
        assert resolved_config["max_train_epochs"] == 6
        assert resolved_config["save_every_n_epochs"] == 6

    def test_build_train_cmd_with_network_weights(self, tmp_path):
        config = {
            "network_type": "lokr",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.000138,
            "max_train_epochs": 4,
            "network_weights": "/path/to/checkpoint.safetensors",
        }
        stage_dir = tmp_path / "train_s2"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s2", config, stage_dir, {})
        resolved = executor.prepare_config({})
        assert resolved["network_weights"] == "/path/to/checkpoint.safetensors"
        # LyCORIS variants must NOT auto-infer rank from weights
        assert resolved["dim_from_weights"] is False

    def test_build_train_cmd_with_network_weights_lora_infers_dim(self, tmp_path):
        """For plain LoRA, network_weights triggers dim_from_weights=True
        so rank is auto-inferred from the warm-start checkpoint."""
        config = {
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.000138,
            "max_train_epochs": 4,
            "network_weights": "/path/to/lora.safetensors",
        }
        stage_dir = tmp_path / "train_s3"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s3", config, stage_dir, {})
        resolved = executor.prepare_config({})
        assert resolved["network_weights"] == "/path/to/lora.safetensors"
        assert resolved["dim_from_weights"] is True

    def test_build_train_cmd_forwards_bucket_families_list(self, tmp_path):
        """bucket_families as list is forwarded as comma-separated --bucket_families."""
        config = {
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0001,
            "max_train_epochs": 1,
            "bucket_families": ["XS"],
        }
        stage_dir = tmp_path / "train_s4"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s4", config, stage_dir, {})
        resolved = executor.prepare_config({})
        cmd = executor._build_train_cmd(resolved, Path("/tmp/dataset.toml"))
        idx = cmd.index("--bucket_families")
        assert cmd[idx + 1] == "XS"

    def test_build_train_cmd_forwards_bucket_families_string(self, tmp_path):
        """bucket_families as string is forwarded directly."""
        config = {
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0001,
            "max_train_epochs": 1,
            "bucket_families": "XS,S1",
        }
        stage_dir = tmp_path / "train_s5"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s5", config, stage_dir, {})
        resolved = executor.prepare_config({})
        cmd = executor._build_train_cmd(resolved, Path("/tmp/dataset.toml"))
        idx = cmd.index("--bucket_families")
        assert cmd[idx + 1] == "XS,S1"

    def test_build_train_cmd_no_bucket_families_when_unset(self, tmp_path):
        """When bucket_families is not in config, --bucket_families is not in cmd."""
        config = {
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0001,
            "max_train_epochs": 1,
        }
        stage_dir = tmp_path / "train_s6"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s6", config, stage_dir, {})
        resolved = executor.prepare_config({})
        cmd = executor._build_train_cmd(resolved, Path("/tmp/dataset.toml"))
        assert "--bucket_families" not in cmd

    def test_bucket_families_inherited_from_preprocess_outputs(self, tmp_path):
        """bucket_families auto-discovered from preprocess stage_outputs."""
        config = {
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0001,
            "max_train_epochs": 1,
        }
        stage_dir = tmp_path / "train_s7"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s7", config, stage_dir, {})
        stage_outputs = {
            "preprocess_1": {
                "dataset_dir": "/some/path",
                "bucket_families": ["XS"],
            }
        }
        resolved = executor.prepare_config(stage_outputs)
        assert resolved["bucket_families"] == ["XS"]
        cmd = executor._build_train_cmd(resolved, Path("/tmp/dataset.toml"))
        idx = cmd.index("--bucket_families")
        assert cmd[idx + 1] == "XS"

    def test_bucket_families_config_overrides_stage_outputs(self, tmp_path):
        """Explicit bucket_families in train config takes priority over stage_outputs."""
        config = {
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0001,
            "max_train_epochs": 1,
            "bucket_families": ["S1"],
        }
        stage_dir = tmp_path / "train_s8"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s8", config, stage_dir, {})
        stage_outputs = {
            "preprocess_1": {
                "dataset_dir": "/some/path",
                "bucket_families": ["XS"],
            }
        }
        resolved = executor.prepare_config(stage_outputs)
        assert resolved["bucket_families"] == ["S1"]
        cmd = executor._build_train_cmd(resolved, Path("/tmp/dataset.toml"))
        idx = cmd.index("--bucket_families")
        assert cmd[idx + 1] == "S1"
