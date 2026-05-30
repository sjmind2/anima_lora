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
        assert resolved["dim_from_weights"] is True
