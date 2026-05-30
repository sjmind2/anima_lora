from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

import tomli_w

from workflow.stages.base import StageBase, StageResult
from workflow.models import SubsetInfo

_NETWORK_MODULE_KWARGS = {
    "network_type",
    "conv_dim",
    "conv_alpha",
    "lokr_factor",
    "decompose_both",
    "use_tucker",
    "use_scalar",
    "block_lr",
    "down_lr_weight",
    "mid_lr_weight",
    "up_lr_weight",
    "dora_wd",
    "cp_alpha",
    "cp_scale",
}

_DATASET_KEYS = {
    "datasets",
    "subsets",
    "source_image_dir",
    "bucket_families",
    "drop_lowres_images",
    "min_pixels",
    "dataset_subsets",
    "general",
}

_METADATA_KEYS = {
    "stop_epoch",
}

_NARGS_STAR_KEYS = {"optimizer_args"}


def _resolve_default_model(key: str, infra: dict) -> str:
    val = infra.get(key, "")
    if val:
        return val
    from library.env import resolve_under_home
    defaults = {
        "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
        "qwen3": "models/text_encoders/qwen_3_06b_base.safetensors",
        "vae": "models/vae/qwen_image_vae.safetensors",
    }
    if key in defaults:
        resolved = resolve_under_home(defaults[key])
        if resolved.exists():
            return str(resolved)
    return val


class TrainExecutor(StageBase):
    _TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "train.py"

    _AUTO_DEFAULTS = {
        "network_module": "networks.lora_anima",
        "network_train_unet_only": True,
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "attn_mode": "flash",
        "use_vae_cache": True,
        "use_text_cache": True,
        "skip_cache_check": True,
        "vae_chunk_size": 64,
        "vae_disable_cache": True,
        "masked_loss": True,
        "log_every_n_steps": 2,
        "dataloader_pin_memory": True,
        "persistent_data_loader_workers": True,
        "use_cmmd": False,
        "save_model_as": "safetensors",
    }

    def prepare_config(self, stage_outputs: dict) -> dict:
        resolved = {**self.infrastructure, **self.config}
        for key, default_val in self._AUTO_DEFAULTS.items():
            if key not in resolved:
                resolved[key] = default_val

        if not resolved.get("network_weights"):
            resolved["dim_from_weights"] = False
            for sid, outputs in (stage_outputs or {}).items():
                if outputs.get("safetensors_path"):
                    resolved["network_weights"] = outputs["safetensors_path"]
                    nt = resolved.get("network_type", "lora").lower()
                    if nt in ("lokr", "loha", "locon"):
                        resolved["dim_from_weights"] = False
                    else:
                        resolved["dim_from_weights"] = True
                    break

        if not resolved.get("datasets"):
            datasets = []
            for sid, outputs in (stage_outputs or {}).items():
                if outputs.get("dataset_dir"):
                    datasets.append(sid)
            if datasets:
                resolved["datasets"] = datasets

        if "torch_compile" not in resolved:
            resolved["torch_compile"] = True

        stop_epoch = resolved.get("stop_epoch")
        if stop_epoch is not None:
            resolved["max_train_epochs"] = int(stop_epoch)
            resolved["save_every_n_epochs"] = int(stop_epoch)

        return resolved

    def _write_dataset_toml(self, resolved_config: dict, stage_outputs: dict) -> Path:
        dataset_config_path = self.stage_dir / "dataset_config.toml"
        dataset_config_path.parent.mkdir(parents=True, exist_ok=True)

        toml_data: dict = {}
        general = dict(resolved_config.get("general", {}))
        if general:
            toml_data["general"] = general

        datasets = resolved_config.get("datasets")
        if datasets:
            resolved_datasets = []
            for ds in datasets:
                if isinstance(ds, str):
                    ref_outputs = stage_outputs.get(ds)
                    if not ref_outputs:
                        raise ValueError(f"Dataset reference '{ds}' not found in stage outputs")
                    ref_subsets = ref_outputs.get("subsets", [])
                    dataset_entry = {}
                    if ref_subsets:
                        subset_entries = []
                        for s in ref_subsets:
                            entry = {"image_dir": s["image_dir"]}
                            if s.get("cache_dir"):
                                entry["cache_dir"] = s["cache_dir"]
                            entry["num_repeats"] = 1
                            subset_entries.append(entry)
                        dataset_entry["subsets"] = subset_entries
                    else:
                        dataset_dir = ref_outputs.get("dataset_dir", "")
                        dataset_entry["subsets"] = [{"image_dir": dataset_dir, "num_repeats": 1}]
                    resolved_datasets.append(dataset_entry)
                else:
                    resolved_datasets.append(ds)
            toml_data["datasets"] = resolved_datasets

        with open(dataset_config_path, "wb") as f:
            tomli_w.dump(toml_data, f)

        return dataset_config_path

    def _build_train_cmd(self, resolved_config: dict, dataset_toml_path: Path) -> list[str]:
        cmd = [sys.executable, str(self._TRAIN_SCRIPT)]

        cmd += ["--dataset_config", str(dataset_toml_path)]

        network_kwargs: dict[str, str] = {}
        skip_keys = _DATASET_KEYS | _METADATA_KEYS | _NETWORK_MODULE_KWARGS

        for key, value in resolved_config.items():
            if key in skip_keys:
                continue
            if value is None:
                continue
            if key in _NARGS_STAR_KEYS:
                cmd.append(f"--{key}")
                if isinstance(value, list):
                    for item in value:
                        cmd.append(str(item))
                elif isinstance(value, str):
                    import re
                    items = [m.group(0).strip().rstrip(",") for m in re.finditer(r'[^=\s]+=\S+', value)]
                    for item in items:
                        if item:
                            cmd.append(item)
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                for item in value:
                    cmd.append(f"--{key}")
                    cmd.append(str(item))
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        for key in _NETWORK_MODULE_KWARGS:
            if key in resolved_config:
                val = resolved_config[key]
                network_kwargs[key] = str(val)

        if network_kwargs:
            cmd.append("--network_args")
            for k, v in network_kwargs.items():
                cmd.append(f"{k}={v}")

        return cmd

    def execute(
        self,
        on_stdout: Callable | None = None,
        on_progress: Callable | None = None,
        stage_outputs: dict | None = None,
    ) -> StageResult:
        try:
            resolved = self.prepare_config(stage_outputs or {})

            dataset_toml_path = self._write_dataset_toml(resolved, stage_outputs or {})
            cmd = self._build_train_cmd(resolved, dataset_toml_path)

            output_dir = self.stage_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--output_dir", str(output_dir)]

            if "pretrained_model_name_or_path" not in resolved:
                model_path = _resolve_default_model("pretrained_model_name_or_path", self.infrastructure)
                if model_path:
                    cmd += ["--pretrained_model_name_or_path", model_path]

            for model_key in ("qwen3", "vae"):
                if model_key not in resolved:
                    model_path = _resolve_default_model(model_key, self.infrastructure)
                    if model_path:
                        cmd += [f"--{model_key}", model_path]

            cmd_path = self.stage_dir / "command.txt"
            with open(cmd_path, "w", encoding="utf-8") as f:
                f.write(" ".join(cmd))

            cmd_str = " ".join(cmd)
            if on_stdout:
                on_stdout(self.stage_id, f"[COMMAND] {cmd_str}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            self._current_proc = proc
            for line in proc.stdout:
                if self.should_stop:
                    proc.terminate()
                    break
                if on_stdout:
                    on_stdout(self.stage_id, line.rstrip())
            proc.wait()
            self._current_proc = None

            if self.should_stop:
                return StageResult(success=False, error="stopped")

            if proc.returncode != 0:
                return StageResult(
                    success=False,
                    error=f"train.py failed with exit code {proc.returncode}",
                )

            safetensors = sorted(output_dir.glob("*.safetensors"))
            safetensors_path = str(safetensors[-1]) if safetensors else ""

            return StageResult(
                success=True,
                outputs={
                    "safetensors_path": safetensors_path,
                    "checkpoint_dir": str(output_dir),
                },
            )
        except Exception as e:
            return StageResult(success=False, error=str(e))
