from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

from workflow.stages.base import StageBase, StageResult
from workflow.models import SubsetInfo


class TrainExecutor(StageBase):
    _TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "train.py"

    def prepare_config(self, stage_outputs: dict) -> dict:
        resolved = {**self.infrastructure, **self.config}

        stop_epoch = resolved.get("stop_epoch")
        if stop_epoch is not None:
            resolved["max_train_epochs"] = int(stop_epoch)
            resolved["save_every_n_epochs"] = int(stop_epoch)

        if resolved.get("network_weights"):
            resolved["dim_from_weights"] = True

        return resolved

    def _build_train_cmd(self, resolved_config: dict) -> list[str]:
        cmd = [sys.executable, str(self._TRAIN_SCRIPT)]

        method = resolved_config.get("network_type", "lora")
        cmd += ["--method", method]

        skip_keys = {
            "stop_epoch",
            "datasets",
            "subsets",
            "source_image_dir",
            "bucket_families",
            "drop_lowres_images",
            "min_pixels",
        }
        for key, value in resolved_config.items():
            if key in skip_keys:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        return cmd

    def execute(
        self,
        on_stdout: Callable | None = None,
        on_progress: Callable | None = None,
    ) -> StageResult:
        try:
            resolved = self.prepare_config({})
            cmd = self._build_train_cmd(resolved)

            output_dir = self.stage_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--output_dir", str(output_dir)]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            for line in proc.stdout:
                if on_stdout:
                    on_stdout(self.stage_id, line.rstrip())
            proc.wait()

            if proc.returncode != 0:
                return StageResult(
                    success=False,
                    error=f"train.py failed with exit code {proc.returncode}",
                )

            safetensors = list(output_dir.glob("*.safetensors"))
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
