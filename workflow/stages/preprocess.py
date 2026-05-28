from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from workflow.stages.base import StageBase, StageResult
from workflow.models import SubsetInfo


class PreprocessExecutor(StageBase):
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "preprocess"

    def prepare_config(self, stage_outputs: dict) -> dict:
        merged = {**self.infrastructure, **self.config}
        return merged

    def _build_resize_cmd(self) -> list[str]:
        src = self.config["source_image_dir"]
        dst = str(self.stage_dir / "post_image_dataset")
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "resize_images.py"),
            "--src", src,
            "--dst", dst,
            "--tree",
        ]
        families = self.config.get("bucket_families", ["S1"])
        if families:
            cmd += ["--bucket_families", ",".join(families)]
        if "min_pixels" in self.config:
            cmd += ["--min_pixels", str(self.config["min_pixels"])]
        return cmd

    def _build_vae_cmd(self) -> list[str]:
        dst = str(self.stage_dir / "post_image_dataset")
        vae = self.infrastructure.get("vae", "")
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "cache_latents.py"),
            "--dir", dst,
            "--tree",
        ]
        if vae:
            cmd += ["--vae", vae]
        cmd += ["--cache_dir", dst]
        return cmd

    def _build_te_cmd(self) -> list[str]:
        dst = str(self.stage_dir / "post_image_dataset")
        qwen3 = self.infrastructure.get("qwen3", "")
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "cache_text_embeddings.py"),
            "--dir", dst,
            "--tree",
        ]
        if qwen3:
            cmd += ["--qwen3", qwen3]
        cmd += ["--cache_dir", dst]
        return cmd

    def discover_subsets(self) -> list[SubsetInfo]:
        post_dir = self.stage_dir / "post_image_dataset"
        if not post_dir.exists():
            return []
        subsets: list[SubsetInfo] = []
        for dataset_dir in sorted(post_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for subset_dir in sorted(dataset_dir.iterdir()):
                if not subset_dir.is_dir():
                    continue
                resized = subset_dir / ".resized"
                lora = subset_dir / ".lora"
                if resized.exists() or lora.exists():
                    subsets.append(
                        SubsetInfo(
                            name=subset_dir.name,
                            image_dir=str(resized),
                            cache_dir=str(lora),
                            num_repeats=1,
                        )
                    )
        return subsets

    def execute(
        self,
        on_stdout: Callable | None = None,
        on_progress: Callable | None = None,
    ) -> StageResult:
        try:
            for step_name, cmd_builder in [
                ("resize", self._build_resize_cmd),
                ("vae", self._build_vae_cmd),
                ("te", self._build_te_cmd),
            ]:
                cmd = cmd_builder()
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
                        error=f"{step_name} failed with exit code {proc.returncode}",
                    )

            subsets = self.discover_subsets()
            dataset_dir = str(self.stage_dir / "post_image_dataset")
            return StageResult(
                success=True,
                outputs={"dataset_dir": dataset_dir},
                subsets=subsets,
            )
        except Exception as e:
            return StageResult(success=False, error=str(e))
