from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from workflow.stages.base import StageBase, StageResult
from workflow.models import SubsetInfo


def _resolve_default_model(key: str, infra: dict) -> str:
    val = infra.get(key, "")
    if val:
        return val
    from library.env import resolve_under_home
    defaults = {
        "vae": "models/vae/qwen_image_vae.safetensors",
        "qwen3": "models/text_encoders/qwen_3_06b_base.safetensors",
        "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
    }
    if key in defaults:
        resolved = resolve_under_home(defaults[key])
        if resolved.exists():
            return str(resolved)
    return val


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
            cmd += ["--bucket_families", ",".join(str(f) for f in families)]
        min_pixels = self.config.get("min_pixels", 500000)
        cmd += ["--min_pixels", str(min_pixels)]
        return cmd

    def _build_vae_cmd(self) -> list[str]:
        dst = str(self.stage_dir / "post_image_dataset")
        vae = _resolve_default_model("vae", self.infrastructure)
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "cache_latents.py"),
            "--dir", dst,
            "--tree",
            "--vae", vae,
            "--cache_dir", dst,
        ]
        return cmd

    def _build_te_cmd(self) -> list[str]:
        dst = str(self.stage_dir / "post_image_dataset")
        qwen3 = _resolve_default_model("qwen3", self.infrastructure)
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "cache_text_embeddings.py"),
            "--dir", dst,
            "--tree",
            "--qwen3", qwen3,
            "--cache_dir", dst,
            "--min_pixels", "0",
        ]
        return cmd

    def discover_subsets(self) -> list[SubsetInfo]:
        post_dir = self.stage_dir / "post_image_dataset"
        if not post_dir.exists():
            return []
        subsets: list[SubsetInfo] = []
        for dataset_dir in sorted(post_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            resized = dataset_dir / ".resized"
            lora = dataset_dir / ".lora"
            if resized.exists() or lora.exists():
                subsets.append(
                    SubsetInfo(
                        name=dataset_dir.name,
                        image_dir=str(resized),
                        cache_dir=str(lora),
                        num_repeats=1,
                    )
                )
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
        stage_outputs: dict | None = None,
    ) -> StageResult:
        try:
            for step_name, cmd_builder in [
                ("resize", self._build_resize_cmd),
                ("vae", self._build_vae_cmd),
                ("te", self._build_te_cmd),
            ]:
                cmd = cmd_builder()
                if on_stdout:
                    on_stdout(self.stage_id, f"[COMMAND:{step_name}] " + " ".join(cmd))
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
