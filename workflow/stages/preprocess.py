from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import hashlib
import json
import shutil
from typing import Any, Callable

from workflow.stages.base import StageBase, StageResult
from workflow.models import SubsetInfo


def _parse_num_repeats(name: str) -> int:
    """Extract repeat count from a '{repeat}_{tag}' directory name."""
    tokens = name.split("_")
    try:
        n = int(tokens[0])
        return n if n >= 1 else 1
    except (ValueError, IndexError):
        return 1


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

    @staticmethod
    def _compute_config_hash(config: dict) -> str:
        """Deterministic hash of config contents, independent of key order."""
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _invalidate_shared_cache(self, post_dir: Path, hash_file: Path) -> None:
        """Delete both cache data and hash snapshot to prevent stale matches."""
        shutil.rmtree(str(post_dir), ignore_errors=True)
        hash_file.unlink(missing_ok=True)
        config_file = hash_file.parent / "config.toml"
        config_file.unlink(missing_ok=True)

    def _should_skip_shared_cache(self, resolved_config: dict, on_stdout) -> bool:
        """Check shared cache: return True if preprocessing can be skipped."""
        post_dir = self.stage_dir / "post_image_dataset"
        hash_file = self.stage_dir / ".config_hash"

        if not post_dir.exists():
            # No cache data; also clean up stale hash if present
            if hash_file.exists():
                hash_file.unlink(missing_ok=True)
                config_file = hash_file.parent / "config.toml"
                config_file.unlink(missing_ok=True)
            return False

        current_hash = self._compute_config_hash(resolved_config)

        if hash_file.exists():
            saved_hash = hash_file.read_text(encoding="utf-8").strip()
            if saved_hash == current_hash:
                # Verify cache is complete: all subsets must have both .resized and .lora dirs
                subsets = self.discover_subsets()
                if subsets and all(
                    Path(s.image_dir).exists() and Path(s.cache_dir).exists()
                    for s in subsets
                ):
                    if on_stdout:
                        from workflow.i18n import t

                        on_stdout(
                            self.stage_id,
                            t("backend.stages.sharedCacheHit", stage_id=self.stage_id),
                        )
                    return True
                else:
                    if on_stdout:
                        from workflow.i18n import t

                        on_stdout(
                            self.stage_id,
                            t(
                                "backend.stages.sharedCacheMismatch",
                                stage_id=self.stage_id,
                            ),
                        )
                    self._invalidate_shared_cache(post_dir, hash_file)
                    return False
            else:
                if on_stdout:
                    from workflow.i18n import t

                    on_stdout(
                        self.stage_id,
                        t("backend.stages.sharedCacheMismatch", stage_id=self.stage_id),
                    )
                self._invalidate_shared_cache(post_dir, hash_file)
        else:
            # Cache data exists but no hash snapshot — incomplete run, invalidate
            if on_stdout:
                from workflow.i18n import t

                on_stdout(
                    self.stage_id,
                    t("backend.stages.sharedCacheMismatch", stage_id=self.stage_id),
                )
            self._invalidate_shared_cache(post_dir, hash_file)

        return False

    def _save_config_snapshot(self, resolved_config: dict) -> None:
        """Save config.toml and .config_hash for shared cache comparison."""
        from workflow.config import save_stage_toml

        save_stage_toml(resolved_config, self.stage_dir / "config.toml")
        hash_val = self._compute_config_hash(resolved_config)
        (self.stage_dir / ".config_hash").write_text(hash_val, encoding="utf-8")

    def prepare_config(self, stage_outputs: dict) -> dict:
        merged = {**self.infrastructure, **self.config}
        return merged

    def _build_resize_cmd(self) -> list[str]:
        src = self.config["source_image_dir"]
        dst = str(self.stage_dir / "post_image_dataset")
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "resize_images.py"),
            "--src",
            src,
            "--dst",
            dst,
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
            "--dir",
            dst,
            "--tree",
            "--vae",
            vae,
            "--cache_dir",
            dst,
        ]
        return cmd

    def _build_te_cmd(self) -> list[str]:
        dst = str(self.stage_dir / "post_image_dataset")
        qwen3 = _resolve_default_model("qwen3", self.infrastructure)
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "cache_text_embeddings.py"),
            "--dir",
            dst,
            "--tree",
            "--qwen3",
            qwen3,
            "--cache_dir",
            dst,
            "--min_pixels",
            "0",
        ]
        return cmd

    def _build_pe_cmd(self) -> list[str]:
        dst = str(self.stage_dir / "post_image_dataset")
        encoder = self.config.get("pe_encoder", "pe_spatial")
        cmd = [
            sys.executable,
            str(self._SCRIPTS_DIR / "cache_pe_encoder.py"),
            "--dir",
            dst,
            "--tree",
            "--encoder",
            encoder,
            "--cache_dir",
            dst,
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
                        num_repeats=_parse_num_repeats(dataset_dir.name),
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
                            num_repeats=_parse_num_repeats(subset_dir.name),
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
            use_shared = self.config.get("shared_cache", True)
            resolved_config = self.prepare_config(stage_outputs or {})

            # Shared cache skip check
            if use_shared and self._should_skip_shared_cache(
                resolved_config, on_stdout
            ):
                subsets = self.discover_subsets()
                dataset_dir = str(self.stage_dir / "post_image_dataset")
                outputs: dict[str, Any] = {"dataset_dir": dataset_dir}
                families = self.config.get("bucket_families")
                if families:
                    outputs["bucket_families"] = families
                return StageResult(
                    success=True,
                    outputs=outputs,
                    subsets=subsets,
                )

            # Run preprocessing pipeline
            pipeline_steps = [
                ("resize", self._build_resize_cmd),
                ("vae", self._build_vae_cmd),
                ("te", self._build_te_cmd),
            ]
            # PE-Spatial cache: only when pe_encoder is set (train stage
            # enabled REPA and the workflow config propagated the encoder name).
            if self.config.get("pe_encoder"):
                pipeline_steps.append(("pe", self._build_pe_cmd))

            for step_name, cmd_builder in pipeline_steps:
                cmd = cmd_builder()
                if on_stdout:
                    on_stdout(self.stage_id, f"[COMMAND:{step_name}] " + " ".join(cmd))
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
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

            # Save config snapshot for shared cache
            if use_shared:
                self._save_config_snapshot(resolved_config)

            subsets = self.discover_subsets()
            dataset_dir = str(self.stage_dir / "post_image_dataset")
            outputs = {"dataset_dir": dataset_dir}
            families = self.config.get("bucket_families")
            if families:
                outputs["bucket_families"] = families
            return StageResult(
                success=True,
                outputs=outputs,
                subsets=subsets,
            )
        except Exception as e:
            return StageResult(success=False, error=str(e))
