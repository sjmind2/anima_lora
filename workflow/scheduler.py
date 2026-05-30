from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from workflow.config import load_stage_toml, save_stage_toml, resolve_placeholders
from workflow.logger import EventQueue, WorkflowLogger
from workflow.models import WorkflowDefinition, WorkflowStage, StageOutput
from workflow.stages.preprocess import PreprocessExecutor
from workflow.stages.train import TrainExecutor


class _StdoutBuffer:
    def __init__(self, logger: WorkflowLogger, flush_interval: float = 0.3) -> None:
        self._logger = logger
        self._buf: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._timer: threading.Timer | None = None
        self._stopped = True

    def add(self, stage_id: str, line: str) -> None:
        with self._lock:
            self._buf.setdefault(stage_id, []).append(line)

    def start(self) -> None:
        self._stopped = False
        self._schedule_next()

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            snapshot = {k: v[:] for k, v in self._buf.items()}
            self._buf.clear()
        for sid, lines in snapshot.items():
            self._logger.stage_stdout_batch(sid, lines)
        if not self._stopped:
            self._schedule_next()

    def stop(self) -> None:
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self._flush()


class WorkflowScheduler:
    def __init__(self, wf_dir: Path, wf: WorkflowDefinition, event_queue: EventQueue) -> None:
        self.wf_dir = wf_dir
        self.wf = wf
        self.event_queue = event_queue
        self._stop_flag = threading.Event()
        self._current_executor = None

    def _create_run_dir(self) -> Path:
        runs_dir = self.wf_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _resolve_and_write_config(self, stage_id: str, run_dir: Path,
                                   stage_outputs: dict[str, dict[str, str]]) -> dict:
        stage = next(s for s in self.wf.stages if s.id == stage_id)
        config_path = self.wf_dir / "configs" / stage.config_file
        raw_config = load_stage_toml(config_path)
        resolved = resolve_placeholders(raw_config, stage_outputs)

        stage_run_dir = run_dir / stage_id
        stage_run_dir.mkdir(parents=True, exist_ok=True)
        save_stage_toml(resolved, stage_run_dir / "config.toml")
        return resolved

    def _make_executor(self, stage: WorkflowStage, config: dict, run_dir: Path):
        stage_dir = run_dir / stage.id
        stage_dir.mkdir(parents=True, exist_ok=True)
        infra = self.wf.infrastructure or {}
        if stage.type == "preprocess":
            return PreprocessExecutor(stage.id, config, stage_dir, infra)
        elif stage.type == "train":
            return TrainExecutor(stage.id, config, stage_dir, infra)
        raise ValueError(f"Unknown stage type: {stage.type}")

    def _write_status(
        self,
        run_dir: Path,
        stages: list[WorkflowStage],
        stage_statuses: dict[str, dict],
        current_stage_id: str | None,
        overall_status: str,
        started_at: str,
    ) -> None:
        payload: dict[str, Any] = {
            "workflow": self.wf.name,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "status": overall_status,
            "current_stage": current_stage_id,
            "stages": [],
        }
        for s in stages:
            entry: dict[str, Any] = {
                "id": s.id,
                "type": s.type,
                "status": "pending",
            }
            info = stage_statuses.get(s.id)
            if info:
                entry["status"] = info.get("status", "pending")
                if info.get("error"):
                    entry["error"] = info["error"]
                if info.get("outputs"):
                    entry["outputs"] = info["outputs"]
            payload["stages"].append(entry)
        tmp = run_dir / "status.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(run_dir / "status.json")

    def run(self, log_file: Path | None = None) -> bool:
        log_file = log_file or (self._create_run_dir() / "run.log")
        logger = WorkflowLogger(log_file, self.event_queue)
        ordered = self.wf.topological_order()
        stage_outputs: dict[str, dict[str, str]] = {}
        stage_statuses: dict[str, dict] = {}
        all_success = True
        started_at = datetime.now().isoformat()
        run_dir = log_file.parent

        buffer = _StdoutBuffer(logger, flush_interval=0.3)
        buffer.start()

        logger.workflow_start(len(ordered))
        self._write_status(run_dir, ordered, stage_statuses, None, "running", started_at)

        for stage in ordered:
            if self._stop_flag.is_set():
                stage_statuses[stage.id] = {"status": "stopped"}
                logger.stage_end(stage.id, "stopped")
                self._write_status(run_dir, ordered, stage_statuses, stage.id, "stopped", started_at)
                all_success = False
                break

            try:
                resolved = self._resolve_and_write_config(stage.id, run_dir, stage_outputs)
            except Exception as e:
                stage_statuses[stage.id] = {"status": "config_error", "error": str(e)}
                logger.stage_end(stage.id, f"config_error: {e}")
                self._write_status(run_dir, ordered, stage_statuses, stage.id, "error", started_at)
                all_success = False
                break

            executor = self._make_executor(stage, resolved, run_dir)
            self._current_executor = executor
            logger.stage_start(stage.id, stage.type)
            stage_statuses[stage.id] = {"status": "running"}
            self._write_status(run_dir, ordered, stage_statuses, stage.id, "running", started_at)

            config_path = run_dir / stage.id / "config.toml"
            if config_path.exists():
                logger._log(stage.id, "INFO", f"config file: {config_path}")
                try:
                    content = config_path.read_text(encoding="utf-8").strip()
                    for line in content.splitlines():
                        logger._log(stage.id, "INFO", f"  {line}")
                except Exception:
                    pass

            def on_stdout(sid: str, line: str) -> None:
                buffer.add(sid, line)

            result = executor.execute(on_stdout=on_stdout, stage_outputs=stage_outputs)
            self._current_executor = None

            if result.success:
                outputs = dict(result.outputs)
                if result.subsets:
                    outputs["subsets"] = [
                        {"name": s.name, "image_dir": s.image_dir, "cache_dir": s.cache_dir}
                        for s in result.subsets
                    ]
                stage_outputs[stage.id] = outputs
                stage_statuses[stage.id] = {"status": "ok", "outputs": outputs}
                logger.stage_end(stage.id, "ok")
                self._write_status(run_dir, ordered, stage_statuses, None, "running", started_at)
            else:
                all_success = False
                stage_statuses[stage.id] = {"status": "error", "error": result.error}
                logger.stage_end(stage.id, f"error: {result.error}")
                self._write_status(run_dir, ordered, stage_statuses, stage.id, "error", started_at)
                break

        buffer.stop()
        status = "ok" if all_success else "error"
        logger.workflow_end(status)
        self._write_status(run_dir, ordered, stage_statuses, None, status, started_at)
        runs_dir = run_dir.parent
        latest = runs_dir / "latest"
        try:
            if latest.exists() or latest.is_symlink():
                if latest.is_dir() and not latest.is_symlink():
                    import shutil
                    shutil.rmtree(str(latest))
                else:
                    latest.unlink()
            if sys.platform == "win32":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(latest), str(run_dir)],
                    check=True, capture_output=True,
                )
            else:
                os.symlink(str(run_dir), str(latest))
        except Exception:
            pass
        return all_success

    def stop(self) -> None:
        self._stop_flag.set()
        executor = self._current_executor
        if executor is not None:
            executor.terminate()
