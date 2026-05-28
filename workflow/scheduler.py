from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from workflow.config import load_stage_toml, save_stage_toml, resolve_placeholders
from workflow.logger import EventQueue, WorkflowLogger
from workflow.models import WorkflowDefinition, WorkflowStage, StageOutput
from workflow.stages.preprocess import PreprocessExecutor
from workflow.stages.train import TrainExecutor


class WorkflowScheduler:
    def __init__(self, wf_dir: Path, wf: WorkflowDefinition, event_queue: EventQueue) -> None:
        self.wf_dir = wf_dir
        self.wf = wf
        self.event_queue = event_queue
        self._stop_flag = threading.Event()
        self._current_proc = None

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

    def run(self, log_file: Path | None = None) -> bool:
        log_file = log_file or (self._create_run_dir() / "run.log")
        logger = WorkflowLogger(log_file, self.event_queue)
        ordered = self.wf.topological_order()
        stage_outputs: dict[str, dict[str, str]] = {}
        all_success = True

        logger.workflow_start(len(ordered))

        for stage in ordered:
            if self._stop_flag.is_set():
                logger.stage_end(stage.id, "stopped")
                all_success = False
                break

            try:
                resolved = self._resolve_and_write_config(stage.id, log_file.parent, stage_outputs)
            except Exception as e:
                logger.stage_end(stage.id, f"config_error: {e}")
                all_success = False
                break

            executor = self._make_executor(stage, resolved, log_file.parent)
            logger.stage_start(stage.id, stage.type)

            def on_stdout(sid: str, line: str) -> None:
                logger.info(sid, line)

            result = executor.execute(on_stdout=on_stdout)

            if result.success:
                stage_outputs[stage.id] = result.outputs
                logger.stage_end(stage.id, "ok")
            else:
                all_success = False
                logger.stage_end(stage.id, f"error: {result.error}")
                break

        status = "ok" if all_success else "error"
        logger.workflow_end(status)
        return all_success

    def stop(self) -> None:
        self._stop_flag.set()
