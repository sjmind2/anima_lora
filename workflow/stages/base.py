from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from workflow.models import SubsetInfo


class StageResult:
    def __init__(
        self,
        success: bool,
        outputs: dict[str, str] | None = None,
        subsets: list[SubsetInfo] | None = None,
        error: str = "",
    ):
        self.success = success
        self.outputs = outputs or {}
        self.subsets = subsets or []
        self.error = error


class StageBase(ABC):
    def __init__(
        self,
        stage_id: str,
        config: dict,
        stage_dir: Path,
        infrastructure: dict,
    ):
        self.stage_id = stage_id
        self.config = config
        self.stage_dir = stage_dir
        self.infrastructure = infrastructure
        self._stop_flag = threading.Event()
        self._current_proc = None

    @abstractmethod
    def prepare_config(self, stage_outputs: dict) -> dict: ...

    @abstractmethod
    def execute(self, on_stdout, on_progress, stage_outputs=None) -> StageResult: ...

    def terminate(self) -> None:
        self._stop_flag.set()
        proc = self._current_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    @property
    def should_stop(self) -> bool:
        return self._stop_flag.is_set()
