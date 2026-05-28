from __future__ import annotations

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

    @abstractmethod
    def prepare_config(self, stage_outputs: dict) -> dict: ...

    @abstractmethod
    def execute(self, on_stdout, on_progress) -> StageResult: ...
