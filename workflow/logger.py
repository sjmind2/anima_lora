from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class EventQueue:
    def __init__(self) -> None:
        self._queue: list[dict] = []
        self._lock = threading.Lock()

    def put(self, event: dict) -> None:
        with self._lock:
            self._queue.append(event)

    def drain(self) -> list[dict]:
        with self._lock:
            events = self._queue[:]
            self._queue.clear()
            return events


class WorkflowLogger:
    def __init__(self, log_file: Path, event_queue: EventQueue) -> None:
        self._log_file = log_file
        self._event_queue = event_queue
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, stage_id: str, level: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] [{stage_id}] {message}\n"
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(line)

    def _emit(self, event: dict) -> None:
        event["ts"] = time.time()
        self._event_queue.put(event)

    def workflow_start(self, total_stages: int) -> None:
        self._emit({"ev": "workflow_start", "total_stages": total_stages})

    def workflow_end(self, status: str) -> None:
        self._emit({"ev": "workflow_end", "status": status})

    def stage_start(self, stage_id: str, stage_type: str) -> None:
        self._log(stage_id, "INFO", f"stage start ({stage_type})")
        self._emit({"ev": "stage_start", "stage_id": stage_id, "stage_type": stage_type})

    def stage_progress(self, stage_id: str, **kwargs: Any) -> None:
        self._emit({"ev": "stage_progress", "stage_id": stage_id, **kwargs})

    def stage_ckpt(self, stage_id: str, path: str, epoch: int) -> None:
        self._log(stage_id, "INFO", f"checkpoint saved: {path} (epoch {epoch})")
        self._emit({"ev": "stage_ckpt", "stage_id": stage_id, "path": path, "epoch": epoch})

    def stage_end(self, stage_id: str, status: str) -> None:
        self._log(stage_id, "INFO" if status == "ok" else "ERROR", f"stage end: {status}")
        self._emit({"ev": "stage_end", "stage_id": stage_id, "status": status})

    def info(self, stage_id: str, message: str) -> None:
        self._log(stage_id, "INFO", message)
