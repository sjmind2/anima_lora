import pytest
from workflow.logger import EventQueue, WorkflowLogger


class TestEventQueue:
    def test_put_and_get(self):
        q = EventQueue()
        q.put({"ev": "workflow_start", "total_stages": 4})
        events = q.drain()
        assert len(events) == 1
        assert events[0]["ev"] == "workflow_start"

    def test_drain_empty(self):
        q = EventQueue()
        assert q.drain() == []


class TestWorkflowLogger:
    def test_stage_progress(self, tmp_path):
        log_file = tmp_path / "run.log"
        eq = EventQueue()
        logger = WorkflowLogger(log_file, eq)
        logger.stage_start("preprocess_s1", "preprocess")
        logger.stage_progress("preprocess_s1", pct=50, cur=50, total=100, desc="Resizing")
        logger.stage_end("preprocess_s1", "ok")
        events = eq.drain()
        assert len(events) == 3
        assert events[0]["ev"] == "stage_start"
        assert events[1]["ev"] == "stage_progress"
        assert events[1]["pct"] == 50
        assert events[2]["ev"] == "stage_end"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "preprocess_s1" in content
