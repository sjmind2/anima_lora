import pytest
from pathlib import Path
from workflow.scheduler import WorkflowScheduler
from workflow.models import WorkflowDefinition, WorkflowStage
from workflow.logger import EventQueue


class TestWorkflowScheduler:
    def test_create_run_directory(self, tmp_path):
        wf_dir = tmp_path / "test-wf"
        wf_dir.mkdir()
        wf = WorkflowDefinition(name="test-wf", stages=[
            WorkflowStage(id="p1", type="preprocess", config_file="p1.toml", depends_on=[]),
        ])
        eq = EventQueue()
        scheduler = WorkflowScheduler(wf_dir, wf, eq)
        run_dir = scheduler._create_run_dir()
        assert run_dir.exists()
        assert run_dir.parent.name == "runs"

    def test_resolve_stage_config_writes_resolved(self, tmp_path):
        wf_dir = tmp_path / "test-wf"
        configs_dir = wf_dir / "configs"
        configs_dir.mkdir(parents=True)
        (configs_dir / "p1.toml").write_text('source_image_dir = "O:/data"\nbucket_families = ["S1"]\n', encoding="utf-8")
        wf = WorkflowDefinition(name="test-wf", stages=[
            WorkflowStage(id="p1", type="preprocess", config_file="p1.toml", depends_on=[]),
        ])
        eq = EventQueue()
        scheduler = WorkflowScheduler(wf_dir, wf, eq)
        run_dir = scheduler._create_run_dir()
        resolved = scheduler._resolve_and_write_config("p1", run_dir, {})
        assert resolved["source_image_dir"] == "O:/data"
        resolved_file = run_dir / "p1" / "config.toml"
        assert resolved_file.exists()

    def test_resolve_with_placeholders(self, tmp_path):
        wf_dir = tmp_path / "test-wf"
        configs_dir = wf_dir / "configs"
        configs_dir.mkdir(parents=True)
        (configs_dir / "t1.toml").write_text(
            'network_weights = "${p1.dataset_dir}/model.safetensors"\n',
            encoding="utf-8",
        )
        wf = WorkflowDefinition(name="test-wf", stages=[
            WorkflowStage(id="p1", type="preprocess", config_file="p1.toml", depends_on=[]),
            WorkflowStage(id="t1", type="train", config_file="t1.toml", depends_on=["p1"]),
        ])
        eq = EventQueue()
        scheduler = WorkflowScheduler(wf_dir, wf, eq)
        run_dir = scheduler._create_run_dir()
        stage_outputs = {"p1": {"dataset_dir": "/data/output"}}
        resolved = scheduler._resolve_and_write_config("t1", run_dir, stage_outputs)
        assert resolved["network_weights"] == "/data/output/model.safetensors"
