import pytest
from workflow.models import (
    WorkflowStage, WorkflowDefinition, StageOutput,
    InfrastructureConfig,
)


class TestWorkflowStage:
    def test_preprocess_stage_creation(self):
        stage = WorkflowStage(
            id="preprocess_s1",
            type="preprocess",
            config_file="preprocess_s1.toml",
            depends_on=[],
        )
        assert stage.id == "preprocess_s1"
        assert stage.type == "preprocess"
        assert stage.depends_on == []

    def test_train_stage_with_depends(self):
        stage = WorkflowStage(
            id="train_s2",
            type="train",
            config_file="train_s2.toml",
            depends_on=["train_s1", "preprocess_s2"],
        )
        assert "train_s1" in stage.depends_on

    def test_invalid_stage_type(self):
        with pytest.raises(ValueError):
            WorkflowStage(
                id="bad",
                type="invalid_type",
                config_file="bad.toml",
                depends_on=[],
            )


class TestWorkflowDefinition:
    def test_create_minimal_workflow(self):
        wf = WorkflowDefinition(
            name="test-wf",
            stages=[
                WorkflowStage(id="preprocess_s1", type="preprocess",
                              config_file="p1.toml", depends_on=[]),
            ],
        )
        assert wf.name == "test-wf"
        assert len(wf.stages) == 1

    def test_topological_sort(self):
        wf = WorkflowDefinition(
            name="test-wf",
            stages=[
                WorkflowStage(id="train_s2", type="train",
                              config_file="t2.toml", depends_on=["train_s1", "preprocess_s2"]),
                WorkflowStage(id="preprocess_s1", type="preprocess",
                              config_file="p1.toml", depends_on=[]),
                WorkflowStage(id="train_s1", type="train",
                              config_file="t1.toml", depends_on=["preprocess_s1"]),
                WorkflowStage(id="preprocess_s2", type="preprocess",
                              config_file="p2.toml", depends_on=[]),
            ],
        )
        order = wf.topological_order()
        idx = {s.id: i for i, s in enumerate(order)}
        assert idx["preprocess_s1"] < idx["train_s1"]
        assert idx["train_s1"] < idx["train_s2"]
        assert idx["preprocess_s2"] < idx["train_s2"]

    def test_circular_dependency_raises(self):
        wf = WorkflowDefinition(
            name="test-wf",
            stages=[
                WorkflowStage(id="a", type="train",
                              config_file="a.toml", depends_on=["b"]),
                WorkflowStage(id="b", type="train",
                              config_file="b.toml", depends_on=["a"]),
            ],
        )
        with pytest.raises(ValueError, match="circular"):
            wf.topological_order()


class TestStageOutput:
    def test_preprocess_output(self):
        out = StageOutput(
            stage_id="preprocess_s1",
            stage_type="preprocess",
            dataset_dir="runs/20260528/preprocess_s1/post_image_dataset",
            subsets=[
                {"name": "1_subset_a", "image_dir": ".../.resized",
                 "cache_dir": ".../.lora", "num_repeats": 1},
            ],
        )
        assert out.stage_id == "preprocess_s1"
        assert len(out.subsets) == 1
