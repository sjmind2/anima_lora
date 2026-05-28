from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class WorkflowStage(BaseModel):
    id: str
    type: str
    config_file: str
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"preprocess", "train"}
        if v not in allowed:
            raise ValueError(f"Invalid stage type: {v}. Must be one of {allowed}")
        return v


class SubsetInfo(BaseModel):
    name: str
    image_dir: str
    cache_dir: str
    num_repeats: int = 1


class StageOutput(BaseModel):
    stage_id: str
    stage_type: str
    dataset_dir: str = ""
    safetensors_path: str = ""
    checkpoint_dir: str = ""
    subsets: list[SubsetInfo] = Field(default_factory=list)


class WorkflowDefinition(BaseModel):
    name: str
    description: str = ""
    stages: list[WorkflowStage] = Field(default_factory=list)
    infrastructure: dict = Field(default_factory=dict)

    def topological_order(self) -> list[WorkflowStage]:
        stage_map = {s.id: s for s in self.stages}
        visited: set[str] = set()
        order: list[WorkflowStage] = []
        in_stack: set[str] = set()

        def visit(sid: str) -> None:
            if sid in visited:
                return
            if sid in in_stack:
                raise ValueError(f"circular dependency involving stage: {sid}")
            in_stack.add(sid)
            stage = stage_map[sid]
            for dep in stage.depends_on:
                if dep not in stage_map:
                    raise ValueError(f"unknown dependency: {dep}")
                visit(dep)
            in_stack.remove(sid)
            visited.add(sid)
            order.append(stage)

        for s in self.stages:
            visit(s.id)
        return order


class InfrastructureConfig(BaseModel):
    pretrained_model_name_or_path: str = ""
    qwen3: str = ""
    vae: str = ""
    mixed_precision: str = "bf16"
    attn_mode: str = "flex"
