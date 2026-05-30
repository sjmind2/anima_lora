# Workflow 自动化训练引擎 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建独立的 `workflow/` 模块，提供 WebUI + CLI 的自动化多阶段训练工作流引擎，并通过四阶段端到端验证（Preprocess S1 → Train S1 → Preprocess S2 → Train S2）。

**Architecture:** Schema 驱动的参数系统（YAML 定义参数元数据，前端动态生成表单）+ aiohttp REST API + Vue 3 CDN 前端 + pywebview 桌面窗口。调度器按拓扑排序执行阶段，通过 SSE 推送进度，通过 subprocess 调用现有预处理/训练脚本。

**Tech Stack:** Python 3.11+, aiohttp, pydantic, PyYAML, tomlkit, Vue 3 CDN, pywebview, safetensors

---

## File Structure

```
workflow/
├── __init__.py                    # 模块标记
├── __main__.py                    # python -m workflow 入口
├── app.py                         # aiohttp Application + 路由 + SSE
├── scheduler.py                   # 工作流调度器（拓扑排序 + 占位符替换 + 执行）
├── models.py                      # Pydantic 数据模型
├── config.py                      # YAML/TOML 读写 + 占位符替换
├── logger.py                      # 统一日志 + SSE 队列
├── schemas/
│   ├── preprocess.yaml            # 预处理参数 Schema
│   ├── train_common.yaml          # 训练通用参数 Schema
│   ├── train_lokr.yaml            # LoKR 方法特定参数
│   ├── train_lora.yaml            # LoRA 方法特定参数
│   ├── train_loha.yaml            # LoHA 方法特定参数
│   ├── train_locon.yaml           # LoCON 方法特定参数
│   ├── infrastructure.yaml        # 基础设施参数
│   └── combo_ortho.yaml           # Ortho 组合参数
│   └── combo_hydra.yaml           # MoE 组合参数
│   └── combo_tlora.yaml           # T-LoRA 组合参数
│   └── combo_reft.yaml            # ReFT 组合参数
├── stages/
│   ├── __init__.py
│   ├── base.py                    # 阶段基类
│   ├── preprocess.py              # 预处理执行器
│   └── train.py                   # 训练执行器
├── web/
│   ├── index.html                 # SPA 入口
│   ├── css/style.css              # 暗色主题样式
│   └── js/
│       ├── app.js                 # Vue 3 应用
│       ├── api.js                 # HTTP API 封装
│       └── components/
│           ├── WelcomeScreen.js   # 欢迎/创建页面
│           ├── StageList.js       # 阶段面板
│           ├── StageCard.js       # 阶段卡片
│           ├── SchemaForm.js      # Schema→表单渲染
│           ├── FieldRenderer.js   # 单字段渲染
│           ├── MethodSelector.js  # 方法选择器+组合开关
│           ├── DatasetSelector.js # 数据集选择器
│           ├── RunControl.js      # 运行/停止控制+总进度
│           ├── LogViewer.js       # 日志查看器
│           ├── LossChart.js       # SVG Loss/LR 曲线
│           └── InfraSettings.js   # 基础设施配置
├── scripts/
│   ├── run_workflow.py            # CLI 运行
│   └── create_workflow.py         # CLI 创建
└── templates/
    ├── preprocess_default.toml
    └── train_lokr_default.toml
```

---

### Task 1: 项目骨架 + Pydantic 数据模型

**Files:**
- Create: `workflow/__init__.py`
- Create: `workflow/models.py`
- Test: `tests/test_workflow_models.py`

- [ ] **Step 1: 写 models.py 的测试**

```python
# tests/test_workflow_models.py
import pytest
from workflow.models import (
    WorkflowStage, WorkflowDefinition, StageOutput,
    PreprocessConfig, TrainConfig, InfrastructureConfig,
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_models.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: 实现 models.py**

```python
# workflow/__init__.py
```

```python
# workflow/models.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_models.py -v`
Expected: All PASS

- [ ] **Step 5: 提交**

```bash
git add workflow/__init__.py workflow/models.py tests/test_workflow_models.py
git commit -m "feat(workflow): add Pydantic data models for workflow engine"
```

---

### Task 2: 配置管理（YAML/TOML 读写 + 占位符替换）

**Files:**
- Create: `workflow/config.py`
- Test: `tests/test_workflow_config.py`

- [ ] **Step 1: 写 config.py 的测试**

```python
# tests/test_workflow_config.py
import pytest
import tempfile
from pathlib import Path
from workflow.config import (
    load_workflow_yaml,
    save_workflow_yaml,
    load_stage_toml,
    save_stage_toml,
    resolve_placeholders,
)


class TestWorkflowYaml:
    def test_load_and_save(self, tmp_path):
        wf_data = {
            "name": "test",
            "stages": [
                {"id": "p1", "type": "preprocess", "config_file": "p1.toml", "depends_on": []},
            ],
        }
        wf_file = tmp_path / "workflow.yaml"
        save_workflow_yaml(wf_data, wf_file)
        loaded = load_workflow_yaml(wf_file)
        assert loaded["name"] == "test"
        assert len(loaded["stages"]) == 1

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_workflow_yaml(tmp_path / "missing.yaml")


class TestStageToml:
    def test_load_and_save(self, tmp_path):
        toml_data = {
            "network_type": "lokr",
            "network_dim": 16,
            "learning_rate": 0.0004,
        }
        toml_file = tmp_path / "train.toml"
        save_stage_toml(toml_data, toml_file)
        loaded = load_stage_toml(toml_file)
        assert loaded["network_type"] == "lokr"
        assert loaded["network_dim"] == 16


class TestPlaceholderResolution:
    def test_resolve_stage_output(self):
        stage_outputs = {
            "preprocess_s1": {
                "dataset_dir": "/runs/20260528/preprocess_s1/post_image_dataset",
            },
            "train_s1": {
                "safetensors_path": "/runs/20260528/train_s1/output/anima_lokr.safetensors",
            },
        }
        text = "${preprocess_s1.dataset_dir}/hanechan/.resized"
        result = resolve_placeholders(text, stage_outputs)
        assert result == "/runs/20260528/preprocess_s1/post_image_dataset/hanechan/.resized"

    def test_resolve_nested_placeholder(self):
        stage_outputs = {
            "train_s1": {"safetensors_path": "/path/to/model.safetensors"},
        }
        toml_data = {"network_weights": "${train_s1.safetensors_path}"}
        result = resolve_placeholders(toml_data, stage_outputs)
        assert result["network_weights"] == "/path/to/model.safetensors"

    def test_resolve_dict_recursively(self):
        stage_outputs = {"p1": {"dataset_dir": "/data"}}
        toml_data = {
            "key1": "${p1.dataset_dir}/a",
            "sections": {"key2": "${p1.dataset_dir}/b"},
        }
        result = resolve_placeholders(toml_data, stage_outputs)
        assert result["key1"] == "/data/a"
        assert result["sections"]["key2"] == "/data/b"

    def test_unresolved_raises(self):
        with pytest.raises(ValueError, match="unresolved"):
            resolve_placeholders("${missing.foo}", {})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_config.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 config.py**

```python
# workflow/config.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import tomli_w


def load_workflow_yaml(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_workflow_yaml(data: dict, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_stage_toml(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TOML file not found: {p}")
    with open(p, "rb") as f:
        return tomllib.load(f)


def save_stage_toml(data: dict, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        tomli_w.dump(data, f)


_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\.(\w+)\}")


def resolve_placeholders(obj: Any, stage_outputs: dict[str, dict[str, str]]) -> Any:
    if isinstance(obj, str):
        def _replace(m):
            stage_id = m.group(1)
            key = m.group(2)
            if stage_id not in stage_outputs:
                raise ValueError(f"unresolved placeholder: stage '{stage_id}' not found")
            outputs = stage_outputs[stage_id]
            if key not in outputs:
                raise ValueError(f"unresolved placeholder: '{key}' not in stage '{stage_id}' outputs")
            return outputs[key]
        return _PLACEHOLDER_RE.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: resolve_placeholders(v, stage_outputs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_placeholders(item, stage_outputs) for item in obj]
    return obj
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_config.py -v`
Expected: All PASS

- [ ] **Step 5: 提交**

```bash
git add workflow/config.py tests/test_workflow_config.py
git commit -m "feat(workflow): add config management with YAML/TOML and placeholder resolution"
```

---

### Task 3: Schema 加载器 + Schema YAML 文件

**Files:**
- Create: `workflow/schemas/preprocess.yaml`
- Create: `workflow/schemas/train_common.yaml`
- Create: `workflow/schemas/train_lokr.yaml`
- Create: `workflow/schemas/train_lora.yaml`
- Create: `workflow/schemas/infrastructure.yaml`
- Test: `tests/test_workflow_schema.py`

- [ ] **Step 1: 写 schema 加载器的测试**

```python
# tests/test_workflow_schema.py
import pytest
from pathlib import Path
from workflow.config import load_schema


class TestSchemaLoading:
    def test_load_preprocess_schema(self):
        schema = load_schema("preprocess")
        assert schema["type"] == "preprocess"
        groups = schema["groups"]
        group_names = [g["name"] for g in groups]
        assert "data_source" in group_names
        assert "bucket" in group_names

    def test_load_train_common_schema(self):
        schema = load_schema("train_common")
        assert schema["type"] == "train_common"
        all_fields = []
        for g in schema["groups"]:
            all_fields.extend(g["fields"])
        keys = [f["key"] for f in all_fields]
        assert "learning_rate" in keys
        assert "max_train_epochs" in keys
        assert "optimizer_type" in keys

    def test_load_train_lokr_schema(self):
        schema = load_schema("train_lokr")
        assert schema["method"] == "lokr"
        all_fields = []
        for g in schema["groups"]:
            all_fields.extend(g["fields"])
        keys = [f["key"] for f in all_fields]
        assert "lokr_factor" in keys
        assert "decompose_both" in keys
        assert "scale_weight_norms" in keys

    def test_load_infrastructure_schema(self):
        schema = load_schema("infrastructure")
        assert schema["type"] == "infrastructure"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent_method")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_schema.py -v`
Expected: FAIL

- [ ] **Step 3: 创建所有 Schema YAML 文件**

创建 `workflow/schemas/preprocess.yaml`、`train_common.yaml`、`train_lokr.yaml`、`train_lora.yaml`、`infrastructure.yaml`，内容参照设计文档 §4.7 ~ §4.9 中的完整 Schema 定义。每个文件包含 `type`、`label`、`description`、`groups`（每个 group 包含 `name`、`label`、`fields`），每个 field 包含 `key`、`type`、`required`、`label`、`default` 等完整元数据。

`train_lokr.yaml` 的 method 特定参数包括：network_type(lokr), network_dim(16), network_alpha(8), conv_dim(1), conv_alpha(4), lokr_factor(8), decompose_both(true), use_tucker(true), use_scalar(false), weight_decompose(false), full_matrix(false), scale_weight_norms(1.0, auto_set)。

`train_common.yaml` 的 optimizer_type choices 包含完整列表：AdamW, AdamW8bit, Lion, CAME, Prodigy, Prodigy_Adv, Adopt_Adv, Adafactor, RAdamScheduleFree, AdamWScheduleFree, SGDScheduleFree, DAdaptAdam, DAdaptSGD, PagedAdamW, PagedAdamW8bit。

- [ ] **Step 4: 在 config.py 中添加 load_schema 函数**

在 `workflow/config.py` 中添加：

```python
_SCHEMA_DIR = Path(__file__).parent / "schemas"


def load_schema(name: str) -> dict:
    schema_file = _SCHEMA_DIR / f"{name}.yaml"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_file}")
    with open(schema_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_schema.py -v`
Expected: All PASS

- [ ] **Step 6: 提交**

```bash
git add workflow/schemas/ workflow/config.py tests/test_workflow_schema.py
git commit -m "feat(workflow): add parameter Schema YAML files and loader"
```

---

### Task 4: 统一日志 + SSE 事件队列

**Files:**
- Create: `workflow/logger.py`
- Test: `tests/test_workflow_logger.py`

- [ ] **Step 1: 写 logger 的测试**

```python
# tests/test_workflow_logger.py
import pytest
import asyncio
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_logger.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 logger.py**

```python
# workflow/logger.py
from __future__ import annotations

import json
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_logger.py -v`
Expected: All PASS

- [ ] **Step 5: 提交**

```bash
git add workflow/logger.py tests/test_workflow_logger.py
git commit -m "feat(workflow): add unified logger with SSE event queue"
```

---

### Task 5: 阶段执行器（Preprocess + Train）

**Files:**
- Create: `workflow/stages/__init__.py`
- Create: `workflow/stages/base.py`
- Create: `workflow/stages/preprocess.py`
- Create: `workflow/stages/train.py`
- Test: `tests/test_workflow_stages.py`

- [ ] **Step 1: 写阶段执行器的测试**

```python
# tests/test_workflow_stages.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from workflow.stages.base import StageBase, StageResult
from workflow.stages.preprocess import PreprocessExecutor
from workflow.stages.train import TrainExecutor


class TestPreprocessExecutor:
    def test_build_resize_command(self, tmp_path):
        config = {
            "source_image_dir": "O:/LoRATraining/hanechan",
            "bucket_families": ["S1"],
        }
        stage_dir = tmp_path / "preprocess_s1"
        stage_dir.mkdir()
        infra = {"pretrained_model_name_or_path": "", "vae": "", "qwen3": ""}
        executor = PreprocessExecutor("preprocess_s1", config, stage_dir, infra)
        cmd = executor._build_resize_cmd()
        assert "resize_images.py" in cmd[0] or "resize_images.py" in str(cmd)
        assert "--bucket_families" in cmd

    def test_discover_subsets_after_run(self, tmp_path):
        stage_dir = tmp_path / "preprocess_s1"
        post_dir = stage_dir / "post_image_dataset" / "hanechan" / "1_subset_a"
        resized = post_dir / ".resized"
        lora = post_dir / ".lora"
        resized.mkdir(parents=True)
        lora.mkdir(parents=True)
        (resized / "img.png").write_bytes(b"fake")
        (lora / "img_anima_te.safetensors").write_bytes(b"fake")
        config = {"source_image_dir": "O:/LoRATraining/hanechan"}
        executor = PreprocessExecutor("preprocess_s1", config, stage_dir, {})
        subsets = executor.discover_subsets()
        assert len(subsets) == 1
        assert subsets[0].name == "1_subset_a"
        assert subsets[0].num_repeats == 1


class TestTrainExecutor:
    def test_build_train_cmd_with_stop_epoch(self, tmp_path):
        config = {
            "network_type": "lokr",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.0004,
            "lr_scheduler": "cosine",
            "max_train_epochs": 10,
            "stop_epoch": 6,
            "optimizer_type": "CAME",
        }
        stage_dir = tmp_path / "train_s1"
        stage_dir.mkdir()
        infra = {"pretrained_model_name_or_path": "/dit", "vae": "/vae", "qwen3": "/te", "mixed_precision": "bf16"}
        executor = TrainExecutor("train_s1", config, stage_dir, infra)
        resolved_config = executor.prepare_config({})
        assert resolved_config["max_train_epochs"] == 6
        assert resolved_config["save_every_n_epochs"] == 6

    def test_build_train_cmd_with_network_weights(self, tmp_path):
        config = {
            "network_type": "lokr",
            "network_dim": 16,
            "network_alpha": 8,
            "learning_rate": 0.000138,
            "max_train_epochs": 4,
            "network_weights": "/path/to/checkpoint.safetensors",
        }
        stage_dir = tmp_path / "train_s2"
        stage_dir.mkdir()
        executor = TrainExecutor("train_s2", config, stage_dir, {})
        resolved = executor.prepare_config({})
        assert resolved["network_weights"] == "/path/to/checkpoint.safetensors"
        assert resolved["dim_from_weights"] is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_stages.py -v`
Expected: FAIL

- [ ] **Step 3: 实现阶段执行器**

`workflow/stages/__init__.py`: 空

`workflow/stages/base.py`:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from workflow.models import SubsetInfo


class StageResult:
    def __init__(self, success: bool, outputs: dict[str, str] | None = None,
                 subsets: list[SubsetInfo] | None = None, error: str = ""):
        self.success = success
        self.outputs = outputs or {}
        self.subsets = subsets or []
        self.error = error


class StageBase(ABC):
    def __init__(self, stage_id: str, config: dict, stage_dir: Path, infrastructure: dict):
        self.stage_id = stage_id
        self.config = config
        self.stage_dir = stage_dir
        self.infrastructure = infrastructure

    @abstractmethod
    def prepare_config(self, stage_outputs: dict) -> dict: ...

    @abstractmethod
    def execute(self, on_stdout, on_progress) -> StageResult: ...
```

`workflow/stages/preprocess.py`: 实现 PreprocessExecutor，包含 `_build_resize_cmd()`、`_build_vae_cmd()`、`_build_te_cmd()`、`discover_subsets()` 方法。`execute()` 按顺序调用三步 subprocess，解析 tqdm 输出调用 `on_progress`。

`workflow/stages/train.py`: 实现 TrainExecutor，包含 `prepare_config()`（处理 stop_epoch → max_train_epochs 覆盖、network_weights + dim_from_weights 注入）和 `execute()` 方法。`execute()` 组装 train.py CLI 参数列表，通过 subprocess 调用，同时增量读取 progress.jsonl 和解析 stdout tqdm。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_stages.py -v`
Expected: All PASS

- [ ] **Step 5: 提交**

```bash
git add workflow/stages/ tests/test_workflow_stages.py
git commit -m "feat(workflow): add stage executors for preprocess and train"
```

---

### Task 6: 工作流调度器

**Files:**
- Create: `workflow/scheduler.py`
- Test: `tests/test_workflow_scheduler.py`

- [ ] **Step 1: 写调度器的测试**

```python
# tests/test_workflow_scheduler.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_scheduler.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 scheduler.py**

`workflow/scheduler.py`: 实现 WorkflowScheduler 类。核心方法：
- `_create_run_dir()`: 创建 `runs/{timestamp}/` 目录
- `_resolve_and_write_config()`: 读取阶段 TOML → resolve_placeholders → 写入运行目录
- `run()`: 按拓扑排序执行每个阶段，收集输出，传递给下游阶段
- `stop()`: 设置停止标志，杀死当前子进程
- 进度捕获：StageProgressWatcher 类，轮询 progress.jsonl + 解析 stdout tqdm

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_scheduler.py -v`
Expected: All PASS

- [ ] **Step 5: 提交**

```bash
git add workflow/scheduler.py tests/test_workflow_scheduler.py
git commit -m "feat(workflow): add workflow scheduler with topological execution"
```

---

### Task 7: aiohttp HTTP 服务 + REST API + SSE

**Files:**
- Create: `workflow/app.py`
- Test: `tests/test_workflow_app.py`

- [ ] **Step 1: 写 API 的测试**

```python
# tests/test_workflow_app.py
import pytest
from pathlib import Path
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from workflow.app import create_app


@pytest.fixture
def app(tmp_path):
    wf_root = tmp_path / "workflows"
    wf_root.mkdir()
    return create_app(workflows_root=wf_root)


@pytest.fixture
async def client(aiohttp_client, app):
    return await aiohttp_client(app)


class TestAPI:
    @pytest.mark.asyncio
    async def test_list_workflows_empty(self, client):
        resp = await client.get("/api/workflows")
        assert resp.status == 200
        data = await resp.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_create_workflow(self, client):
        resp = await client.post("/api/workflows", json={"name": "test-wf"})
        assert resp.status == 201
        data = await resp.json()
        assert data["name"] == "test-wf"

    @pytest.mark.asyncio
    async def test_get_schema(self, client):
        resp = await client.get("/api/schemas/preprocess")
        assert resp.status == 200
        data = await resp.json()
        assert data["type"] == "preprocess"

    @pytest.mark.asyncio
    async def test_get_schemas_train_common(self, client):
        resp = await client.get("/api/schemas/train_common")
        assert resp.status == 200
        data = await resp.json()
        assert data["type"] == "train_common"

    @pytest.mark.asyncio
    async def test_get_infrastructure_schema(self, client):
        resp = await client.get("/api/schemas/infrastructure")
        assert resp.status == 200
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_workflow_app.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 app.py**

`workflow/app.py`: 实现 `create_app()` 函数，注册所有 REST API 路由：
- `GET /api/workflows` → 列出工作流
- `POST /api/workflows` → 创建工作流
- `GET /api/workflows/{name}` → 获取详情
- `PUT /api/workflows/{name}` → 更新工作流
- `DELETE /api/workflows/{name}/runs` → 清除运行记录
- `GET/PUT /api/workflows/{name}/infrastructure` → 基础设施配置
- `POST /api/workflows/{name}/run` → 启动调度器
- `POST /api/workflows/{name}/stop` → 停止
- `GET /api/runs/{run_id}/events` → SSE 流
- `GET /api/runs/{run_id}/log` → 日志
- `GET /api/runs/{run_id}/loss-curve` → loss 数据
- `GET /api/schemas/{stage_type}` → Schema
- `GET /api/recent-workflows` → 最近列表
- `POST /api/import-toml` → TOML 导入
- Static file serving for `/` → web/index.html

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_workflow_app.py -v`
Expected: All PASS

- [ ] **Step 5: 提交**

```bash
git add workflow/app.py tests/test_workflow_app.py
git commit -m "feat(workflow): add aiohttp REST API with SSE and schema endpoints"
```

---

### Task 8: 入口 + pywebview 双模式

**Files:**
- Create: `workflow/__main__.py`
- Modify: `pyproject.toml` (add pywebview dependency)

- [ ] **Step 1: 实现 __main__.py**

```python
# workflow/__main__.py
import argparse
import sys
import threading
from workflow.app import create_app, start_server


def main():
    parser = argparse.ArgumentParser(description="Anima Workflow Engine")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-gui", action="store_true", help="Run without webview (browser mode)")
    parser.add_argument("--workflows-root", type=str, default=None)
    args = parser.parse_args()

    app = create_app(workflows_root=args.workflows_root)
    port = args.port

    if args.no_gui:
        start_server(app, port)
    else:
        try:
            import webview
            server_thread = threading.Thread(target=start_server, args=(app, port), daemon=True)
            server_thread.start()
            webview.create_window("Anima Workflow", f"http://localhost:{port}", width=1200, height=800)
        except ImportError:
            print("pywebview not installed, falling back to browser mode")
            start_server(app, port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 添加 pywebview 到 pyproject.toml**

在 `pyproject.toml` 的 dependencies 列表中添加 `"pywebview>=5.0"`。

- [ ] **Step 3: 手动验证启动**

Run: `python -m workflow --no-gui --port 8765` (短时启动验证无报错后 Ctrl+C)

- [ ] **Step 4: 提交**

```bash
git add workflow/__main__.py pyproject.toml
git commit -m "feat(workflow): add dual-mode entry point (pywebview + browser)"
```

---

### Task 9: 前端 WebUI — 基础框架 + 暗色主题

**Files:**
- Create: `workflow/web/index.html`
- Create: `workflow/web/css/style.css`
- Create: `workflow/web/js/app.js`
- Create: `workflow/web/js/api.js`

- [ ] **Step 1: 创建 index.html**

单页应用入口，引入 Vue 3 CDN、app.js、api.js 和 style.css。包含 Vue 3 挂载点 `<div id="app">`。

- [ ] **Step 2: 创建 style.css**

实现 §11 中定义的暗色主题：窗口背景 `#1e1e1e`、面板 `#232323`、输入框 `#2a2a2a`、文本 `#dcdcdc`、高亮 `#3c78c8`、边框 `#555`。按钮颜色语义：绿 `#27ae60`、红 `#c0392b`、蓝 `#2980b9`、橙 `#e67e22`、灰 `#7f8c8d`。字体、间距按 §11.3-4 定义。

- [ ] **Step 3: 创建 api.js**

封装所有 HTTP API 调用：`listWorkflows()`, `createWorkflow(name)`, `getWorkflow(name)`, `updateWorkflow(name, data)`, `runWorkflow(name)`, `stopWorkflow(name)`, `getSchema(type)`, `getLossCurve(runId, stageId)`, `importToml(tomlText, schemaType)`。SSE 连接：`connectEventStream(runId, onEvent)`。

- [ ] **Step 4: 创建 app.js**

Vue 3 应用主入口，管理全局状态：当前工作流、阶段列表、运行状态、SSE 事件处理。实现欢迎页/工作流页面的路由切换。

- [ ] **Step 5: 手动验证**

Run: `python -m workflow --no-gui` → 浏览器打开 `http://localhost:8765` → 确认暗色主题渲染正常

- [ ] **Step 6: 提交**

```bash
git add workflow/web/
git commit -m "feat(workflow): add WebUI base with dark theme and Vue 3 app shell"
```

---

### Task 10: 前端组件 — StageList + StageCard + SchemaForm

**Files:**
- Create: `workflow/web/js/components/StageList.js`
- Create: `workflow/web/js/components/StageCard.js`
- Create: `workflow/web/js/components/SchemaForm.js`
- Create: `workflow/web/js/components/FieldRenderer.js`
- Create: `workflow/web/js/components/MethodSelector.js`

- [ ] **Step 1: 实现 StageList.js**

阶段列表面板组件：显示阶段卡片列表、添加阶段按钮（下拉选择 Preprocess/Train）、拖拽排序（HTML5 drag API）、运行控制（运行/停止按钮 + 总进度条）。

- [ ] **Step 2: 实现 StageCard.js**

阶段卡片组件：显示类型图标 + 自动编号 + 状态（⏳等待/▶启动/🔄运行/✅完成/❌失败）+ 迷你进度条 + 依赖关系标注。点击选中 → 右侧加载配置表单。

- [ ] **Step 3: 实现 SchemaForm.js**

Schema 驱动动态表单组件：从 API 加载 Schema → 按 group 分区渲染 → Basic 始终展开 / Advanced 折叠 → 必选红 `*` / 条件必选橙 `*` / 自动设置蓝 `⚡` → 切换方法时保留 commonParams。

- [ ] **Step 4: 实现 FieldRenderer.js**

单字段渲染器：根据 Schema type 分发控件（int→number, float→text, bool→toggle, enum→select, path→input+browse, dataset_ref→checkbox list, checkpoint_ref→radio list）。

- [ ] **Step 5: 实现 MethodSelector.js**

方法选择器：基础类型下拉（LoRA/LoHA/LoCON/LoKR）+ 组合开关面板（仅 LoRA 基础类型显示 Ortho/MoE/T-LoRA/ReFT）。切换方法时：保存 commonParams → 加载新方法 Schema → 恢复 commonParams。

- [ ] **Step 6: 手动验证**

浏览器中：添加 Preprocess 阶段 → 看到动态表单 → 切换到 Train 阶段 → 选择 LoKR → 看到方法特定参数 → 切换到 LoRA → 通用参数保留

- [ ] **Step 7: 提交**

```bash
git add workflow/web/js/components/
git commit -m "feat(workflow): add StageList, StageCard, SchemaForm, MethodSelector components"
```

---

### Task 11: 前端组件 — DatasetSelector + LogViewer + LossChart + RunControl

**Files:**
- Create: `workflow/web/js/components/DatasetSelector.js`
- Create: `workflow/web/js/components/LogViewer.js`
- Create: `workflow/web/js/components/LossChart.js`
- Create: `workflow/web/js/components/RunControl.js`
- Create: `workflow/web/js/components/InfraSettings.js`

- [ ] **Step 1: 实现 DatasetSelector.js**

数据集选择器：复选框列出可用 Preprocess 阶段输出 → 展开显示子集列表 → num_repeats 编辑表格（自动从目录前缀推断 / 自定义）。

- [ ] **Step 2: 实现 LogViewer.js**

日志查看器：monospace 渲染 + 自动滚动 + 暂停滚动（用户上滚时暂停）+ 搜索高亮 + 阶段过滤下拉 + 日志级别彩色标记（ℹ️蓝/⚠️黄/❌红/✅绿）+ 5000 行限制。

- [ ] **Step 3: 实现 LossChart.js**

SVG Loss/LR 曲线：纯 SVG + JS（零外部依赖），双层曲线（原始浅色 + EMA 深色，α=0.05），X 轴 step + Y 轴自适应，多阶段对比模式（竖线分隔 + 颜色区分），降采样（≤500 不降 / 500-2000 每2取1保留min/max / >10000 LTTB），鼠标悬停十字线。

- [ ] **Step 4: 实现 RunControl.js**

运行控制：运行按钮（绿色）+ 停止按钮（红色）+ 总进度条 + 各阶段进度指示。

- [ ] **Step 5: 实现 InfraSettings.js**

基础设施配置面板：模型路径 + 硬件设置，保存到 workflow.yaml 的 infrastructure 节。

- [ ] **Step 6: 手动验证**

浏览器中完整操作：创建工作流 → 配置基础设施 → 添加阶段 → 配置参数 → 保存 → 确认配置持久化

- [ ] **Step 7: 提交**

```bash
git add workflow/web/js/components/
git commit -m "feat(workflow): add DatasetSelector, LogViewer, LossChart, RunControl, InfraSettings"
```

---

### Task 12: CLI 脚本 + 配置模板

**Files:**
- Create: `workflow/scripts/run_workflow.py`
- Create: `workflow/scripts/create_workflow.py`
- Create: `workflow/templates/preprocess_default.toml`
- Create: `workflow/templates/train_lokr_default.toml`

- [ ] **Step 1: 实现 run_workflow.py**

CLI 脚本：接受工作流目录参数 → 加载 workflow.yaml → 创建调度器 → 按拓扑排序执行 → 日志输出到 stdout + 文件。支持 `--clear-runs` 清除运行目录。

- [ ] **Step 2: 实现 create_workflow.py**

CLI 脚本：接受名称和存储位置 → 创建目录结构 + 空 workflow.yaml + configs/ 目录。

- [ ] **Step 3: 创建默认模板**

`preprocess_default.toml`: 默认预处理配置（bucket_families=S1, drop_lowres_images=true）
`train_lokr_default.toml`: 默认 LoKR 训练配置（network_dim=16, network_alpha=8, lokr_factor=8, learning_rate=0.0004, lr_scheduler=cosine, optimizer_type=CAME）

- [ ] **Step 4: 手动验证**

Run: `python workflow/scripts/create_workflow.py --name test-wf --root O:\loratool\anima_lora_fork\workflows`

- [ ] **Step 5: 提交**

```bash
git add workflow/scripts/ workflow/templates/
git commit -m "feat(workflow): add CLI scripts and default config templates"
```

---

### Task 13: 端到端验证 — 四阶段工作流

**Files:**
- Create: `workflows/hanechan-lokr-two-stage/workflow.yaml`
- Create: `workflows/hanechan-lokr-two-stage/configs/preprocess_s1.toml`
- Create: `workflows/hanechan-lokr-two-stage/configs/train_s1.toml`
- Create: `workflows/hanechan-lokr-two-stage/configs/preprocess_s2.toml`
- Create: `workflows/hanechan-lokr-two-stage/configs/train_s2.toml`

- [ ] **Step 1: 清除旧运行目录**

```bash
Remove-Item -Recurse -Force "O:\loratool\anima_lora_fork\workflows\hanechan-lokr-two-stage\runs\*" -ErrorAction SilentlyContinue
```

- [ ] **Step 2: 创建工作流配置文件**

`workflow.yaml`:
```yaml
name: hanechan-lokr-two-stage
description: "双阶段 LoKR 训练验证 (S1→Train1→S2→Train2)"

infrastructure:
  mixed_precision: "bf16"
  attn_mode: "flex"

stages:
  - id: preprocess_s1
    type: preprocess
    config_file: preprocess_s1.toml
    depends_on: []

  - id: train_s1
    type: train
    config_file: train_s1.toml
    depends_on: [preprocess_s1]

  - id: preprocess_s2
    type: preprocess
    config_file: preprocess_s2.toml
    depends_on: []

  - id: train_s2
    type: train
    config_file: train_s2.toml
    depends_on: [train_s1, preprocess_s2]
```

`configs/preprocess_s1.toml`:
```toml
source_image_dir = "O:/LoRATraining/hanechan"
bucket_families = ["S1"]
drop_lowres_images = true
min_pixels = 500000
```

`configs/train_s1.toml`:
```toml
network_type = "lokr"
network_dim = 16
network_alpha = 8
conv_dim = 1
conv_alpha = 4
lokr_factor = 8
decompose_both = true
use_tucker = true
scale_weight_norms = 1.0

learning_rate = 0.0004
lr_scheduler = "cosine"
max_train_epochs = 10
stop_epoch = 6
optimizer_type = "CAME"

output_name = "anima_lokr"

[[datasets]]
validation_split_num = 0

[[datasets.subsets]]
image_dir = "${preprocess_s1.dataset_dir}/hanechan/.resized"
cache_dir = "${preprocess_s1.dataset_dir}/hanechan/.lora"
num_repeats = 1
recursive = true
```

`configs/preprocess_s2.toml`:
```toml
source_image_dir = "O:/LoRATraining/hanechan"
bucket_families = ["S2"]
drop_lowres_images = true
min_pixels = 500000
```

`configs/train_s2.toml`:
```toml
network_type = "lokr"
network_dim = 16
network_alpha = 8
conv_dim = 1
conv_alpha = 4
lokr_factor = 8
decompose_both = true
use_tucker = true
scale_weight_norms = 1.0

learning_rate = 0.000138
lr_scheduler = "constant"
max_train_epochs = 4
optimizer_type = "CAME"

network_weights = "${train_s1.safetensors_path}"
dim_from_weights = true

output_name = "anima_lokr"

[[datasets]]
validation_split_num = 0

[[datasets.subsets]]
image_dir = "${preprocess_s1.dataset_dir}/hanechan/.resized"
cache_dir = "${preprocess_s1.dataset_dir}/hanechan/.lora"
num_repeats = 1
recursive = true

[[datasets]]
validation_split_num = 0

[[datasets.subsets]]
image_dir = "${preprocess_s2.dataset_dir}/hanechan/.resized"
cache_dir = "${preprocess_s2.dataset_dir}/hanechan/.lora"
num_repeats = 1
recursive = true
```

- [ ] **Step 3: 通过 CLI 运行工作流**

```bash
cd O:\loratool\anima_lora_fork
python workflow/scripts/run_workflow.py --workflow-dir workflows/hanechan-lokr-two-stage
```

- [ ] **Step 4: 检查运行结果**

验证：
1. `runs/` 下创建了时间戳目录
2. `preprocess_s1/` 目录下有 `post_image_dataset/hanechan/` 目录树（含 `.resized/` 和 `.lora/`）
3. `train_s1/` 目录下有 `output/anima_lokr-000006.safetensors`
4. `preprocess_s2/` 目录下有独立目录树（S2 分辨率）
5. `train_s2/` 目录下有 `output/anima_lokr.safetensors`
6. `run.log` 无 ERROR 行
7. 各阶段目录的 `config.toml` 中占位符已正确替换

- [ ] **Step 5: 提交**

```bash
git add workflows/hanechan-lokr-two-stage/
git commit -m "feat(workflow): add e2e verification workflow config for dual-stage LoKR training"
```

---

## Task Dependencies

- Task 2 依赖 Task 1（config 需要 models 的类型）
- Task 3 依赖 Task 2（schema 加载器需要 config.py 的 load_schema）
- Task 4 独立（logger 不依赖 models/config）
- Task 5 依赖 Task 1 + Task 2（stages 需要 models + config）
- Task 6 依赖 Task 1 + Task 2 + Task 4 + Task 5（scheduler 组合所有组件）
- Task 7 依赖 Task 3 + Task 6（API 需要 schemas + scheduler）
- Task 8 依赖 Task 7（入口需要 app）
- Task 9 依赖 Task 7（前端需要 API 端点）
- Task 10 依赖 Task 9（组件需要基础框架）
- Task 11 依赖 Task 10（更多组件）
- Task 12 依赖 Task 6（CLI 需要 scheduler）
- Task 13 依赖 Task 1-12（端到端验证需要全部组件）
