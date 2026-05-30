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

from workflow.i18n import t


def load_workflow_yaml(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(t("backend.config.workflowNotFound", path=p))
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
        raise FileNotFoundError(t("backend.config.tomlNotFound", path=p))
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
                raise ValueError(t("backend.config.unresolvedPlaceholderStage", stage=stage_id))
            outputs = stage_outputs[stage_id]
            if key not in outputs:
                raise ValueError(t("backend.config.unresolvedPlaceholderKey", key=key, stage=stage_id))
            return outputs[key]
        return _PLACEHOLDER_RE.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: resolve_placeholders(v, stage_outputs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_placeholders(item, stage_outputs) for item in obj]
    return obj


_SCHEMA_DIR = Path(__file__).parent / "schemas"


def load_schema(name: str) -> dict:
    schema_file = _SCHEMA_DIR / f"{name}.yaml"
    if not schema_file.exists():
        raise FileNotFoundError(t("backend.config.schemaNotFound", path=schema_file))
    with open(schema_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
