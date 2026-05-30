from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_LOCALES_DIR = Path(__file__).parent / "locales"
_current_locale: ContextVar[str] = ContextVar("locale", default="en")
_messages: dict[str, dict] = {}
_loaded: set[str] = set()


def _load_locale(name: str) -> dict:
    if name in _loaded:
        return _messages.get(name, {})
    path = _LOCALES_DIR / f"{name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    _messages[name] = data
    _loaded.add(name)
    return data


def _resolve(obj: Any, parts: list[str]) -> str | None:
    cur = obj
    for p in parts:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if isinstance(cur, str):
        return cur
    return None


def _interpolate(template: str, params: dict[str, Any]) -> str:
    if not params:
        return template
    for k, v in params.items():
        template = template.replace(f"{{{k}}}", str(v))
    return template


def get_locale() -> str:
    return _current_locale.get()


def set_locale(locale: str) -> None:
    _current_locale.set(locale)


def t(key: str, **params: Any) -> str:
    parts = key.split(".")
    locale = get_locale()
    if locale != "en":
        msgs = _load_locale(locale)
        val = _resolve(msgs, parts)
        if val is not None:
            return _interpolate(val, params)
    fallback = _load_locale("en")
    val = _resolve(fallback, parts)
    if val is not None:
        return _interpolate(val, params)
    return key
