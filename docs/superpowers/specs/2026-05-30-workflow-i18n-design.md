# Workflow Frontend i18n

Date: 2026-05-30

## Problem

All ~290 user-visible strings in the workflow frontend are hardcoded in a single language (Chinese in JS/HTML, English in Python backend). The YAML schemas contain ~138 Chinese labels. No i18n infrastructure exists. The goal is to support three languages (中文 / English / 日本語) with browser auto-detection + manual switching, and all i18n resources must live inside `workflow/`.

## Scope

| Layer | Strings | Current language | Approach |
|-------|---------|-----------------|----------|
| Frontend JS/HTML (11 files) | ~100+ | Chinese | Replace with `t('key')` |
| YAML schemas (7 files) | ~138 | Chinese | Runtime overlay (files unchanged) |
| Backend Python (~8 files) | ~55+ | English | Replace with `t('key', args)` |
| CLI scripts (`scripts/*.py`) | ~20 | English | **Not in scope** — keep English |

## Approach: Centralised translation keys (Option A)

All translation resources live under `workflow/i18n/`. Translation keys are stable dot-path identifiers (e.g. `app.workflowSaved`). The English JSON file is the baseline — missing keys in other locales fall back to English.

## Directory structure

```
workflow/
  i18n/
    __init__.py          ← re-export backend.py public API
    index.js             ← Frontend i18n runtime (~100 lines)
    backend.py           ← Backend Python i18n module
    schema_overlay.py    ← Schema translation overlay
    locales/
      en.json            ← English baseline (all keys present)
      zh-CN.json         ← Chinese (fallback to en)
      ja.json            ← Japanese (fallback to en)
```

## JSON format

Nested objects, not flat `a.b.c` keys. All locale files share the same structure.

```json
// en.json (baseline)
{
  "app": {
    "workflowSaved": "Workflow saved",
    "saveFailed": "Save failed: {error}",
    "itemsCount": "{n} items"
  },
  "schema": {
    "train_common": {
      "root": { "label": "Common Training Parameters", "description": "Parameters shared by all methods" },
      "group": { "training": "Training Hyperparameters" },
      "field": { "learning_rate": "Learning Rate" },
      "help": { "stop_epoch": "Stop after saving at this epoch" }
    }
  },
  "backend": {
    "config": {
      "notFound": "Workflow file not found: {path}"
    }
  }
}

// zh-CN.json
{
  "app": {
    "workflowSaved": "工作流已保存",
    "saveFailed": "保存失败: {error}"
  },
  "schema": {
    "train_common": {
      "root": { "label": "训练通用参数", "description": "所有训练方法共享的参数，切换方法时保留" },
      "group": { "training": "训练超参" },
      "field": { "learning_rate": "学习率" }
    }
  }
}
```

### Fallback chain

```
current locale JSON → en.json → key itself (e.g. "app.workflowSaved")
```

## Frontend i18n runtime (`index.js`)

### API

```javascript
await I18n.init()                           // load JSON, detect browser lang, read localStorage
t('app.workflowSaved')                      // → translated string
t('app.saveFailed', { error: 'disk full' }) // → interpolated string
I18n.setLocale('ja')                        // switch + persist to localStorage
I18n.getLocale()                            // → 'zh-CN' | 'en' | 'ja'
I18n.onChange(callback)                     // register re-render callback
```

### Loading strategy

1. `index.html` loads scripts in order: `i18n/index.js` → `app.js` → components
2. `I18n.init()` fetches `i18n/locales/{locale}.json` via `fetch()`
3. Browser detection: `navigator.language` → `zh-CN` (zh-*), `ja`, fallback `en`
4. `localStorage` key `anima-locale` overrides browser detection

### Vue integration

No Vue plugin — components call global `t()` directly in template strings:

```javascript
`<span>${t('app.workflowSaved')}</span>`
```

Language switch triggers `$forceUpdate` via `I18n.onChange`:

```javascript
I18n.onChange(() => vm.$forceUpdate?.())
```

### Language switcher UI

A compact dropdown in the top navbar, right side (left of the ⚙ settings button):

```
[🌐 中文 ▾]   ← click to expand
  ┌──────────┐
  │ 中文      │  ← current language highlighted
  │ English   │
  │ 日本語    │
  └──────────┘
```

- Pure CSS dropdown + click-outside-to-close
- No page refresh needed — instant re-render via `$forceUpdate`
- `api.js` injects `Accept-Language: I18n.getLocale()` header on every request

## Backend i18n (`backend.py`)

### API

```python
from workflow.i18n import t, get_locale, set_locale

t('backend.config.notFound', path='/foo/bar')
# → "Workflow file not found: /foo/bar"

get_locale()       # → 'zh-CN' | 'en' | 'ja'
set_locale('ja')   # per-request locale
```

### Locale source priority (backend)

```
1. Accept-Language request header (parsed in app.py middleware)
2. ANIMA_LOCALE environment variable (CLI scripts)
3. Default 'en'
```

### Implementation

- `contextvars.ContextVar` for per-request locale (thread-safe, async-safe)
- `app.py` middleware: parse `Accept-Language` → `set_locale()` → handle request
- JSON loaded once at module import (not per-request)

## Schema translation overlay (`schema_overlay.py`)

YAML schema files are **not modified**. The API endpoint applies translations at runtime.

```python
def translate_schema(schema: dict, locale: str) -> dict:
    """
    Walk schema dict, overlay translated values for:
    - label
    - description
    - help
    - choice_labels (map values)
    """
```

Called in `app.py`'s `GET /api/schemas/{name}` endpoint before returning.

Translation keys map to nested JSON:

```json
{
  "schema": {
    "train_common": {
      "root": { "label": "...", "description": "..." },
      "group": { "<group_key>": "..." },
      "field": { "<field_key>": "..." },
      "help": { "<field_key>": "..." },
      "choice_labels": { "<field_key>.<choice>": "..." }
    }
  }
}
```

## Implementation phases

| Phase | Content | Files touched |
|-------|---------|---------------|
| P0: Infrastructure | Create `i18n/` dir + `index.js` + `backend.py` + `schema_overlay.py` + `en.json` | 5 new files |
| P1: Frontend migration | 11 JS files: hardcoded Chinese → `t()` + language switcher in navbar | ~12 files |
| P2: Schema translation | Extract 138 strings into `en.json` schema section + `app.py` overlay hook | `en.json` + `app.py` |
| P3: Backend migration | Python files: English strings → `t()` + `app.py` middleware | ~8 files |
| P4: Chinese + Japanese | Create `zh-CN.json` + `ja.json` full translations | 2 files |
| P5: Verification | Browser test all three languages + fallback | — |

## Out of scope

- Modifying YAML schema files themselves
- Introducing npm / build toolchain
- Translating SSE event type identifiers (structural, not user-visible)
- Translating CLI script console output (`scripts/*.py` `print()`)
