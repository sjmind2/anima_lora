# Workflow UX Fixes: Run History, Script Output, Model Paths, Root Directory

Date: 2026-05-30

## Problem

Four UX issues in the workflow frontend:

1. **Run history broken**: Running logs overlay history records; status always shows ✗ regardless of actual outcome; no stage-level progress visible.
2. **Script output lost between stages**: When a workflow has multiple stages, entering the next stage quickly overwrites the previous stage's output due to a 500-line truncation on a single shared array.
3. **No model path configuration UI**: Infrastructure settings (model paths, hardware) exist in the schema and API but have no frontend entry point. Users cannot set model paths globally or override per-node.
4. **Workflow root directory in wrong location**: Default is `~/.anima_workflow` (user home), should be under the project directory. Not user-configurable.

## Sub-projects (implementation order)

A → B → C → D (fix bugs first, then add features)

---

## Sub-project A: Run History Status Fix + Log Viewer

### A1: Status tracking via status.json

Each run creates `{run_dir}/status.json`:

```json
{
  "status": "running",
  "current_stage": "preprocess_1",
  "stages": {
    "preprocess_1": {"status": "ok"},
    "train_1": {"status": "running"}
  },
  "updated_at": "2026-05-30T14:30:00"
}
```

- **Status values**: `running`, `ok`, `stopped`, `error`
- **Write timing**:
  - `status: running` + `current_stage` — when scheduler starts a stage
  - Per-stage `status` — when stage ends (ok/error/stopped)
  - Top-level `status` — when entire workflow ends
- **Read**: `_handle_list_runs()` reads `status.json` instead of parsing `run.log`. If file doesn't exist, status = "unknown".

### A2: History tab shows records only

The history tab displays a list of completed/running runs, not live logs:

```
2026-05-30 14:30  |  ✅ ok     |  预处理 ✓ → 训练 ✓      |  [查看日志] [打开目录]
2026-05-29 10:00  |  ❌ error  |  预处理 ✓ → 训练 ✗      |  [查看日志] [打开目录]
🔄 运行中          |  预处理 ✓ → 训练 🔄      |  [查看日志]
```

Each record shows: timestamp, status icon, stage progress chain, action buttons.

### A3: View log button + modal

- Each history record has a "查看日志" button
- Clicking opens a modal dialog showing the full `run.log` content for that run
- New/modified API: `GET /api/workflows/{name}/runs/{run_id}/log` — reads `{run_dir}/run.log` and returns `{"lines": ["..."]}`
- Modal supports scroll, close button

### Files to change

| File | Change |
|------|--------|
| `workflow/scheduler.py` | Write `status.json` on stage start/end/workflow end |
| `workflow/app.py` | `_handle_list_runs()` reads `status.json`; `_handle_log()` returns actual log content |
| `workflow/web/js/app.js` | History tab UI: records list with stage progress, view-log button, modal |
| `workflow/web/css/style.css` | Modal styles, history record styles |

---

## Sub-project B: Script Output Per-Stage Persistence

### B1: Per-stage log storage

Change `scriptLogs` from a single flat array to `Map<stageId, Array<{time, text}>>`:

```javascript
// Before: scriptLogs = ref([])
// After:  scriptLogs = ref({})  // keyed by stageId
```

- Each stage accumulates its own log entries independently
- Truncation (500 → 400) applies per stage, not globally
- `scriptProgress` (tqdm data) also keyed by stageId

### B2: Stage selector dropdown

Script output tab gets a dropdown at the top:

```
┌─ 脚本输出 ──────────────────────────────────┐
│  [阶段: 预处理 ▾]    进度: 100%  50/50      │
│  ────────────────────────────────────────── │
│  [14:30:01] Resizing image_001.png...       │
│  [14:30:02] Resizing image_002.png...       │
└─────────────────────────────────────────────┘
```

- Dropdown lists all stages (including completed ones)
- Selecting a stage shows that stage's log + progress
- Default selection: current running stage (or last completed if idle)

### Files to change

| File | Change |
|------|--------|
| `workflow/web/js/app.js` | `scriptLogs` → per-stage map; add stage selector dropdown; handle `stage_stdout_batch` per stageId |

---

## Sub-project C: Global Model Path Settings + Per-Node Override

### C1: Settings gear button in toolbar

Add a ⚙ gear icon button to the top toolbar (next to "打开工作流" / "新建工作流"). Clicking opens a modal dialog with:

**Global Settings Modal Sections:**

1. **工作流根目录** — path input + browse button
2. **模型路径** — three path fields:
   - DiT 模型 (`pretrained_model_name_or_path`)
   - 文本编码器 (`qwen3`)
   - VAE 模型 (`vae`)
3. **硬件设置** — existing infrastructure fields:
   - 混合精度 (`mixed_precision`)
   - 注意力模式 (`attn_mode`)

Each path field has a text input and a "浏览" (Browse) button.

### C2: File browse mechanism

New API endpoint `POST /api/browse`:

```json
// Request
{"action": "open", "type": "file", "filters": [{"name": "Safetensors", "extensions": ["safetensors"]}]}

// Response (from pywebview file dialog or fallback)
{"path": "O:/loratool/anima_lora_fork/models/diffusion_models/anima-base-v1.0.safetensors"}
```

For `--no-gui` mode (browser only), use native `<input type="file">` element hidden behind the browse button.

### C3: Per-node model path override

In each preprocess/train stage's config form, add a collapsible "高级设置 — 模型路径覆盖" section at the **top** of the form, **collapsed by default**.

When expanded, shows the same model path fields (DiT, text encoder, VAE) but empty by default. Empty = use global setting. Filled = override for this stage.

Implementation:
- Add model path fields to `preprocess.yaml` and `train_*.yaml` schemas with `layer: "override"`, `required: false`, `group: "advanced"`
- Frontend renders these in a collapsible section at the top of the form
- Merge logic already exists: `{**infrastructure, **config}` — stage config overrides infrastructure

### Files to change

| File | Change |
|------|--------|
| `workflow/web/js/app.js` | Add gear button, settings modal, browse functionality |
| `workflow/web/css/style.css` | Modal styles, gear button styles |
| `workflow/app.py` | Add `POST /api/browse` endpoint |
| `workflow/schemas/preprocess.yaml` | Add advanced group with model path override fields |
| `workflow/schemas/train_common.yaml` | Add advanced group with model path override fields |
| `workflow/stages/train.py` | No change needed (merge logic already correct) |
| `workflow/stages/preprocess.py` | No change needed (already uses infrastructure dict) |

---

## Sub-project D: Workflow Root Directory Configurable

### D1: Change default to project directory

Change default `workflows_root` from `Path.home() / ".anima_workflow"` to `{repo_root} / ".anima_workflow"`.

`repo_root` obtained via `library.env.anima_home()` or `Path(__file__).resolve().parent.parent` as fallback.

### D2: Persistent configuration file

Config file at `{repo_root}/.anima_workflow_config.json`:

```json
{
  "workflows_root": "O:/loratool/anima_lora_fork/.anima_workflow"
}
```

### D3: Startup resolution order

1. CLI `--workflows-root` (highest priority)
2. `.anima_workflow_config.json` `workflows_root` field
3. Default: `{repo_root}/.anima_workflow`

### D4: Runtime changes via settings modal

When user changes workflows root in the settings modal:
1. Write new value to `.anima_workflow_config.json`
2. Change takes effect on next server restart (show notification to user)
3. Do NOT move existing workflow data automatically

### Files to change

| File | Change |
|------|--------|
| `workflow/app.py` | `create_app()` reads config file, changes default |
| `workflow/__main__.py` | Pass resolved root to `create_app()` |
| `workflow/web/js/app.js` | Settings modal saves workflows_root via new API |

---

## Error handling

- `status.json` read failure → treat as "unknown" status
- `run.log` read failure → return empty lines with error message
- Browse API failure → show error in modal, fall back to manual text input
- Config file write failure → show error, keep using current root
- Invalid workflows_root path → create on startup, show error if not writable
