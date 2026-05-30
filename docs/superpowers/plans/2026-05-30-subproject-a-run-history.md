# Sub-project A: Run History Status Fix + Log Viewer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix run history to show correct per-stage status via status.json, add "查看日志" button with modal, and stop overlaying live logs on history.

**Architecture:** Scheduler writes `status.json` at each stage transition. Backend reads `status.json` instead of parsing `run.log`. Frontend history tab shows records with stage progress chains and a log-view modal.

**Tech Stack:** Python (aiohttp backend), JavaScript (Vue 3 frontend), JSON status file

---

## File Structure

| File | Responsibility |
|------|---------------|
| `workflow/scheduler.py` | Write `status.json` on stage/workflow transitions |
| `workflow/app.py` | `_handle_list_runs()` reads `status.json`; `_handle_log()` returns actual log content |
| `workflow/web/js/app.js` | History tab UI: records with stage progress, view-log button, log modal |
| `workflow/web/css/style.css` | Modal and history record styles |
| `tests/test_workflow_app.py` | Integration tests for status.json and log API |

---

### Task 1: Add `_write_status` helper to scheduler.py

**Files:**
- Modify: `workflow/scheduler.py`

- [ ] **Step 1: Add the helper method and integrate into scheduler.run()**

Add to `WorkflowScheduler` class, after `_make_executor` (after line 92):

```python
    def _write_status(self, run_dir: Path, status: str, current_stage: str = "",
                      stages: dict | None = None) -> None:
        import json
        data = {
            "status": status,
            "current_stage": current_stage,
            "stages": stages or {},
            "updated_at": datetime.now().isoformat(),
        }
        (run_dir / "status.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

Now integrate into the `run()` method. After `logger.workflow_start(len(ordered))` (line 104), add initialization:

```python
        run_dir = log_file.parent
        stage_status: dict[str, str] = {}
        self._write_status(run_dir, "running", "", {})
```

After `logger.stage_start(stage.id, stage.type)` (line 121), add:

```python
                stage_status[stage.id] = "running"
                self._write_status(run_dir, "running", stage.id, stage_status)
```

In the success branch after `logger.stage_end(stage.id, "ok")` (line 147), add:

```python
                    stage_status[stage.id] = "ok"
                    self._write_status(run_dir, "running", stage.id, stage_status)
```

In the error branch after `logger.stage_end(stage.id, ...)` (line 150), add:

```python
                    stage_status[stage.id] = "error"
                    self._write_status(run_dir, "error", stage.id, stage_status)
```

In the stop_flag branch after `logger.stage_end(stage.id, "stopped")` (line 108), add:

```python
                stage_status[stage.id] = "stopped"
                self._write_status(run_dir, "stopped", stage.id, stage_status)
```

Before `logger.workflow_end(status)` (line 155), add:

```python
        self._write_status(run_dir, status, "", stage_status)
```

- [ ] **Step 2: Verify syntax**

Run: `.venv\Scripts\python.exe -c "from workflow.scheduler import WorkflowScheduler; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add workflow/scheduler.py
git commit -m "feat: scheduler writes status.json at stage transitions"
```

---

### Task 2: Update _handle_list_runs to read status.json

**Files:**
- Modify: `workflow/app.py` (replace `_handle_list_runs` body)

- [ ] **Step 1: Replace the handler**

Replace the entire `_handle_list_runs` function (lines 148-185) with:

```python
async def _handle_list_runs(req: web.Request) -> web.Response:
    import json as _json
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    runs_dir = root / name / "runs"
    if not runs_dir.exists():
        return web.json_response([])
    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.name == "latest" or not d.is_dir():
            continue
        status_file = d / "status.json"
        if status_file.exists():
            try:
                sd = _json.loads(status_file.read_text(encoding="utf-8"))
                status = sd.get("status", "unknown")
                current_stage = sd.get("current_stage", "")
                stages = sd.get("stages", {})
            except Exception:
                status = "unknown"
                current_stage = ""
                stages = {}
        else:
            status = "unknown"
            current_stage = ""
            stages = {}
        from datetime import datetime as _dt
        created = _dt.fromtimestamp(d.stat().st_ctime).isoformat()
        runs.append({
            "id": d.name,
            "status": status,
            "current_stage": current_stage,
            "stages": stages,
            "created_at": created,
        })
    return web.json_response(runs)
```

- [ ] **Step 2: Verify syntax**

Run: `.venv\Scripts\python.exe -c "from workflow.app import create_app; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add workflow/app.py
git commit -m "feat: _handle_list_runs reads status.json"
```

---

### Task 3: Fix _handle_log to return actual log content

**Files:**
- Modify: `workflow/app.py` (replace `_handle_log` body)

- [ ] **Step 1: Replace the handler**

Replace the `_handle_log` function (lines 289-291) with:

```python
async def _handle_log(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    run_id = req.match_info["run_id"]
    root = req.app["workflows_root"]
    log_file = root / name / "runs" / run_id / "run.log"
    lines = []
    if log_file.exists():
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
        except Exception:
            lines = [f"Error reading log file: {log_file}"]
    return web.json_response({"lines": lines})
```

Also add the route for this handler — the route `GET /api/runs/{run_id}/log` already exists (line 49), but it doesn't include `{name}`. We need a workflow-scoped version. Add after the existing route:

```python
    app.router.add_get("/api/workflows/{name}/runs/{run_id}/log", _handle_run_log)
```

And rename `_handle_log` to `_handle_run_log` or add a new handler. The simplest approach: keep the existing `_handle_log` for backward compatibility and add a new handler. Actually, let's just update the existing one to accept both path patterns. The simplest is to add a new route with name parameter.

Add the new route after line 51:

```python
    app.router.add_get("/api/workflows/{name}/runs/{run_id}/log", _handle_workflow_run_log)
```

Add the handler before `start_server`:

```python
async def _handle_workflow_run_log(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    run_id = req.match_info["run_id"]
    root = req.app["workflows_root"]
    log_file = root / name / "runs" / run_id / "run.log"
    lines = []
    if log_file.exists():
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
        except Exception:
            lines = [f"Error reading log file"]
    return web.json_response({"lines": lines})
```

- [ ] **Step 2: Verify syntax**

Run: `.venv\Scripts\python.exe -c "from workflow.app import create_app; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add workflow/app.py
git commit -m "feat: add workflow-scoped run log API endpoint"
```

---

### Task 4: Add API method and update frontend history tab

**Files:**
- Modify: `workflow/web/js/api.js` (add `getRunLog` with name)
- Modify: `workflow/web/js/app.js` (history tab UI overhaul)

- [ ] **Step 1: Add API method**

In `workflow/web/js/api.js`, add to the returned object (after `openRunDir`):

```javascript
    getWorkflowRunLog: function(name, runId) {
      return get("/api/workflows/" + encodeURIComponent(name) + "/runs/" + encodeURIComponent(runId) + "/log");
    },
```

- [ ] **Step 2: Update history tab in app.js**

In `workflow/web/js/app.js`, find the history tab template section (the `v-if="activeTab === 'history'"` block). Replace the history rendering with:

1. Add new reactive data: `logModalRun`, `logModalLines`, `logModalLoading`
2. Replace history tab template with records showing stage progress and view-log button
3. Add log modal template

Add these refs in the `setup()` function (after `runHistory`):

```javascript
    var logModalRun = ref(null);
    var logModalLines = ref([]);
    var logModalLoading = ref(false);

    var viewRunLog = function(run) {
      logModalRun.value = run;
      logModalLoading.value = true;
      logModalLines.value = [];
      AnimaAPI.getWorkflowRunLog(workflowName.value, run.id).then(function(data) {
        logModalLines.value = data.lines || [];
        logModalLoading.value = false;
      }).catch(function() {
        logModalLines.value = ["Failed to load log"];
        logModalLoading.value = false;
      });
    };

    var closeLogModal = function() {
      logModalRun.value = null;
      logModalLines.value = [];
    };
```

Return them from setup.

Replace the history tab template. Find the `v-if="activeTab === 'history'"` section and replace its content with a records list:

```html
<div v-if="activeTab === 'history'" class="history-tab">
  <div v-if="runHistory.length === 0" class="empty-state">暂无运行历史</div>
  <div v-for="run in runHistory" :key="run.id" class="history-record">
    <span class="history-time">{{ run.created_at ? run.created_at.substring(0, 16).replace('T', ' ') : run.id }}</span>
    <span class="history-status" :class="'status-' + run.status">
      <template v-if="run.status === 'ok'">✅</template>
      <template v-else-if="run.status === 'running'">🔄</template>
      <template v-else-if="run.status === 'stopped'">⏹</template>
      <template v-else-if="run.status === 'error'">❌</template>
      <template v-else>❓</template>
      {{ run.status }}
    </span>
    <span class="history-stages">
      <template v-for="(sname, idx) in Object.keys(run.stages || {})" :key="sname">
        <template v-if="idx > 0"> → </template>
        <span :class="'stage-' + (run.stages[sname] || 'unknown')">{{ sname }}</span>
      </template>
    </span>
    <span class="history-actions">
      <button class="btn btn-ghost btn-xs" @click="viewRunLog(run)">查看日志</button>
      <button class="btn btn-ghost btn-xs" @click="AnimaAPI.openRunDir(workflowName, run.id)">打开目录</button>
    </span>
  </div>
</div>
```

Add log modal at the end of the root template (before the closing `</div>`):

```html
<div v-if="logModalRun" class="modal-overlay" @click.self="closeLogModal">
  <div class="modal-content modal-log">
    <div class="modal-header">
      <span>运行日志 — {{ logModalRun.id }}</span>
      <button class="modal-close" @click="closeLogModal">✕</button>
    </div>
    <div class="modal-body">
      <div v-if="logModalLoading" class="empty-state">加载中...</div>
      <pre v-else class="log-content">{{ logModalLines.join("\n") }}</pre>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Verify JS syntax**

Run: `node -c workflow/web/js/api.js` and `node -c workflow/web/js/app.js`
Expected: No syntax errors

- [ ] **Step 4: Commit**

```bash
git add workflow/web/js/api.js workflow/web/js/app.js
git commit -m "feat: history tab with stage progress and log viewer modal"
```

---

### Task 5: Add CSS styles for history records and log modal

**Files:**
- Modify: `workflow/web/css/style.css`

- [ ] **Step 1: Append styles**

Append to `workflow/web/css/style.css`:

```css
.history-tab {
  padding: 4px 0;
}

.history-record {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}

.history-record:last-child {
  border-bottom: none;
}

.history-time {
  color: var(--text-dim);
  white-space: nowrap;
  min-width: 130px;
}

.history-status {
  font-size: 11px;
  min-width: 60px;
}

.history-stages {
  flex: 1;
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.stage-ok { color: #4caf50; }
.stage-running { color: #2196f3; }
.stage-error { color: #f44336; }
.stage-stopped { color: #ff9800; }
.stage-unknown { color: var(--text-dim); }

.history-actions {
  display: flex;
  gap: 4px;
  white-space: nowrap;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
}

.modal-log {
  width: 80%;
  max-width: 900px;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 500;
}

.modal-close {
  background: none;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 16px;
  padding: 0 4px;
}

.modal-close:hover {
  color: var(--text);
}

.modal-body {
  padding: 10px 14px;
  overflow-y: auto;
  flex: 1;
}

.log-content {
  font-family: monospace;
  font-size: 11px;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 0;
  color: var(--text);
}
```

Also add `.btn-xs` style if it doesn't exist:

```css
.btn-xs {
  font-size: 10px;
  padding: 2px 6px;
}
```

- [ ] **Step 2: Commit**

```bash
git add workflow/web/css/style.css
git commit -m "feat: add history record and log modal CSS styles"
```

---

### Task 6: Run tests and verify

- [ ] **Step 1: Run backend tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_workflow_app.py tests/test_workflow_scheduler.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `.venv\Scripts\python.exe -m ruff check workflow/scheduler.py workflow/app.py --fix`
Expected: No new errors

- [ ] **Step 3: Manual browser test**

Start workflow server, open a workflow with run history, verify:
- History tab shows records with status icons and stage progress
- "查看日志" button opens modal with log content
- Running a workflow creates status.json and shows correct stage progress
