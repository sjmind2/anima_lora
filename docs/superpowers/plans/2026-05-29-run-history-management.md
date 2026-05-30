# Run History Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add run history management — timestamped directories, history list UI, open-in-explorer button.

**Architecture:** Backend creates timestamped run dirs + `latest` junction. New REST endpoints list runs and open directories. Frontend adds a third tab in the bottom panel.

**Tech Stack:** Python aiohttp, Vue 3, `os.startfile`/`subprocess` for file manager.

---

### Task 1: Backend — Fix run directory creation + latest junction

**Files:**
- Modify: `workflow/app.py` (line 179, `_handle_run`)
- Modify: `workflow/scheduler.py` (after `run()` method)

- [ ] **Step 1: Fix `_handle_run` to use timestamped directories**

In `app.py`, change line 179 from:
```python
scheduler.run(log_file=wf_dir / "runs" / "latest" / "run.log")
```
to:
```python
scheduler.run()
```

- [ ] **Step 2: Add `latest` junction creation in `scheduler.run()`**

After `logger.workflow_end(status)` and before `return all_success`, add:
```python
        latest = runs_dir / "latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_dir() and not latest.is_symlink():
                import shutil
                shutil.rmtree(latest)
            else:
                latest.unlink()
        if sys.platform == "win32":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(latest), str(log_file.parent)], check=True, capture_output=True)
        else:
            os.symlink(str(log_file.parent), str(latest))
```

Add imports at top of scheduler.py: `import os, subprocess, sys`.

---

### Task 2: Backend — Add run history API endpoints

**Files:**
- Modify: `workflow/app.py`

- [ ] **Step 1: Add `GET /api/workflows/{name}/runs` handler**

```python
async def _handle_list_runs(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    runs_dir = root / name / "runs"
    if not runs_dir.exists():
        return web.json_response([])
    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.name == "latest" or not d.is_dir():
            continue
        log_file = d / "run.log"
        status = "unknown"
        stages = []
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                if "workflow_end" in line:
                    if "status=ok" in line or '"ok"' in line:
                        status = "ok"
                    elif "stopped" in line:
                        status = "stopped"
                    else:
                        status = "error"
                if "stage_end" in line:
                    sid = ""
                    sstatus = "ok"
                    for part in line.split():
                        if part.startswith("stage_id="):
                            sid = part.split("=", 1)[1].strip('"').strip(",")
                        if "error" in part.lower() or "stopped" in part.lower():
                            sstatus = "error" if "error" in part.lower() else "stopped"
                    if sid:
                        stages.append({"id": sid, "status": sstatus})
        stat = d.stat()
        from datetime import datetime
        created = datetime.fromtimestamp(stat.st_ctime).isoformat()
        runs.append({"id": d.name, "status": status, "stages": stages, "created_at": created})
    return web.json_response(runs)
```

- [ ] **Step 2: Add `POST /api/workflows/{name}/runs/{run_id}/open` handler**

```python
async def _handle_open_run(req: web.Request) -> web.Response:
    import platform
    name = req.match_info["name"]
    run_id = req.match_info["run_id"]
    root = req.app["workflows_root"]
    run_dir = root / name / "runs" / run_id
    if not run_dir.exists():
        return web.json_response({"error": "not found"}, status=404)
    path = str(run_dir.resolve())
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
    return web.json_response({"status": "opened"})
```

- [ ] **Step 3: Register new routes in `create_app`**

After existing route registrations, add:
```python
    app.router.add_get("/api/workflows/{name}/runs", _handle_list_runs)
    app.router.add_post("/api/workflows/{name}/runs/{run_id}/open", _handle_open_run)
```

---

### Task 3: Frontend — Add run history tab + API calls

**Files:**
- Modify: `workflow/web/js/api.js`
- Modify: `workflow/web/js/app.js`
- Modify: `workflow/web/css/style.css`

- [ ] **Step 1: Add API functions in `api.js`**

After existing functions, add:
```js
  function listRuns(name) {
    return fetch(BASE + "/api/workflows/" + encodeURIComponent(name) + "/runs")
      .then(function (r) { return r.json(); });
  }

  function openRunDir(name, runId) {
    return fetch(BASE + "/api/workflows/" + encodeURIComponent(name) + "/runs/" + encodeURIComponent(runId) + "/open", { method: "POST" })
      .then(function (r) { return r.json(); });
  }
```

And export them in the return object.

- [ ] **Step 2: Add reactive state and methods in `app.js`**

Add after `activeLogTab`:
```js
      var runHistory = ref([]);
```

Add `loadRunHistory` function:
```js
      function loadRunHistory() {
        if (!workflowName.value) return;
        AnimaAPI.listRuns(workflowName.value)
          .then(function(runs) { runHistory.value = runs; })
          .catch(function() { runHistory.value = []; });
      }
```

Add `openRunDir` method:
```js
      function openRunDir(runId) {
        if (!workflowName.value) return;
        AnimaAPI.openRunDir(workflowName.value, runId)
          .catch(function(err) { showToast("打开失败: " + err, "error"); });
      }
```

Expose in return: `runHistory: runHistory, loadRunHistory: loadRunHistory, openRunDir: openRunDir`.

Call `loadRunHistory()` when switching to history tab and when `workflow_end` fires.

- [ ] **Step 3: Add third tab button in template**

Change the log-tab-bar to include three tabs:
```js
      '      <div class="log-tab-bar">',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'system\' }" @click="activeLogTab = \'system\'">系统日志</button>',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'script\' }" @click="activeLogTab = \'script\'">脚本输出 <span v-if="scriptLogs.length" style="opacity:0.6;">({{ scriptLogs.length }})</span></button>',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'history\' }" @click="activeLogTab = \'history\'; loadRunHistory()">运行历史</button>',
      '      </div>',
```

- [ ] **Step 4: Add history tab content in template**

After the `</template>` for script output, add:
```js
      '      <template v-if="activeLogTab === \'history\'">',
      '        <div v-if="runHistory.length === 0" style="color:var(--text-dim);font-style:italic;padding:8px 0;">暂无运行记录</div>',
      '        <div v-for="run in runHistory" :key="run.id" class="run-history-row">',
      '          <span class="run-status-icon">{{ run.status === \'ok\' ? \'✅\' : run.status === \'stopped\' ? \'⏹\' : run.status === \'error\' ? \'❌\' : \'❓\' }}</span>',
      '          <span class="run-time">{{ run.created_at ? run.created_at.replace(\'T\', \' \').substring(0, 16) : run.id }}</span>',
      '          <span class="run-stage-summary">{{ run.stages.length }} 阶段</span>',
      '          <button class="btn btn-ghost btn-sm" @click="openRunDir(run.id)" title="在文件管理器中打开">📂</button>',
      '        </div>',
      '      </template>',
```

- [ ] **Step 5: Add CSS for run history rows**

```css
.run-history-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  font-size: 12px;
  border-bottom: 1px solid var(--border-light);
}

.run-status-icon {
  font-size: 14px;
}

.run-time {
  color: var(--text);
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
}

.run-stage-summary {
  color: var(--text-dim);
  flex: 1;
}
```
