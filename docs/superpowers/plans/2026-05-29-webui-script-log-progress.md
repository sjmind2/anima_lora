# WebUI Script Log & Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push subprocess stdout to the WebUI in real-time with tqdm-aware progress rendering — dual progress bars, dual log tabs, tqdm overwrite semantics.

**Architecture:** Backend buffers stdout lines per stage and flushes them as batched SSE events every 0.3s. Frontend receives batches, classifies lines as tqdm or normal, updates a reactive progress object for tqdm lines (overwrite), and appends normal lines to a capped array.

**Tech Stack:** Python aiohttp SSE (existing), Vue 3 reactive (existing), vanilla JS tqdm regex parser.

---

### Task 1: Backend — Add `stage_stdout_batch` to WorkflowLogger

**Files:**
- Modify: `workflow/logger.py`

- [ ] **Step 1: Add `stage_stdout_batch` method to `WorkflowLogger`**

Add after the existing `info` method (line 64):

```python
    def stage_stdout_batch(self, stage_id: str, lines: list[str]) -> None:
        for line in lines:
            self._log(stage_id, "STDOUT", line)
        self._emit({
            "ev": "stage_stdout_batch",
            "stage_id": stage_id,
            "lines": lines,
        })
```

- [ ] **Step 2: Verify no import changes needed**

`threading`, `time`, `datetime`, `Path`, `Any` are already imported. No new imports needed.

---

### Task 2: Backend — Replace `on_stdout` with buffered flush in Scheduler

**Files:**
- Modify: `workflow/scheduler.py`

- [ ] **Step 1: Add `_StdoutBuffer` class before `WorkflowScheduler`**

Insert at line 14 (after imports, before class):

```python
class _StdoutBuffer:
    def __init__(self, logger: WorkflowLogger, flush_interval: float = 0.3) -> None:
        self._logger = logger
        self._buf: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._timer: threading.Timer | None = None

    def add(self, stage_id: str, line: str) -> None:
        with self._lock:
            self._buf.setdefault(stage_id, []).append(line)

    def start(self) -> None:
        self._schedule_next()

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            snapshot = {k: v[:] for k, v in self._buf.items()}
            self._buf.clear()
        for sid, lines in snapshot.items():
            self._logger.stage_stdout_batch(sid, lines)
        if not self._stopped:
            self._schedule_next()

    def stop(self) -> None:
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self._flush()

    _stopped: bool = False
```

Note: `_stopped` is set before use in `stop()` which is called after `start()`. The initial value `False` on the class body works as default.

Actually, fix this — set `_stopped = True` default and flip in `start`:

```python
class _StdoutBuffer:
    def __init__(self, logger: WorkflowLogger, flush_interval: float = 0.3) -> None:
        self._logger = logger
        self._buf: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._timer: threading.Timer | None = None
        self._stopped = True

    def add(self, stage_id: str, line: str) -> None:
        with self._lock:
            self._buf.setdefault(stage_id, []).append(line)

    def start(self) -> None:
        self._stopped = False
        self._schedule_next()

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            snapshot = {k: v[:] for k, v in self._buf.items()}
            self._buf.clear()
        for sid, lines in snapshot.items():
            self._logger.stage_stdout_batch(sid, lines)
        if not self._stopped:
            self._schedule_next()

    def stop(self) -> None:
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self._flush()
```

- [ ] **Step 2: Modify `run()` to use `_StdoutBuffer`**

Replace the `run()` method body (lines 53-93). The change is:
- Create buffer before the stage loop, start it
- Change `on_stdout` lambda to call `buffer.add` instead of `logger.info`
- Stop buffer after the loop

Replace lines 53-93 with:

```python
    def run(self, log_file: Path | None = None) -> bool:
        log_file = log_file or (self._create_run_dir() / "run.log")
        logger = WorkflowLogger(log_file, self.event_queue)
        ordered = self.wf.topological_order()
        stage_outputs: dict[str, dict[str, str]] = {}
        all_success = True

        buffer = _StdoutBuffer(logger, flush_interval=0.3)
        buffer.start()

        logger.workflow_start(len(ordered))

        for stage in ordered:
            if self._stop_flag.is_set():
                logger.stage_end(stage.id, "stopped")
                all_success = False
                break

            try:
                resolved = self._resolve_and_write_config(stage.id, log_file.parent, stage_outputs)
            except Exception as e:
                logger.stage_end(stage.id, f"config_error: {e}")
                all_success = False
                break

            executor = self._make_executor(stage, resolved, log_file.parent)
            logger.stage_start(stage.id, stage.type)

            def on_stdout(sid: str, line: str) -> None:
                buffer.add(sid, line)

            result = executor.execute(on_stdout=on_stdout)

            if result.success:
                stage_outputs[stage.id] = result.outputs
                logger.stage_end(stage.id, "ok")
            else:
                all_success = False
                logger.stage_end(stage.id, f"error: {result.error}")
                break

        buffer.stop()
        status = "ok" if all_success else "error"
        logger.workflow_end(status)
        return all_success
```

---

### Task 3: Frontend — Add tqdm parser and new reactive state

**Files:**
- Modify: `workflow/web/js/app.js`

- [ ] **Step 1: Add new reactive state variables**

After line 33 (`var completedStages = ref(0);`), add:

```js
      var scriptLogs = ref([]);
      var scriptProgress = reactive({
        pct: 0, current: 0, total: 0,
        elapsed: "", eta: "", rate: "",
        metrics: {}, rawLine: "", active: false,
      });
      var activeLogTab = ref("system");
```

- [ ] **Step 2: Add tqdm parser function**

After the `generateId` function (line 74), add:

```js
      var TQDM_RE = /^(\S+):\s+(\d+)%\|[^|]*\|\s+(\d+)\/(\d+)\s+\[([^\]]+)\](?:\s+(.+))?/;
      var TQDM_METRIC_RE = /(\w[\w_]*)=([^\s,]+)/g;

      function parseTqdmLine(line) {
        var m = line.match(TQDM_RE);
        if (!m) return null;
        var result = {
          prefix: m[1], pct: parseInt(m[2]),
          current: parseInt(m[3]), total: parseInt(m[4]),
          timing: m[5], extra: m[6] || ""
        };
        var tm = result.timing.match(/^([^<]+)<([^,]+),\s*(.+)$/);
        if (tm) {
          result.elapsed = tm[1].trim();
          result.eta = tm[2].trim();
          result.rate = tm[3].trim();
        }
        result.metrics = {};
        TQDM_METRIC_RE.lastIndex = 0;
        var mm;
        while ((mm = TQDM_METRIC_RE.exec(result.extra)) !== null) {
          result.metrics[mm[1]] = mm[2];
        }
        return result;
      }
```

- [ ] **Step 3: Add `stage_stdout_batch` handler in `handleEvent`**

In the `handleEvent` switch (line 242), add a new case before `case "stream_error"`:

```js
          case "stage_stdout_batch":
            ev.lines.forEach(function(line) {
              var parsed = parseTqdmLine(line);
              if (parsed) {
                Object.assign(scriptProgress, {
                  pct: parsed.pct,
                  current: parsed.current,
                  total: parsed.total,
                  elapsed: parsed.elapsed,
                  eta: parsed.eta,
                  rate: parsed.rate,
                  metrics: parsed.metrics,
                  rawLine: line,
                  active: true,
                });
              } else {
                scriptLogs.value.push({
                  text: line,
                  ts: new Date().toLocaleTimeString(),
                  stage_id: ev.stage_id,
                });
                if (scriptLogs.value.length > 500) {
                  scriptLogs.value.splice(0, scriptLogs.value.length - 500);
                }
              }
            });
            break;
```

- [ ] **Step 4: Add stage cleanup in `stage_start` handler**

In the `case "stage_start":` block (line 249), after the existing `addLog` call, add:

```js
            scriptProgress.active = false;
            scriptProgress.rawLine = "";
            scriptProgress.pct = 0;
            scriptLogs.value = [];
```

- [ ] **Step 5: Expose new state in the return object**

In the return object (line 326), add these properties:

```js
        scriptLogs: scriptLogs,
        scriptProgress: scriptProgress,
        activeLogTab: activeLogTab,
```

---

### Task 4: Frontend — Update bottom panel template

**Files:**
- Modify: `workflow/web/js/app.js` (template section, lines 466-483)

- [ ] **Step 1: Replace the bottom panel template**

Replace lines 466-483 (the entire `<div v-if="workflowName" class="bottom-panel">` block) with:

```js
      '  <div v-if="workflowName" class="bottom-panel">',
      '    <div class="bottom-panel-header">',
      '      <span class="bottom-panel-title">日志</span>',
      '      <div class="log-tab-bar">',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'system\' }" @click="activeLogTab = \'system\'">系统日志</button>',
      '        <button class="log-tab-btn" :class="{ active: activeLogTab === \'script\' }" @click="activeLogTab = \'script\'">脚本输出 <span v-if="scriptLogs.length" style="opacity:0.6;">({{ scriptLogs.length }})</span></button>',
      '      </div>',
      '      <span style="font-size:11px;color:var(--text-dim);">{{ completedStages }}/{{ totalStages }} 阶段</span>',
      '    </div>',
      '    <div v-if="isRunning || completedStages > 0" class="progress-bar-container">',
      '      <div class="progress-bar-track">',
      '        <div class="progress-bar-fill stage-progress" :class="progressStatus" :style="{ width: overallProgress + \'%\' }"></div>',
      '      </div>',
      '      <div class="progress-label">阶段 {{ overallProgress }}%</div>',
      '    </div>',
      '    <div v-if="scriptProgress.active" class="script-progress-section">',
      '      <div class="progress-bar-track">',
      '        <div class="progress-bar-fill script-progress" :style="{ width: scriptProgress.pct + \'%\' }"></div>',
      '      </div>',
      '      <div class="script-status-line">',
      '        <span>{{ scriptProgress.current }}/{{ scriptProgress.total }}</span>',
      '        <span v-if="scriptProgress.eta">[{{ scriptProgress.elapsed }}&lt;{{ scriptProgress.eta }}, {{ scriptProgress.rate }}]</span>',
      '        <span v-for="(v, k) in scriptProgress.metrics" :key="k" class="metric-badge">{{ k }}={{ v }}</span>',
      '      </div>',
      '    </div>',
      '    <div class="log-viewer">',
      '      <template v-if="activeLogTab === \'system\'">',
      '        <div v-for="(line, i) in logLines" :key="i" class="log-line" :class="line.cls">',
      '          [{{ line.ts }}] {{ line.text }}',
      '        </div>',
      '        <div v-if="logLines.length === 0" style="color:var(--text-dim);font-style:italic;">等待运行...</div>',
      '      </template>',
      '      <template v-else>',
      '        <div v-for="(line, i) in scriptLogs" :key="\'s\' + i" class="log-line script">',
      '          <span v-if="line.stage_id" class="log-stage-tag">[{{ line.stage_id }}]</span>',
      '          <span v-if="line.ts" class="log-ts-tag">[{{ line.ts }}]</span>',
      '          {{ line.text }}',
      '        </div>',
      '        <div v-if="scriptLogs.length === 0 && !scriptProgress.active" style="color:var(--text-dim);font-style:italic;">暂无脚本输出</div>',
      '        <div v-if="scriptLogs.length === 0 && scriptProgress.active" class="log-line script" style="color:var(--text-dim);">{{ scriptProgress.rawLine }}</div>',
      '      </template>',
      '    </div>',
      '  </div>',
```

---

### Task 5: Frontend — Add CSS styles for new components

**Files:**
- Modify: `workflow/web/css/style.css`

- [ ] **Step 1: Add log tab bar styles**

After the existing `.progress-label` block (line 451), add:

```css
.log-tab-bar {
  display: flex;
  gap: 2px;
}

.log-tab-btn {
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 12px;
  padding: 2px 8px;
  cursor: pointer;
  border-radius: 3px;
  transition: background 0.15s, color 0.15s;
}

.log-tab-btn:hover {
  background: var(--bg-input);
  color: var(--text);
}

.log-tab-btn.active {
  background: var(--bg-input);
  color: var(--text);
  font-weight: 500;
}

.script-progress-section {
  padding: 0 12px 6px;
}

.progress-bar-fill.script-progress {
  background: linear-gradient(90deg, var(--blue), var(--accent));
}

.script-status-line {
  display: flex;
  gap: 8px;
  font-size: 11px;
  color: var(--text-dim);
  font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
  margin-top: 2px;
  align-items: center;
  flex-wrap: wrap;
}

.metric-badge {
  background: var(--bg-input);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 10px;
  color: var(--accent);
}

.log-line.script {
  color: var(--text);
  opacity: 0.85;
}

.log-stage-tag {
  color: var(--blue);
  margin-right: 4px;
}

.log-ts-tag {
  color: var(--text-dim);
  margin-right: 4px;
}
```

---

### Task 6: Verify — Restart server and test with running workflow

**Files:** None (verification only)

- [ ] **Step 1: Stop the running workflow server**

Kill the existing server process on port 8766.

- [ ] **Step 2: Restart the server**

Run: `python -m workflow --no-gui --port 8766 --workflows-root O:\loratool\anima_lora_fork\workflows`

- [ ] **Step 3: Navigate browser to the workflow UI**

Open `http://localhost:8766/`, load `hanechan-lokr-webui`, click run.

- [ ] **Step 4: Verify dual progress bars and dual log tabs**

Expected behavior:
- Stage progress bar shows 25% after preprocess_s1 completes
- Script progress bar appears during training with parsed tqdm data
- Switching to "脚本输出" tab shows non-tqdm script output
- tqdm lines update the progress bar and status line, not the log array
- Status line shows `95/1224 [02:35<30:44] 1.63s/it` + metric badges
