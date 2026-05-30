# Run History Management Design

## Problem

1. `_handle_run` hardcodes `log_file=wf_dir / "runs" / "latest" / "run.log"`, overwriting previous runs
2. No UI to view past run history
3. No way to open run directories from the WebUI

## Design

### Backend

#### Fix run directory creation

In `app.py` `_handle_run`: remove the hardcoded `log_file` parameter, let `scheduler.run()` use its default `_create_run_dir()` which creates `runs/YYYYMMDD-HHMMSS/` timestamped directories. After run completes, update `runs/latest` junction/symlink to point to the new directory.

#### New API: `GET /api/workflows/{name}/runs`

List all runs for a workflow. Scans `{wf_dir}/runs/` for timestamped directories, parses each `run.log` to extract status.

Response:
```json
[
  {
    "id": "20260529-150401",
    "status": "ok",
    "stages": [
      {"id": "preprocess_1", "status": "ok"},
      {"id": "train_1", "status": "ok"}
    ],
    "created_at": "2026-05-29T15:04:01"
  }
]
```

Status is inferred from run.log: if it contains `workflow_end` with `status=ok` → "ok", if `status=error` → "error", if `stopped` → "stopped". If log doesn't have `workflow_end` and the run is not active → "interrupted".

#### New API: `POST /api/workflows/{name}/runs/{run_id}/open`

Open the run directory in the system file manager. Uses `os.startfile()` on Windows, `subprocess.Popen(['open', path])` on macOS, `subprocess.Popen(['xdg-open', path])` on Linux.

#### New API: `GET /api/workflows/{name}/runs/{run_id}/log`

Read and return the run.log content (already exists as `_handle_log` but needs to support timestamped paths).

### Frontend

Add "运行历史" as a third tab in the bottom panel alongside "系统日志" and "脚本输出".

Tab content shows a list of past runs:
- Each row: timestamp, status icon (✅/❌/⏹/🔄), stage summary, [📂 Open] button
- [📂 Open] calls `POST /api/.../open` to open in file manager
- Current active run (if any) shown at top with "🔄 运行中" badge
- Auto-refresh list when a run completes (listen to `workflow_end` SSE event)

### Files Changed

| File | Change |
|------|--------|
| `workflow/app.py` | Fix `_handle_run` to use timestamped dirs; add `GET /runs`, `POST /runs/{id}/open` handlers |
| `workflow/scheduler.py` | After run, create/update `runs/latest` junction |
| `workflow/web/js/app.js` | Add "运行历史" tab, run history list, open-in-explorer button |
| `workflow/web/css/style.css` | Styles for run history rows |

### Constraints

- `runs/latest` is a convenience pointer, not the source of truth
- Run log parsing is simple line-scanning, no complex parsing
- `os.startfile` is Windows-only; fallback to `open`/`xdg-open` on other platforms
- Run history list is fetched on demand (tab switch), not polled
