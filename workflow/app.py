from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

from aiohttp import web

from workflow.i18n import set_locale, get_locale
from workflow.config import (
    load_workflow_yaml,
    save_workflow_yaml,
    load_stage_toml,
    save_stage_toml,
    load_schema,
)
from workflow.logger import EventQueue
from workflow.models import WorkflowDefinition
from workflow.scheduler import WorkflowScheduler


@web.middleware
async def _locale_middleware(req, handler):
    accept = req.headers.get("Accept-Language", "")
    if accept.startswith("zh"):
        set_locale("zh-CN")
    elif accept.startswith("ja"):
        set_locale("ja")
    else:
        set_locale("en")
    return await handler(req)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _config_path() -> Path:
    return _project_root() / ".anima_workflow_config.json"


def _load_config() -> dict:
    cp = _config_path()
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(data: dict) -> None:
    cp = _config_path()
    cp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_workflows_root() -> Path:
    cfg = _load_config()
    custom = cfg.get("workflows_root")
    if custom:
        return Path(custom)
    return _project_root() / ".anima_workflow"


def create_app(workflows_root: Path | str | None = None) -> web.Application:
    if workflows_root is None:
        workflows_root = _resolve_workflows_root()
    workflows_root = Path(workflows_root)
    workflows_root.mkdir(parents=True, exist_ok=True)

    app = web.Application(middlewares=[_locale_middleware])
    app["workflows_root"] = workflows_root
    app["event_queues"]: list[EventQueue] = []
    app["scheduler_thread"] = None
    app["active_scheduler"] = None

    app.router.add_get("/api/workflows", _handle_list_workflows)
    app.router.add_post("/api/workflows", _handle_create_workflow)
    app.router.add_get("/api/workflows/{name}", _handle_get_workflow)
    app.router.add_put("/api/workflows/{name}", _handle_update_workflow)
    app.router.add_delete("/api/workflows/{name}/runs", _handle_clear_runs)
    app.router.add_get("/api/workflows/{name}/runs", _handle_list_runs)
    app.router.add_post(
        "/api/workflows/{name}/runs/{run_id}/open", _handle_open_run_dir
    )
    app.router.add_get(
        "/api/workflows/{name}/runs/{run_id}/log", _handle_workflow_run_log
    )
    app.router.add_get("/api/workflows/{name}/infrastructure", _handle_get_infra)
    app.router.add_put("/api/workflows/{name}/infrastructure", _handle_set_infra)
    app.router.add_post("/api/workflows/{name}/run", _handle_run)
    app.router.add_post("/api/workflows/{name}/stop", _handle_stop)
    app.router.add_get("/api/runs/{run_id}/events", _handle_events)
    app.router.add_get("/api/runs/{run_id}/log", _handle_log)
    app.router.add_get("/api/schemas/{schema_name}", _handle_get_schema)
    app.router.add_get("/api/recent-workflows", _handle_recent)
    app.router.add_post("/api/dataset/bucket-stats", _handle_bucket_stats)
    app.router.add_get("/api/settings", _handle_get_settings)
    app.router.add_put("/api/settings", _handle_set_settings)
    app.router.add_post("/api/browse", _handle_browse)

    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.router.add_get("/", _handle_index)
        app.router.add_static("/static", str(web_dir))

    i18n_dir = Path(__file__).parent / "i18n"
    if i18n_dir.exists():
        app.router.add_static("/static/i18n", str(i18n_dir))

    return app


async def _handle_index(req: web.Request) -> web.Response:
    web_dir = Path(__file__).parent / "web"
    index_file = web_dir / "index.html"
    if not index_file.exists():
        return web.json_response({"error": "not found", "path": "/"}, status=404)
    return web.FileResponse(index_file)


async def _handle_list_workflows(req: web.Request) -> web.Response:
    root = req.app["workflows_root"]
    workflows = []
    for d in sorted(root.iterdir()):
        wf_file = d / "workflow.yaml"
        if d.is_dir() and wf_file.exists():
            data = load_workflow_yaml(wf_file)
            data["dir"] = d.name
            workflows.append(data)
    return web.json_response(workflows)


async def _handle_create_workflow(req: web.Request) -> web.Response:
    body = await req.json()
    name = body.get("name", "untitled")
    root = req.app["workflows_root"]
    wf_dir = root / name
    wf_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = wf_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    wf_data = {"name": name, "stages": [], "infrastructure": {}}
    save_workflow_yaml(wf_data, wf_dir / "workflow.yaml")
    return web.json_response(wf_data, status=201)


async def _handle_get_workflow(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    wf_file = root / name / "workflow.yaml"
    if not wf_file.exists():
        return web.json_response({"error": "not found"}, status=404)
    data = load_workflow_yaml(wf_file)
    stage_configs = {}
    configs_dir = root / name / "configs"
    for stage in data.get("stages", []):
        sid = stage.get("id")
        cf = stage.get("config_file")
        if sid and cf:
            toml_path = configs_dir / cf
            if toml_path.exists():
                try:
                    stage_configs[sid] = load_stage_toml(toml_path)
                except Exception:
                    stage_configs[sid] = {}
            else:
                stage_configs[sid] = {}
    data["stage_configs"] = stage_configs
    return web.json_response(data)


async def _handle_update_workflow(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    wf_file = root / name / "workflow.yaml"
    body = await req.json()
    stage_configs = body.pop("stage_configs", None)
    save_workflow_yaml(body, wf_file)
    if stage_configs and isinstance(stage_configs, dict):
        configs_dir = root / name / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        stages = body.get("stages", [])
        for stage in stages:
            sid = stage.get("id")
            cf = stage.get("config_file")
            if sid and cf and sid in stage_configs:
                save_stage_toml(stage_configs[sid], configs_dir / cf)
    return web.json_response(body)


async def _handle_clear_runs(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    runs_dir = root / name / "runs"
    if runs_dir.exists():
        import shutil

        shutil.rmtree(runs_dir)
    return web.json_response({"status": "ok"})


async def _handle_list_runs(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    runs_dir = root / name / "runs"
    if not runs_dir.exists():
        return web.json_response([])
    import re as _re
    _run_pattern = _re.compile(r"^\d{8}-\d{6}$")
    runs = []
    for d in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir() or not _run_pattern.match(d.name):
            continue
        status_file = d / "status.json"
        status = "unknown"
        stages = []
        created_at = None
        if status_file.exists():
            try:
                payload = json.loads(status_file.read_text(encoding="utf-8"))
                status = payload.get("status", "unknown")
                created_at = payload.get("started_at")
                for s in payload.get("stages", []):
                    stages.append({"id": s["id"], "status": s.get("status", "pending")})
            except (json.JSONDecodeError, KeyError):
                pass
        if not created_at:
            from datetime import datetime as _dt
            created_at = _dt.fromtimestamp(d.stat().st_ctime).isoformat()
        runs.append(
            {"id": d.name, "status": status, "stages": stages, "created_at": created_at}
        )
    return web.json_response(runs)


async def _handle_open_run_dir(req: web.Request) -> web.Response:
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
        import subprocess as _sp

        _sp.Popen(["open", path])
    else:
        import subprocess as _sp

        _sp.Popen(["xdg-open", path])
    return web.json_response({"status": "opened"})


async def _handle_workflow_run_log(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    run_id = req.match_info["run_id"]
    root = req.app["workflows_root"]
    log_file = root / name / "runs" / run_id / "run.log"
    if not log_file.exists():
        return web.json_response({"lines": []})
    text = log_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return web.json_response({"lines": lines})


async def _handle_get_infra(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    wf_file = root / name / "workflow.yaml"
    data = load_workflow_yaml(wf_file)
    return web.json_response(data.get("infrastructure", {}))


async def _handle_set_infra(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    wf_file = root / name / "workflow.yaml"
    data = load_workflow_yaml(wf_file)
    infra = await req.json()
    data["infrastructure"] = infra
    save_workflow_yaml(data, wf_file)
    return web.json_response(infra)


async def _handle_run(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    wf_dir = root / name
    wf_file = wf_dir / "workflow.yaml"
    if not wf_file.exists():
        return web.json_response({"error": "not found"}, status=404)
    wf_data = load_workflow_yaml(wf_file)
    wf = WorkflowDefinition(**wf_data)
    eq = EventQueue()
    req.app["event_queues"].append(eq)
    scheduler = WorkflowScheduler(wf_dir, wf, eq)
    req.app["active_scheduler"] = scheduler

    def _run_in_thread():
        scheduler.run()

    t = threading.Thread(target=_run_in_thread, daemon=True)
    t.start()
    req.app["scheduler_thread"] = t
    return web.json_response({"status": "started"})


async def _handle_stop(req: web.Request) -> web.Response:
    scheduler = req.app.get("active_scheduler")
    if scheduler:
        scheduler.stop()
    return web.json_response({"status": "stopping"})


async def _handle_events(req: web.Request) -> web.Response:
    run_id = req.match_info["run_id"]
    eq = req.app["event_queues"][-1] if req.app["event_queues"] else EventQueue()

    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    await resp.prepare(req)

    try:
        while True:
            events = eq.drain()
            for ev in events:
                await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
            if events and events[-1].get("ev") in ("workflow_end",):
                break
            await asyncio.sleep(0.5)
    except ConnectionResetError:
        pass
    return resp


async def _handle_log(req: web.Request) -> web.Response:
    run_id = req.match_info["run_id"]
    return web.json_response({"lines": []})


async def _handle_get_schema(req: web.Request) -> web.Response:
    schema_name = req.match_info["schema_name"]
    try:
        schema = load_schema(schema_name)
        from workflow.i18n.schema_overlay import translate_schema
        schema = translate_schema(schema, schema_name, get_locale())
        return web.json_response(schema)
    except FileNotFoundError:
        return web.json_response({"error": "schema not found"}, status=404)


async def _handle_recent(req: web.Request) -> web.Response:
    root = req.app["workflows_root"]
    recent = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
        wf_file = d / "workflow.yaml"
        if d.is_dir() and wf_file.exists():
            data = load_workflow_yaml(wf_file)
            data["dir"] = d.name
            recent.append(data)
    return web.json_response(recent)


async def _handle_bucket_stats(req: web.Request) -> web.Response:
    body = await req.json()
    source_dir = body.get("source_dir", "")
    enabled_families = body.get("enabled_families", [])
    if not source_dir:
        return web.json_response({"error": "source_dir is required"}, status=400)
    from library.datasets.buckets import scan_dataset_bucket_distribution

    result = scan_dataset_bucket_distribution(source_dir, enabled_families)
    return web.json_response(result)


async def _handle_get_settings(req: web.Request) -> web.Response:
    cfg = _load_config()
    cfg["workflows_root"] = cfg.get("workflows_root", str(req.app["workflows_root"]))
    return web.json_response(cfg)


async def _handle_set_settings(req: web.Request) -> web.Response:
    body = await req.json()
    cfg = _load_config()
    cfg.update(body)
    _save_config(cfg)
    return web.json_response(cfg)


async def _handle_browse(req: web.Request) -> web.Response:
    body = await req.json()
    browse_path = body.get("path", "")
    browse_type = body.get("type", "directory")
    if not browse_path:
        return web.json_response({"entries": [], "error": "path is required"}, status=400)
    p = Path(browse_path)
    if browse_type == "file":
        parent = p.parent if not p.is_dir() else p
        name_filter = p.name if not p.is_dir() else ""
    else:
        parent = p if p.is_dir() else p.parent
        name_filter = ""
    if not parent.exists():
        return web.json_response({"entries": [], "error": "path not found"}, status=404)
    entries = []
    try:
        for child in sorted(parent.iterdir()):
            if name_filter and not child.name.lower().startswith(name_filter.lower()):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
            })
    except PermissionError:
        return web.json_response({"entries": [], "error": "permission denied"}, status=403)
    return web.json_response({"entries": entries, "path": str(parent)})


def start_server(app: web.Application, port: int = 8765) -> None:
    web.run_app(app, host="0.0.0.0", port=port, print=lambda msg: None)
