from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from aiohttp import web

from workflow.config import (
    load_workflow_yaml,
    save_workflow_yaml,
    load_stage_toml,
    save_stage_toml,
    load_schema,
    resolve_placeholders,
)
from workflow.logger import EventQueue
from workflow.models import WorkflowDefinition, WorkflowStage
from workflow.scheduler import WorkflowScheduler


def create_app(workflows_root: Path | str | None = None) -> web.Application:
    if workflows_root is None:
        workflows_root = Path.home() / ".anima_workflow"
    workflows_root = Path(workflows_root)
    workflows_root.mkdir(parents=True, exist_ok=True)

    app = web.Application()
    app["workflows_root"] = workflows_root
    app["event_queues"]: list[EventQueue] = []
    app["scheduler_thread"] = None
    app["active_scheduler"] = None

    app.router.add_get("/api/workflows", _handle_list_workflows)
    app.router.add_post("/api/workflows", _handle_create_workflow)
    app.router.add_get("/api/workflows/{name}", _handle_get_workflow)
    app.router.add_put("/api/workflows/{name}", _handle_update_workflow)
    app.router.add_delete("/api/workflows/{name}/runs", _handle_clear_runs)
    app.router.add_get("/api/workflows/{name}/infrastructure", _handle_get_infra)
    app.router.add_put("/api/workflows/{name}/infrastructure", _handle_set_infra)
    app.router.add_post("/api/workflows/{name}/run", _handle_run)
    app.router.add_post("/api/workflows/{name}/stop", _handle_stop)
    app.router.add_get("/api/runs/{run_id}/events", _handle_events)
    app.router.add_get("/api/runs/{run_id}/log", _handle_log)
    app.router.add_get("/api/schemas/{schema_name}", _handle_get_schema)
    app.router.add_get("/api/recent-workflows", _handle_recent)

    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.router.add_get("/", _handle_index)
        app.router.add_static("/static", str(web_dir))

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
    return web.json_response(data)


async def _handle_update_workflow(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    wf_file = root / name / "workflow.yaml"
    body = await req.json()
    save_workflow_yaml(body, wf_file)
    return web.json_response(body)


async def _handle_clear_runs(req: web.Request) -> web.Response:
    name = req.match_info["name"]
    root = req.app["workflows_root"]
    runs_dir = root / name / "runs"
    if runs_dir.exists():
        import shutil
        shutil.rmtree(runs_dir)
    return web.json_response({"status": "ok"})


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
        scheduler.run(log_file=wf_dir / "runs" / "latest" / "run.log")

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


def start_server(app: web.Application, port: int = 8765) -> None:
    web.run_app(app, host="0.0.0.0", port=port, print=lambda msg: None)
