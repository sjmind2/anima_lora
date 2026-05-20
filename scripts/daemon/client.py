"""HTTP client for the daemon — used by the CLI commands and the ComfyUI node.

Pure stdlib (``urllib``) so it imports cleanly from inside ComfyUI without
dragging in ``library.*`` / torch. ``ensure_daemon`` auto-starts a console-
detached daemon and waits for ``/health`` — the "spawn it if it isn't up" path
both the ComfyUI node and ``make daemon`` rely on.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Iterator, Optional

from . import config, proc


def venv_python() -> str:
    """Resolve the anima_lora venv interpreter.

    The daemon must run under anima's venv (it builds ``accelerate launch``
    commands with ``sys.executable``), *not* whatever interpreter the caller
    happens to be — notably ComfyUI's. Probe the usual venv layouts under the
    repo root and its parent, then fall back to ``sys.executable``.
    """
    names = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    for base in (config.ROOT, config.ROOT.parent):
        cand = base / ".venv" / names[0] / names[1]
        if cand.exists():
            return str(cand)
    return sys.executable


def _resolve_port() -> int:
    info = proc.read_pidfile(config.PIDFILE)
    if info and info.get("port"):
        return int(info["port"])
    return config.DEFAULT_PORT


class DaemonClient:
    def __init__(self, port: Optional[int] = None) -> None:
        self.port = port or _resolve_port()

    @property
    def base(self) -> str:
        return f"http://{config.HOST}:{self.port}"

    # ----- request plumbing -----

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        timeout: float = 30.0,
    ):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    # ----- typed endpoints -----

    def health(self) -> Optional[dict]:
        try:
            return self._request("GET", "/health", timeout=3.0)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def submit(
        self,
        *,
        method: str,
        preset: str = "default",
        methods_subdir: Optional[str] = None,
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/jobs",
            {
                "method": method,
                "preset": preset,
                "methods_subdir": methods_subdir,
                "overrides": overrides or {},
                "extra": extra or [],
            },
        )

    def list_jobs(self) -> list:
        return self._request("GET", "/jobs") or []

    def get(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def stop(self, job_id: Optional[str] = None) -> dict:
        # No job_id → daemon's "stop the running job" semantics. We resolve the
        # active job here so the URL stays RESTful.
        if job_id is None:
            health = self.health() or {}
            job_id = health.get("active_job")
            if not job_id:
                return {"error": "no active job"}
        return self._request("POST", f"/jobs/{job_id}/stop")

    def shutdown(self, *, kill_jobs: bool = True) -> Optional[dict]:
        try:
            return self._request("POST", "/shutdown", {"kill_jobs": kill_jobs})
        except (urllib.error.URLError, OSError, ValueError):
            return None

    # ----- SSE streams -----

    def stream(self, path: str) -> Iterator[str]:
        """Yield ``data:`` payloads from an SSE endpoint until the socket drops."""
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=None) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("data: "):
                    yield line[len("data: ") :]

    def stream_events(self) -> Iterator[str]:
        return self.stream("/events")

    def stream_logs(self, job_id: str) -> Iterator[str]:
        return self.stream(f"/jobs/{job_id}/logs")


def ensure_daemon(*, timeout: float = 60.0, port: Optional[int] = None) -> DaemonClient:
    """Return a client to a live daemon, starting one if needed.

    Idempotent: if ``/health`` answers we just return a client. Otherwise spawn
    ``python -m scripts.daemon`` detached (stdout → ``daemon.log``) and poll
    ``/health`` until it answers or ``timeout`` elapses.
    """
    port = port or _resolve_port()
    client = DaemonClient(port)
    if client.health() is not None:
        return client

    config.ensure_state_dirs()
    proc.spawn_detached(
        [venv_python(), "-m", "scripts.daemon", str(port)],
        cwd=config.ROOT,
        stdout_path=config.DAEMON_LOG,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.health() is not None:
            return client
        time.sleep(0.5)
    raise RuntimeError(
        f"daemon did not come up within {timeout:.0f}s; see {config.DAEMON_LOG}"
    )


def is_running() -> bool:
    return DaemonClient().health() is not None
