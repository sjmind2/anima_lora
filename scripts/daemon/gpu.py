"""Best-effort GPU-occupancy probe for the serial dequeue guard.

Single GPU, one job at a time → between jobs exactly zero training procs
should hold VRAM. Before launching the next job the manager asks who's holding
the GPU so it can distinguish "free, go" from "a known dead job leaked VRAM,
reap it" from "an unknown proc is using the card, don't blind-kill it".

pynvml first (no subprocess, exact), ``nvidia-smi`` fallback, then give up
gracefully — the guard degrades to "assume free" rather than deadlocking the
queue if neither is available (e.g. CPU-only CI).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def gpu_pids() -> Optional[set[int]]:
    """PIDs with a compute context on any visible GPU.

    ``None`` means "couldn't tell" (no NVML, no nvidia-smi) — distinct from an
    empty set, which means "queried successfully, GPU is free".
    """
    pids = _gpu_pids_nvml()
    if pids is not None:
        return pids
    return _gpu_pids_smi()


def _gpu_pids_nvml() -> Optional[set[int]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        out: set[int] = set()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for fn in (
                pynvml.nvmlDeviceGetComputeRunningProcesses,
                pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                try:
                    for proc in fn(h):
                        out.add(int(proc.pid))
                except Exception:
                    continue
        return out
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_pids_smi() -> Optional[set[int]]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [
                smi,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    pids: set[int] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids
