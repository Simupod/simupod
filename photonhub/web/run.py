"""Cloud run entry points — the prime directive: ``ph.web.run_async`` returns the
**same** :class:`~photonhub.runners.batch.Job` as the local path, so
``job = ph.web.run_async(sim); data = job.result()`` reads identically whether
local or cloud, and a server-side failure surfaces as the same
:class:`SolverRunError`.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from ..data import SimulationData
from ..runners.batch import Job
from ..runners.local import SolverRunError
from . import cache
from .client import HttpClient
from .config import WebConfig, get_config

ProgressCb = Optional[Callable[[dict], None]]


def _poll_and_download(http: HttpClient, cfg: WebConfig, job_id: str, *,
                       progress: ProgressCb, timeout: Optional[float]) -> object:
    deadline = (time.monotonic() + timeout) if timeout else None
    interval = cfg.poll_interval_s
    while True:
        st = http.get_job(job_id)
        state = st["state"]
        if progress and st.get("progress"):
            progress(st["progress"])
        if state == "succeeded":
            break
        if state == "failed":
            err = st.get("error") or {}
            raise SolverRunError(
                f"cloud job {job_id} failed: {err.get('reason', 'unknown')}",
                stderr_tail=err.get("stderr_tail"))
        if state == "cancelled":
            raise SolverRunError(f"cloud job {job_id} was cancelled")
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"cloud job {job_id!r} not finished after {timeout} s "
                "(it is still running; call result() again)")
        time.sleep(interval)
        interval = min(interval * 1.5, cfg.poll_backoff_max_s)
    return cache.download_bundle(http, cfg, job_id)


def _cloud_run(sim, *, name=None, device=None, progress: ProgressCb = None,
               timeout: Optional[float] = None,
               cfg: Optional[WebConfig] = None) -> SimulationData:
    cfg = cfg or get_config()
    http = HttpClient(cfg)
    resp = http.submit_job(sim.to_wire_dict(), name=name, device=device)
    job_id = resp["job_id"]
    bundle_dir = _poll_and_download(http, cfg, job_id, progress=progress,
                                    timeout=timeout)
    try:
        return SimulationData(bundle_dir)
    except (OSError, ValueError, KeyError) as e:
        # mirror run_local's "solver lies" guard
        raise SolverRunError(
            f"cloud job {job_id} returned unreadable outputs: {e}") from e


def run(sim, *, name=None, device=None, progress: ProgressCb = None,
        timeout: Optional[float] = None) -> SimulationData:
    """Submit ``sim`` to the cloud and block until its result is ready. Returns a
    :class:`SimulationData`; raises :class:`SolverRunError` if the run fails,
    :class:`WebError` for transport/auth problems."""
    return _cloud_run(sim, name=name, device=device, progress=progress,
                      timeout=timeout)


def run_async(sim, *, name=None, device=None, progress: ProgressCb = None,
              timeout: Optional[float] = None) -> Job:
    """Submit ``sim`` and return a :class:`Job` handle immediately — the SAME
    handle type as the local ``ph.run_async``. Collect with ``job.result()``."""
    cfg = get_config()  # validate config now, fail fast (not on the worker thread)
    return Job(
        lambda: _cloud_run(sim, name=name, device=device, progress=progress,
                           timeout=timeout, cfg=cfg),
        name=name)
