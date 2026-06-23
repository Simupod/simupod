"""Local result cache: download a job's bundle tarball once and extract it into
``<cache_dir>/<job_id>/``, which :class:`~photonhub.SimulationData` then reads
exactly like a local run's output directory. Re-fetching a finished job is free.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from .client import HttpClient
from .config import WebConfig


def job_dir(cfg: WebConfig, job_id: str) -> Path:
    return Path(cfg.cache_dir) / job_id


def _safe_extract(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"unsafe path in result bundle: {member.name!r}")
        tar.extractall(dest)


def download_bundle(http: HttpClient, cfg: WebConfig, job_id: str) -> Path:
    """Return a local dir holding the job's manifest.json + monitor data,
    downloading + extracting it on first call (cached thereafter)."""
    out = job_dir(cfg, job_id)
    if (out / "manifest.json").is_file() or list(out.glob("*.h5")):
        return out
    data = http.download_result(job_id)
    _safe_extract(data, out)
    return out
