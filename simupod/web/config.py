"""Configuration for the cloud client (``ph.web``).

``configure(api_key=..., url=...)`` sets the active config; values fall back to
``$PHOTONHUB_API_KEY`` / ``$PHOTONHUB_URL`` (mirroring ``find_solver``'s
explicit→env precedence, where a missing required value is an error, not a
silent default). ``WebError`` is raised for config/transport/auth problems —
distinct from ``SolverRunError``, which is reserved for a simulation actually
failing, so the two are never confused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Default endpoint; override with url=/$PHOTONHUB_URL (e.g. the prod API host).
DEFAULT_URL = "http://localhost:8000"


class WebError(RuntimeError):
    """A cloud client/transport/auth error (bad key, network, 4xx/5xx) — NOT a
    simulation failure (that is ``SolverRunError``)."""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class WebConfig:
    url: str
    api_key: str
    cache_dir: Path
    poll_interval_s: float = 2.0
    poll_backoff_max_s: float = 15.0
    request_timeout_s: float = 30.0


_CONFIG: Optional[WebConfig] = None


def _default_cache_dir() -> Path:
    env = os.environ.get("PHOTONHUB_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "photonhub" / "jobs"


def configure(api_key: Optional[str] = None, url: Optional[str] = None, *,
              cache_dir=None, poll_interval_s: float = 2.0,
              poll_backoff_max_s: float = 15.0,
              request_timeout_s: float = 30.0) -> WebConfig:
    """Set the active cloud configuration. Returns it for inspection."""
    global _CONFIG
    key = api_key or os.environ.get("PHOTONHUB_API_KEY")
    if not key:
        raise WebError(
            "no API key: pass api_key= or set $PHOTONHUB_API_KEY "
            "(create one with ph.web.create_api_key after signing in)")
    base = (url or os.environ.get("PHOTONHUB_URL") or DEFAULT_URL).rstrip("/")
    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    _CONFIG = WebConfig(
        url=base, api_key=key, cache_dir=cache,
        poll_interval_s=poll_interval_s, poll_backoff_max_s=poll_backoff_max_s,
        request_timeout_s=request_timeout_s,
    )
    return _CONFIG


def get_config() -> WebConfig:
    """The active config, building one from the environment on first use."""
    if _CONFIG is not None:
        return _CONFIG
    if os.environ.get("PHOTONHUB_API_KEY"):
        return configure()
    raise WebError(
        "simupod.web is not configured; call "
        "ph.web.configure(api_key=..., url=...) or set $PHOTONHUB_API_KEY")


def reset() -> None:
    """Clear the active config (mainly for tests)."""
    global _CONFIG
    _CONFIG = None
