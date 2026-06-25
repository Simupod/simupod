"""Environment-variable lookup with the ``SIMUPOD_`` → legacy ``PHOTONHUB_``
fallback, single-sourced so the back-compat (the old prefix is still honored)
lives in exactly one place rather than being re-implemented per call site."""

from __future__ import annotations

import os
from typing import Optional


def env(suffix: str) -> Optional[str]:
    """Read ``SIMUPOD_<suffix>``, falling back to the legacy ``PHOTONHUB_<suffix>``
    (e.g. ``env("API_KEY")`` → ``$SIMUPOD_API_KEY`` or ``$PHOTONHUB_API_KEY``)."""
    return os.environ.get(f"SIMUPOD_{suffix}") or os.environ.get(f"PHOTONHUB_{suffix}")
