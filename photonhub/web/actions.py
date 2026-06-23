"""One-shot cloud actions that don't need a Job handle."""

from __future__ import annotations

from .client import HttpClient
from .config import get_config


def whoami() -> dict:
    return HttpClient(get_config()).whoami()


def account() -> dict:
    """Balance/usage for the configured account (micro-USD + dollar fields)."""
    return HttpClient(get_config()).account()


def estimate(sim) -> dict:
    """Server-side dollar quote for ``sim`` (matches ``sim.cost_estimate()``)."""
    return HttpClient(get_config()).estimate(sim.to_wire_dict())


def create_api_key(name: str = "default") -> dict:
    """Mint a new API key (the plaintext ``token`` is returned exactly once)."""
    return HttpClient(get_config()).create_api_key(name)


def cancel(job_id: str) -> dict:
    return HttpClient(get_config()).cancel_job(job_id)
