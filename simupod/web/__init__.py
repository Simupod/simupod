"""Cloud client for the SimuPod metered compute API.

Reads identically to the local path — same ``Job`` / ``SimulationData`` /
``SolverRunError`` — only the namespace differs:

>>> import simupod as ph
>>> ph.web.configure(api_key="ph_live_...", url="https://api.simupod.com")
>>> job = ph.web.run_async(sim)     # cf. ph.run_async(sim) locally
>>> data = job.result()             # SimulationData, same as local
>>> probe = data["probe"]           # xarray.DataArray
"""

from .actions import account, cancel, create_api_key, estimate, whoami
from .batch import Batch
from .client import HttpClient
from .config import WebConfig, WebError, configure, get_config, reset
from .run import run, run_async

__all__ = [
    "configure",
    "get_config",
    "reset",
    "WebConfig",
    "WebError",
    "HttpClient",
    "run",
    "run_async",
    "Batch",
    "estimate",
    "account",
    "whoami",
    "create_api_key",
    "cancel",
]
