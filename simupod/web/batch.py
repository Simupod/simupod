"""Cloud batch — submit N simulations and assemble the SAME
:class:`~simupod.runners.batch.BatchData` the local path returns, so per-name
partial failures (``batch_data.errors[name]``) work identically. This realizes
the local-backend docstring's promise that on the cloud a Batch "becomes a
fan-out across GPUs": each name is an independent job the coordinator spreads
across capacity.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Optional

from ..runners.batch import BatchData, _check_batch_name
from ..runners.local import SolverRunError
from .config import get_config
from .run import _cloud_run


class Batch:
    def __init__(self, simulations: Mapping[str, object]):
        for name in simulations:
            _check_batch_name(name)
        self.simulations = dict(simulations)

    def estimate_cost(self, **kwargs):
        """Per-name CostEstimate + the batch total (local, deterministic)."""
        per = {name: sim.cost_estimate(**kwargs)
               for name, sim in self.simulations.items()}
        total = sum(e.usd for e in per.values())
        return per, total

    def run(self, *, device=None, max_workers: int = 4,
            timeout: Optional[float] = None) -> BatchData:
        cfg = get_config()
        results = {}
        errors = {}

        def _one(item):
            name, sim = item
            try:
                return name, _cloud_run(sim, name=name, device=device,
                                        timeout=timeout, cfg=cfg), None
            except SolverRunError as e:
                return name, None, e

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            for name, data, err in ex.map(_one, self.simulations.items()):
                if err is None:
                    results[name] = data
                else:
                    errors[name] = err

        return BatchData(results, errors, cfg.cache_dir,
                         list(self.simulations))
