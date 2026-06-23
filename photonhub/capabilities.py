"""Capability gating — fail unsupported features at MODEL CONSTRUCTION with a
clear "available in <version>" message, never at engine submission.

The roadmap (P1) asks for ``phsolver --capabilities`` to be wired into the
pydantic validators so a user learns a feature is unsupported when they *build*
the :class:`~photonhub.Simulation` — not after a job is submitted and the engine
rejects it. This module is the single place that:

1. pins the feature manifest the v1 **engine** advertises
   (``ENGINE_ADVERTISED_FEATURES``), mirroring ``phsolver --capabilities``;
2. maps each *representable-but-unsupported* feature to the version that will
   introduce it, with a human message (:func:`unavailable` /
   :class:`UnavailableFeature`); the component validators raise these at
   construction; and
3. reads the engine's live manifest (:func:`engine_capabilities`) so a test/CI
   gate can assert the client never drifts from what the solver can actually run
   (:func:`engine_feature_drift`) — the literal "wired to ``--capabilities``"
   link.

Why a registry instead of just raising inline? The cut-list features
(``ClipOperation``/``CustomMedium``/anisotropic + magnetic media) are not even
*representable* in the v1 models (``Medium`` is scalar permittivity + an Ohmic
conductivity, no permeability/tensor), so ``extra="forbid"`` already rejects
them — but with a generic "extra inputs are not permitted". The genuinely
representable-but-unsupported combination is a **graded grid + plane wave**
(the engine restricts plane-wave injection to a uniform grid, NUMERICS.md
§15.9). Centralizing the message here keeps the wording consistent and gives the
next such gate a home instead of folklore.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Union

SCHEMA_MAJOR = 1

# The feature flags the v1 engine advertises via ``phsolver --capabilities``
# (engine/src/main/phsolver.cpp cmd_capabilities). Pinned here so the drift test
# fails the build the moment the engine manifest changes without the client
# being updated in lockstep — in EITHER direction.
#
# Known gap (documented, not yet fixed): the engine also runs a graded
# (nonuniform) grid since schema v1.2, but cmd_capabilities() still advertises
# only "uniform_grid". When the engine starts advertising "graded_grid", add it
# here and the pin test will guide that change rather than silently passing.
ENGINE_ADVERTISED_FEATURES = frozenset({
    "uniform_grid", "point_dipole", "gaussian_pulse", "pec", "periodic",
    "field_time", "field_snapshot",
    # Phase 1a-1 (NUMERICS.md §9-§13)
    "structures", "lossy_media", "pml", "plane_wave", "field_dft", "flux",
})

_PROBE_TIMEOUT_S = 30.0


class UnavailableFeature(ValueError):
    """A feature the client can express but this schema major cannot run.

    A ``ValueError`` so it is caught by pydantic and surfaced as part of a
    ``ValidationError`` when raised inside a model validator. ``key`` is the
    registry key; ``target`` is the version/phase that introduces the feature.
    """

    def __init__(self, key: str, message: str, *, target: str):
        super().__init__(message)
        self.key = key
        self.target = target


# Representable-but-unsupported features: registry key -> (what it is, the
# version/phase that introduces it, a spec reference, and actionable advice).
_UNAVAILABLE = {
    "graded_plane_wave": (
        "plane-wave injection on a graded (nonuniform) grid",
        "a later phase",
        "NUMERICS.md §15.9",
        "use a uniform grid (UniformGridSpec) for the plane-wave source, or "
        "drive the graded grid with a point-dipole source instead",
    ),
    "magnetic_symmetry_z": (
        "an even / magnetic (PMC) symmetry plane on the z axis (symmetry.z = +1)",
        "a later phase",
        "NUMERICS.md §20.4",
        "PMC is available on x and y; for z use -1 (odd / electric, available on "
        "all axes) or 0 (no symmetry on z)",
    ),
}


def unavailable(key: str) -> UnavailableFeature:
    """Build the standardized :class:`UnavailableFeature` for ``key``.

    Message shape (stable — tests and docs depend on the "is not available in
    schema v<major>" / "available in" phrasing):

        "<what> is not available in schema v1 — available in <target>
         (<ref>); <advice>."
    """
    what, target, ref, advice = _UNAVAILABLE[key]
    message = (
        f"{what} is not available in schema v{SCHEMA_MAJOR} — available in "
        f"{target} ({ref}); {advice}."
    )
    return UnavailableFeature(key, message, target=target)


def engine_capabilities(
    solver_path: Union[str, Path, None] = None,
) -> Optional[dict]:
    """Parse ``phsolver --capabilities`` into a dict, or ``None`` when no solver
    binary is configured/found.

    Locates the binary the same way a run does (explicit arg, ``$PHOTONHUB_SOLVER``,
    ``PATH``, then the in-repo build dir) via :func:`photonhub.find_solver`.
    Raises on a present-but-broken binary (non-zero exit or unparseable output)
    — a silent fallback would defeat the point of the drift gate.
    """
    # Lazy import: runners.local imports the components package, which imports
    # this module — importing find_solver at module load would be a cycle.
    from .runners.local import find_solver

    solver = find_solver(solver_path)
    if solver is None:
        return None
    proc = subprocess.run(
        [str(solver), "--capabilities"],
        capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{solver} --capabilities exited {proc.returncode}: "
            f"{proc.stderr.strip()[:400]}")
    return json.loads(proc.stdout)


def engine_feature_drift(
    solver_path: Union[str, Path, None] = None,
) -> Optional[set]:
    """The symmetric difference between the engine's advertised feature set and
    :data:`ENGINE_ADVERTISED_FEATURES`. Empty set = in sync; ``None`` when no
    solver binary is available (so callers can skip rather than fail). The CI
    drift gate asserts this is the empty set.
    """
    caps = engine_capabilities(solver_path)
    if caps is None:
        return None
    advertised = set(caps.get("features", []))
    return advertised ^ set(ENGINE_ADVERTISED_FEATURES)
