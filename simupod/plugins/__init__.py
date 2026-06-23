"""PhotonHub optional plugins — CPU-only analysis tools layered on the client.

Plugins are *self-contained* helpers that do not run on the GPU engine and are
not part of the simulation wire format. They live outside ``components/`` so the
frozen spec models stay pure. Import what you need explicitly::

    from simupod.plugins import ModeSolver

Phase-1 plugins
---------------
``ModeSolver`` — a finite-difference eigenmode (FDE) solver for the guided
modes of a *straight* dielectric waveguide cross-section (semi-vectorial,
quasi-TE/quasi-TM). CPU/numpy only; see :mod:`simupod.plugins.modes`.

``VectorModeSolver`` — the *full-vectorial* FDE solver for the same straight
cross-section (Fallahkhair–Li–Murphy transverse-H operator): real hybrid/TM
``n_eff``, all six field components, and group index. CPU only; requires scipy.
See :mod:`simupod.plugins.vector_modes`.

``run_eme`` — a minimal eigenmode-expansion (EME) propagator: staircase a
z-varying device into z-invariant sections, mode-match at the interfaces and
cascade the per-section/-interface scattering matrices (Redheffer star product)
into one device S-matrix. Built on ``VectorModeSolver``; CPU only. See
:mod:`simupod.plugins.eme`.
"""

from .cvcs import cvcs_sections, interpolate_mode, interpolate_plane
from .eme import (
    EMEResult,
    Section,
    cascade,
    interface_smatrix,
    propagation_smatrix,
    run_eme,
    star_product,
    waveguide_section,
)
from .mode_tracking import (
    TrackingResult,
    match_modes,
    reorder_to_tracks,
    track_modes,
    transverse_overlap,
)
from .mode_devices import (
    ModeMonitor,
    mode_monitor,
    mode_source,
    mode_source_vector,
    solve_modes_by_freq,
    transmission,
)
from .mode_overlap import mode_amplitude, mode_transmission, vector_modal_fields
from .modes import Mode, ModeSolver
from .near_field import FarField, equivalent_currents, far_field
from .smatrix import (
    SPort,
    assemble_smatrix,
    assert_passive,
    assert_reciprocal,
    is_passive,
    is_reciprocal,
    passivity_violation,
    reciprocity_error,
    smatrix,
)
from .vector_modes import VectorMode, VectorModeSolver

__all__ = [
    "EMEResult",
    "FarField",
    "Mode",
    "ModeMonitor",
    "ModeSolver",
    "SPort",
    "Section",
    "TrackingResult",
    "VectorMode",
    "VectorModeSolver",
    "assemble_smatrix",
    "assert_passive",
    "assert_reciprocal",
    "cascade",
    "cvcs_sections",
    "equivalent_currents",
    "far_field",
    "interface_smatrix",
    "interpolate_mode",
    "interpolate_plane",
    "is_passive",
    "is_reciprocal",
    "match_modes",
    "mode_amplitude",
    "mode_monitor",
    "mode_source",
    "mode_source_vector",
    "mode_transmission",
    "passivity_violation",
    "propagation_smatrix",
    "reciprocity_error",
    "reorder_to_tracks",
    "run_eme",
    "smatrix",
    "solve_modes_by_freq",
    "star_product",
    "track_modes",
    "transmission",
    "transverse_overlap",
    "vector_modal_fields",
    "waveguide_section",
]
