"""Mode source & monitor builders — the client bridge from an FDE eigenmode to
the engine's ModeSource (NUMERICS.md §18) and to a mode-resolved transmission
readout.

``mode_source`` resamples a frozen FDE :class:`~simupod.plugins.modes.Mode`
onto a simulation's transverse grid plane and returns a
:class:`~simupod.components.sources.ModeSource` the engine injects via TF/SF.
``mode_monitor`` returns a :class:`ModeMonitor`, which carries a 4-tangential
``FieldDftMonitor`` to add to the simulation and a ``.transmission(data)``
post-process that overlaps the recorded plane onto the mode (forward/backward
power ``T``) via :func:`simupod.plugins.mode_overlap.mode_transmission`.

The injection and the overlap share one scalar-limit modal-H convention
(``h ≈ (n_eff/eta0) z_hat x e``), so a clean single-mode straight waveguide
reads ``T ≈ 1`` forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from ..components.grid import graded_primary_spacings, realized_cells
from ..components.monitors import FieldDftMonitor
from ..components.sources import ModeSource
from ..components.source_time import SourceTimeType
from .mode_overlap import (
    ETA0,
    _TRANSVERSE,
    _cell_widths,
    modal_fields,
    mode_transmission,
    vector_modal_fields,
)
from .modes import Mode

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}

# Free-space speed of light (m/s), matching ModeSolver.C0 — used to map a
# monitor/source frequency (Hz) to the FDE solver's wavelength (microns) via
# wavelength_um = C0 / freq_hz * 1e6 for the broadband (num_freqs) mode solves.
C0 = 2.99792458e8

# The four tangential components (2 E, 2 H) for a plane normal to `axis`, in the
# order mode_transmission expects.
_TANGENTIAL: Dict[str, Tuple[str, str, str, str]] = {
    "x": ("Ey", "Ez", "Hy", "Hz"),
    "y": ("Ez", "Ex", "Hz", "Hx"),
    "z": ("Ex", "Ey", "Hx", "Hy"),
}


def _axis_cell_centers(simulation, axis_name: str) -> np.ndarray:
    """Transverse cell-center coordinates (microns) along one axis.

    A uniform axis uses ``(i + 0.5)·dl``. A GRADED axis (GradedGridSpec coords)
    uses the midpoints of its primary-node cells (the §15.2 dual nodes), so the
    mode profile is sampled at the TRUE cell centers — this is what lets a mode
    source / monitor live on a transverse-graded mesh.

    The mode source's PROPAGATION axis must still be uniform (the §18 aux line
    is 1-D along it); the engine enforces that. This helper only ever samples
    the two TRANSVERSE plane axes, either of which may grade."""
    idx = _AXIS_IDX[axis_name]
    q = simulation._axis_coords_um(idx)
    if q is None:  # uniform axis (UniformGridSpec, or a non-graded graded axis)
        dl = simulation.grid.dl_um
        size = simulation.size_um[idx]
        n = realized_cells(size, dl)
        return (np.arange(n) + 0.5) * dl
    # Graded axis: cell i spans [q[i], q[i+1]] (q[n] = §15.1 replicate-last
    # closing node), so its center is q[i] + dq[i]/2 with dq the primary
    # spacings (replicate-last for the final cell). Matches the engine's §15.2
    # dual-node convention, so the resampled profile lands on the cells the
    # solver injects into.
    qa = np.asarray(q, dtype=float)
    dq = np.asarray(graded_primary_spacings(tuple(q)), dtype=float)
    return qa + dq / 2.0


def _default_center(simulation, axis: str) -> Tuple[float, float]:
    """The transverse domain midpoints (t1, t2) — where a centered waveguide
    sits, used as the default mode location."""
    t1, t2 = _TRANSVERSE[axis]
    return (
        simulation.size_um[_AXIS_IDX[t1]] / 2.0,
        simulation.size_um[_AXIS_IDX[t2]] / 2.0,
    )


def _broadband_arrays(modes_by_freq, resample, central_pol, central_major,
                      central_minor):
    """Pack ``{freq_hz: Mode}`` into the :class:`ModeSource` broadband kwargs
    (``freqs_hz`` / ``n_eff_by_freq`` / ``profiles_by_freq`` [+ minor]).

    Returns ``{}`` for fewer than two entries — the legacy single-mode launch.
    Each mode is resampled with the SAME ``resample`` callable as the band-centre
    mode (it returns ``(major_flat, minor_flat_or_None, polarization)``). Two
    invariants make the engine's partition-of-unity windowing well-posed:
    (1) the major polarization must not change across the band (same guided
    mode); (2) each profile's arbitrary global eigen-sign is aligned to
    ``central_major`` (the same sign applied to the minor to preserve the
    component ratio) so adjacent windowed carriers add coherently rather than
    cancel."""
    if modes_by_freq is None or len(modes_by_freq) < 2:
        return {}
    freqs = sorted(float(f) for f in modes_by_freq)
    has_minor = central_minor is not None
    neffs, majors, minors = [], [], []
    for f in freqs:
        m = modes_by_freq[f]
        maj, minr, pol = resample(m)
        if pol != central_pol:
            raise ValueError(
                f"the mode's major polarization changes across the band "
                f"({central_pol} -> {pol} at {f:.4g} Hz); a broadband source "
                "needs the SAME mode at every frequency (narrow the band, or "
                "select the matching mode_index in solve_modes_by_freq)"
            )
        sign = -1.0 if float(np.dot(maj, central_major)) < 0.0 else 1.0
        majors.append(tuple(float(v) for v in sign * maj))
        neffs.append(float(m.n_eff))
        if has_minor:
            if minr is None:
                raise ValueError(
                    "the band-centre mode is full-vector but the mode at "
                    f"{f:.4g} Hz has no minor component"
                )
            minors.append(tuple(float(v) for v in sign * minr))
    out = dict(
        freqs_hz=tuple(freqs),
        n_eff_by_freq=tuple(neffs),
        profiles_by_freq=tuple(majors),
    )
    if has_minor:
        out["profiles_minor_by_freq"] = tuple(minors)
    return out


def mode_source(
    simulation,
    mode: Mode,
    *,
    axis: str,
    position_um: float,
    source_time: SourceTimeType,
    direction: str = "+",
    amplitude: float = 1.0,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
) -> ModeSource:
    """Build a :class:`ModeSource` injecting ``mode`` on the ``axis`` plane at
    ``position_um`` of ``simulation`` (uniform grid).

    The mode's major transverse-E profile is resampled (peak-normalized) onto
    the grid's transverse cells; ``amplitude`` is then the peak injected field.
    ``center_um`` places the waveguide in the transverse plane (default: the
    domain center, i.e. a centered guide). ``thickness_axis`` is the simulation
    axis along the guide's slab thickness; pass the slab normal (e.g. ``"z"``)
    for any non-x propagation so the mode is not rotated 90 degrees (see
    :func:`~simupod.plugins.mode_overlap.modal_fields`). ``None`` keeps the
    legacy thickness-on-second-transverse-axis mapping.

    **Broadband injection (``num_freqs`` analogue, NUMERICS.md §18.3).** Pass
    ``modes_by_freq`` (``{freq_hz: Mode}`` from :func:`solve_modes_by_freq`, the
    same map a :func:`mode_monitor` takes) to inject a FREQUENCY-DEPENDENT
    profile and ``n_eff`` instead of the single frozen ``mode``. Each mode is
    resampled and the engine partition-of-unity-windows them across the band, so
    a wide-band / dispersive launch stays mode-matched at every frequency. The
    positional ``mode`` remains the band-centre representative (and the global
    sign reference the per-frequency profiles are aligned to). With fewer than
    two entries this is a no-op (the single ``mode`` is used)."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    t1_name, t2_name = _TRANSVERSE[axis]
    u_coords = _axis_cell_centers(simulation, t1_name)
    v_coords = _axis_cell_centers(simulation, t2_name)
    if center_um is None:
        center_um = _default_center(simulation, axis)

    def _resample(m: Mode):
        """Peak-normalized major-E profile (flat C-order) + its polarization,
        resampled onto this plane — the shared scalar-source readout."""
        fields = modal_fields(
            m, u_coords, v_coords, axis=axis, n_eff=m.n_eff,
            center_um=center_um, thickness_axis=thickness_axis,
        )
        # The major-E component is whichever of e1/e2 modal_fields filled (the
        # other is identically zero); read it back rather than re-deriving.
        if np.any(fields["e1"]):
            profile2d, pol = fields["e1"], "E" + t1_name  # [iv, iu]
        else:
            profile2d, pol = fields["e2"], "E" + t2_name
        peak = float(np.max(np.abs(profile2d)))
        if not peak > 0.0:
            raise ValueError(
                "the resampled mode profile is identically zero on this plane "
                "— check the mode window vs the simulation transverse extent / "
                "center"
            )
        # [iv*nu+iu] = [cv*nu+cu]; no minor in the scalar limit.
        return (profile2d / peak).reshape(-1), None, pol

    profile, _, polarization = _resample(mode)

    bb = _broadband_arrays(
        modes_by_freq, _resample, polarization, profile, central_minor=None,
    )
    return ModeSource(
        axis=axis,
        direction=direction,
        position_um=position_um,
        polarization=polarization,
        amplitude=amplitude,
        n_eff=float(mode.n_eff),
        nu=int(u_coords.size),
        nv=int(v_coords.size),
        profile=tuple(float(v) for v in profile),
        source_time=source_time,
        **bb,
    )


def mode_source_vector(
    simulation,
    mode,
    *,
    axis: str,
    position_um: float,
    source_time: SourceTimeType,
    direction: str = "+",
    power_watts: float = 1.0,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, object]] = None,
) -> ModeSource:
    """Build a FULL-VECTOR, power-normalized :class:`ModeSource` from a
    ``VectorMode`` (NUMERICS.md §18).

    Where :func:`mode_source` injects the scalar-limit major-E component
    (peak-normalized, ``amplitude`` = peak field), this packs BOTH transverse-E
    components of the full-vector mode and **power-normalizes** the launch to
    ``power_watts`` (default **1 W**). Both transverse-E profiles are resampled
    onto the grid's transverse cells preserving their true component ratio (via
    :func:`~simupod.plugins.mode_overlap.vector_modal_fields`); the minor
    component rides the same guided-mode aux carrier as the major (engine §18.2),
    with its own scalar-limit paired H.

    **1 W normalization (computed here, on the Python side; the engine stays
    power-agnostic).** The engine injects ``E_t = amplitude * profile`` and the
    scalar-limit paired ``H = (n_eff/eta0)(z_hat x E_t)``, so the launched modal
    Poynting flux is

        P_inj = (1/2) integral Re(E x H*) . z_hat dA
              = (n_eff / (2 eta0)) * amplitude^2
                * integral (|profile_major|^2 + |profile_minor|^2) dA .

    We resample the *unnormalized* transverse-E pair, evaluate that integral on
    the plane's real cell areas, and scale BOTH packed profiles by
    ``1/sqrt(P_inj_at_unit_scale / power_watts)`` so the injected mode carries
    exactly ``power_watts`` in the engine's own (scalar-H) convention. (The
    field-only L2 normalization the FDE solver applies has arbitrary units, so a
    power normalization here is what makes the launch physically meaningful and
    lets transmission read an absolute fraction.) ``amplitude`` is left at 1.0;
    the whole power scaling lives in the profiles.

    Phase note: for a lossless guided mode both transverse-E components are
    co-real (relative phase 0 or π), so the real signed ``profile``/
    ``profile_minor`` capture the launch exactly; any out-of-phase (quadrature)
    part of the minor-E would need a second carrier and is dropped (a no-op for
    the lossless guided modes this targets).

    **Broadband injection (``num_freqs`` analogue, NUMERICS.md §18.3).** Pass
    ``modes_by_freq`` (``{freq_hz: VectorMode}`` from :func:`solve_modes_by_freq`
    over a :class:`~simupod.plugins.vector_modes.VectorModeSolver`) to inject a
    frequency-dependent full-vector profile across the band; each carrier is
    power-normalized to ``power_watts`` and the engine partition-of-unity-windows
    them. The positional ``mode`` stays the band-centre representative and the
    sign reference. Fewer than two entries is a no-op (single ``mode``).
    """
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if not power_watts > 0.0:
        raise ValueError(f"power_watts must be > 0, got {power_watts}")
    t1_name, t2_name = _TRANSVERSE[axis]
    u_coords = _axis_cell_centers(simulation, t1_name)
    v_coords = _axis_cell_centers(simulation, t2_name)
    if center_um is None:
        center_um = _default_center(simulation, axis)
    # dA from the plane's real transverse cell widths (uniform here, but use the
    # shared quadrature so this stays correct if the plane ever grades).
    dA_m2 = np.outer(_cell_widths(v_coords), _cell_widths(u_coords)) * 1e-12

    def _resample(m):
        """Power-normalized (major, minor) real profiles + major polarization,
        resampled onto this plane — the shared full-vector source readout."""
        fv = vector_modal_fields(
            m, u_coords, v_coords, axis=axis, direction=direction,
            center_um=center_um, thickness_axis=thickness_axis,
        )
        e1, e2 = fv["e1"], fv["e2"]  # transverse-E along t1, t2 ([iv, iu])
        # The MAJOR transverse axis carries the larger transverse-E energy.
        if float(np.sum(np.abs(e1) ** 2)) >= float(np.sum(np.abs(e2) ** 2)):
            e_major, pol_maj, e_minor = e1, "E" + t1_name, e2
        else:
            e_major, pol_maj, e_minor = e2, "E" + t2_name, e1
        if not float(np.sum(np.abs(e_major) ** 2)) > 0.0:
            raise ValueError(
                "the resampled mode profile is identically zero on this plane "
                "— check the mode window vs the simulation transverse extent / "
                "center"
            )
        # Real signed profiles (lossless guided mode -> transverse-E co-real;
        # the real part is exact there). Keep the major/minor RATIO.
        maj = np.real(e_major)
        minr = np.real(e_minor)
        # power_watts normalization in the engine's scalar-H convention (see the
        # docstring P_inj derivation), evaluated AT this mode's n_eff.
        p_unit = (float(m.n_eff) / (2.0 * ETA0)) * float(
            np.sum((maj ** 2 + minr ** 2) * dA_m2)
        )
        if not p_unit > 0.0:
            raise ValueError(
                "modal power integral is non-positive; cannot normalize")
        scale = float(np.sqrt(power_watts / p_unit))
        # C-order [iv*nu + iu] = [cv*nu + cu]
        return (maj * scale).reshape(-1), (minr * scale).reshape(-1), pol_maj

    maj, minr, pol_major = _resample(mode)
    pol_minor = ("E" + t2_name) if pol_major == "E" + t1_name else ("E" + t1_name)

    bb = _broadband_arrays(
        modes_by_freq, _resample, pol_major, maj, central_minor=minr,
    )
    return ModeSource(
        axis=axis,
        direction=direction,
        position_um=position_um,
        polarization=pol_major,
        amplitude=1.0,  # the power scaling lives entirely in the profiles
        n_eff=float(mode.n_eff),
        nu=int(u_coords.size),
        nv=int(v_coords.size),
        profile=tuple(float(v) for v in maj),
        minor_polarization=pol_minor,
        profile_minor=tuple(float(v) for v in minr),
        source_time=source_time,
        **bb,
    )


@dataclass(frozen=True)
class ModeMonitor:
    """A mode-resolved transmission monitor: a 4-tangential ``FieldDftMonitor``
    (add ``.field_monitor`` to the simulation) plus a ``.transmission(data)``
    post-process that overlaps the recorded plane onto ``mode``."""

    field_monitor: FieldDftMonitor
    mode: Mode
    axis: str
    center_um: Optional[Tuple[float, float]] = None
    direction: str = "+"
    thickness_axis: Optional[str] = None
    modes_by_freq: Optional[Mapping[float, Mode]] = None

    @property
    def name(self) -> str:
        return self.field_monitor.name

    def mode_power(
        self,
        data,
        *,
        direction: Optional[str] = None,
        n_eff: Optional[float] = None,
        modes_by_freq: Optional[Mapping[float, Mode]] = None,
        colocate: bool = True,
    ) -> Dict[float, float]:
        """The forward (or backward) modal **power** ``{freq_hz: |a_pm|²/P_mode}``
        on this plane — the actual power carried by ``mode`` through it, in the
        run's (source-spectrum-normalized) units. This is NOT a 0–1 transmission
        on its own; ratio two planes for that (see :func:`transmission`).

        Returns true *power* (``|c|²·P_mode``), not the bare squared amplitude
        ``|c|²``, so that ``P_out / P_in`` is the correct power transmission even
        when the two ports carry **different** modes (e.g. a w1→w2 taper, where the
        per-mode ``P_mode`` differs and must not cancel). For same-mode ratios (a
        uniform-width straight, or a reflection ``-``/``+`` at one plane) the
        ``P_mode`` cancels, so those readings are unchanged. ``data`` is the
        ``SimulationData`` from the run; ``data[self.name]`` is the recorded DFT
        plane. Pass ``modes_by_freq`` (``{freq_hz: Mode}``) to project each
        frequency onto its own per-λ mode instead of the frozen ``self.mode``
        (overrides the monitor's stored ``modes_by_freq`` if any)."""
        da = data[self.name]
        planes: Mapping[str, object] = {
            c: da.sel(component=c) for c in _TANGENTIAL[self.axis]
        }
        mbf = modes_by_freq if modes_by_freq is not None else self.modes_by_freq
        return mode_transmission(
            planes,
            self.mode,
            axis=self.axis,
            direction=direction or self.direction,
            n_eff=n_eff,
            center_um=self.center_um,
            thickness_axis=self.thickness_axis,
            modes_by_freq=mbf,
            power=True,
            colocate=colocate,
        )


def transmission(
    out_monitor: ModeMonitor,
    in_monitor: ModeMonitor,
    data,
    *,
    direction: str = "+",
    n_eff: Optional[float] = None,
    colocate: bool = True,
) -> Dict[float, float]:
    """Mode-resolved power transmission ``{freq_hz: T}`` from ``in_monitor`` to
    ``out_monitor`` — the ratio of modal powers, which cancels the source and
    spectrum normalization so a lossless single-mode straight guide reads
    ``T ≈ 1``. Place ``in_monitor`` just after the source (total-field side) and
    ``out_monitor`` at the device output."""
    p_in = in_monitor.mode_power(data, direction=direction, n_eff=n_eff,
                                 colocate=colocate)
    p_out = out_monitor.mode_power(data, direction=direction, n_eff=n_eff,
                                   colocate=colocate)
    return {f: p_out[f] / p_in[f] for f in p_out if f in p_in}


def mode_monitor(
    simulation,
    mode: Mode,
    *,
    axis: str,
    position_um: float,
    freqs_hz,
    name: str,
    direction: str = "+",
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[str] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
) -> ModeMonitor:
    """Build a :class:`ModeMonitor` (a 4-tangential ``FieldDftMonitor`` on the
    ``axis`` plane at ``position_um`` + a transmission post-process onto
    ``mode``). Add ``.field_monitor`` to the simulation's monitors, run, then
    call ``.transmission(data)``. ``thickness_axis`` is the slab-normal axis
    (pass e.g. ``"z"`` for non-x propagation so the overlap mode is not rotated
    90 degrees); ``None`` keeps the legacy mapping. See :func:`mode_source`."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    idx = _AXIS_IDX[axis]
    # A plane carrying mixed Yee offsets (Ex/Ey at integer, Hx/Hy at half-cell
    # along the normal) is rejected unless every component snaps to one cell;
    # placing the plane at (k+0.25)*dl does that (NUMERICS §12). Snap on a
    # uniform grid; leave as-is otherwise (the engine validates).
    dl = getattr(simulation.grid, "dl_um", None)
    if dl:
        k = round(position_um / dl - 0.25)
        position_um = (k + 0.25) * dl
    size = list(simulation.size_um)
    size[idx] = 0.0  # a plane normal to `axis`
    center = [s / 2.0 for s in simulation.size_um]
    center[idx] = position_um
    fm = FieldDftMonitor(
        name=name,
        center_um=tuple(center),
        size_um=tuple(size),
        fields=_TANGENTIAL[axis],
        freqs_hz=tuple(freqs_hz),
    )
    return ModeMonitor(
        field_monitor=fm,
        mode=mode,
        axis=axis,
        center_um=center_um,
        direction=direction,
        thickness_axis=thickness_axis,
        modes_by_freq=modes_by_freq,
    )


def solve_modes_by_freq(
    solver: Any,
    freqs_hz: Iterable[float],
    *,
    mode_index: int = 0,
    **solve_kwargs: Any,
) -> Dict[float, Mode]:
    """Solve the FDE eigenmode at each frequency and return ``{freq_hz: Mode}``,
    ready to hand to :func:`mode_monitor` (or :class:`ModeMonitor`) as
    ``modes_by_freq`` — the readout-side analogue of Tidy3D's ``num_freqs``.

    A single frozen mode is overlapped per frequency by default; with
    ``modes_by_freq`` each recorded DFT frequency is instead projected onto a
    mode solved AT that frequency, which matters when the modal profile / n_eff
    drifts across a wide band (the same motivation as a broadband mode SOURCE,
    see :func:`mode_source`). This helper automates the per-frequency solve that
    fills that map.

    Parameters
    ----------
    solver:
        A :class:`~simupod.plugins.modes.ModeSolver` or
        :class:`~simupod.plugins.vector_modes.VectorModeSolver` carrying the
        waveguide cross-section. It is re-solved on the SAME geometry at each
        frequency via ``solver.at_wavelength(C0 / f * 1e6)`` (the eps is shared
        by reference), so the cross-section is rasterized once.
    freqs_hz:
        The monitor frequencies (Hz). Typically the same tuple passed as the
        ``FieldDftMonitor.freqs_hz`` / ``mode_monitor(freqs_hz=...)``.
    mode_index:
        Which solved mode to keep per frequency (0 = fundamental, the
        descending-``n_eff`` order ``solve`` returns). The branch must support
        the same mode at every frequency.
    **solve_kwargs:
        Forwarded to ``solver.solve`` (e.g. ``polarization="TM"`` for the
        scalar solver, ``num_modes=...``). ``num_modes`` is bumped to at least
        ``mode_index + 1`` so the requested mode is available.

    Returns
    -------
    dict[float, Mode]
        ``{freq_hz: Mode}`` in the input order. Cost: one CPU FDE solve per
        frequency.
    """
    freqs = [float(f) for f in freqs_hz]
    if not freqs:
        raise ValueError("freqs_hz must be non-empty")
    if mode_index < 0:
        raise ValueError(f"mode_index must be >= 0, got {mode_index}")
    num_modes = max(int(solve_kwargs.pop("num_modes", 1)), mode_index + 1)
    out: Dict[float, Mode] = {}
    for f in freqs:
        if not f > 0.0:
            raise ValueError(f"frequencies must be > 0 Hz, got {f}")
        wavelength_um = C0 / f * 1e6
        modes = solver.at_wavelength(wavelength_um).solve(
            num_modes=num_modes, **solve_kwargs
        )
        if mode_index >= len(modes):
            raise ValueError(
                f"requested mode_index {mode_index} but the solver returned only "
                f"{len(modes)} mode(s) at {f:.4g} Hz "
                f"({wavelength_um:.4f} um) — the waveguide may not support it "
                "across the whole band"
            )
        out[f] = modes[mode_index]
    return out
