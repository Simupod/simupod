"""Directional-power mode-overlap: a recorded field plane -> mode-resolved T.

This is the Phase-2 Track-B *mode-monitor transmission* post-processor. Given a
field plane recorded by an FDTD run (the tangential ``E`` and ``H`` DataArrays
on a plane whose normal is the waveguide's propagation axis) and a frozen FDE
:class:`~photonhub.plugins.modes.Mode`, it computes the **forward (or backward)
power transmission** ``T(f)`` into that single mode. There is **NO S-matrix**
here — this is a one-mode-at-a-time projection.

Physics / method
================
**Directional power overlap.** For a monitor plane with outward normal ``n_hat``
along the propagation axis, the complex modal-amplitude coefficient of the
simulated field on the mode is (``e^{-i omega t}`` time convention)

    a_pm = (1/4) * integral_A [ E_sim x h_mode*  +  e_mode* x H_sim ] . n_hat dA

and the power transmission into the (normalized) mode is

    T = |a_pm|^2 / P_mode^2 ,
    P_mode = (1/2) * integral_A Re( e_mode x h_mode* ) . n_hat dA .

NOTE ON THE DENOMINATOR (deviation from the handoff brief): the brief wrote
``T = |a_pm|^2 / P_mode``, but with the ``1/4`` overlap coefficient above the
*self*-overlap evaluates to ``a_pm = P_mode`` exactly (substitute
``E_sim=e_mode, H_sim=h_mode``: both cross terms equal ``2*P_mode * (1/4)``).
``|a_pm|^2 / P_mode`` would then give ``P_mode`` rather than the required
``T=1``. The self-consistent power ratio is ``T = |a_pm|^2 / P_mode^2`` — i.e.
``a_pm`` is the *unnormalized* coefficient and the normalized modal amplitude is
``a_pm / P_mode``. We implement that (so self-overlap == 1 exactly); see the
test suite which pins it.

Carrying *both* the simulated ``E`` and ``H`` is what separates forward from
backward power: a clean single-mode field travelling along ``+n_hat`` reads
``T_forward ~= 1`` and ``T_backward ~= 0``; reverse the field's propagation and
the two swap. ``direction="-"`` selects the backward mode by flipping the modal
transverse ``H`` (``h_mode -> -h_mode``), equivalently picking ``a_minus``.

**Scalar-limit H reconstruction (APPROXIMATION).** The frozen FDE solver returns
only a *scalar* transverse ``E`` profile (the major component ``Ex`` for TE,
``Ey`` for TM) and a real ``n_eff`` — it carries no ``H`` and no minor-component
``E``. We therefore reconstruct the modal transverse ``H`` from the scalar mode
in the **quasi-TEM / weakly-guided limit**:

    e_mode  = major transverse E unit vector * scalar_profile   (minor E := 0),
    h_mode  = (n_eff / eta0) * ( z_hat x e_mode ) ,

with ``eta0`` the vacuum wave impedance and ``z_hat`` the propagation axis. This
is *exact* in the weakly-guided limit and *approximate* for high-contrast SOI
(it drops the longitudinal ``E_z``/``H_z`` and the minor transverse components).
That error is accepted for the MVP and is pinned later by a Tier-2b leakage
gate. With this reconstruction, ``e_mode x h_mode*`` is purely along ``n_hat``
and ``P_mode = (n_eff / (2 eta0)) * integral |profile|^2 dA``.

**Area element.** ``dA`` is taken from the plane's *real* transverse coordinate
spacings (centered-difference cell widths), so graded / non-uniform meshes are
handled correctly — no uniform-spacing assumption.

**Scope.** Fundamental mode, looped over the monitor's frequencies. By default
one scalar mode profile (+ its ``n_eff``) is used for every frequency (the frozen
mode); pass ``modes_by_freq`` to project each frequency onto its OWN solved mode
(profile + ``n_eff``), matching Tidy3D's per-frequency ``ModeMonitor`` and
recovering the waveguide dispersion the frozen mode drops. A scalar per-frequency
``n_eff`` override is also accepted.

Dependency-light: numpy + the xarray DataArrays the rest of PhotonHub already
produces. No matplotlib, no engine calls.
"""

from __future__ import annotations

from typing import Dict, Literal, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from .modes import Mode

__all__ = [
    "ETA0",
    "mode_amplitude",
    "mode_transmission",
    "resample_profile",
    "modal_fields",
    "vector_modal_fields",
]

#: Vacuum wave impedance (ohms), eta0 = sqrt(mu0 / eps0) = mu0 * c0.
ETA0: float = 376.730313668

Axis = Literal["x", "y", "z"]
Direction = Literal["+", "-"]

# For a propagation axis, the (transverse_axis_1, transverse_axis_2) such that
# axis_1 x axis_2 = +propagation_axis (right-handed). z_hat x t1 = t2.
_TRANSVERSE: Dict[str, Tuple[str, str]] = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


def _cell_widths(coords: np.ndarray) -> np.ndarray:
    """Per-sample cell widths for a 1-D set of (possibly non-uniform) sample
    coordinates, via centered differences with half-cells at the ends. The sum
    equals the span plus one mean end-cell, i.e. a midpoint quadrature weight.

    For a single sample (a degenerate 1-cell transverse extent) the width is 1.0
    so the "integral" reduces to that sample's value (a line/point monitor)."""
    c = np.asarray(coords, dtype=np.float64)
    n = c.size
    if n == 1:
        return np.array([1.0])
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.abs(np.diff(edges))


def _colocate_to_node(a: np.ndarray, axis: int) -> np.ndarray:
    """Average a +½-cell Yee-staggered field component onto the cell NODE along
    ``axis`` (``node[j] = ½(a[j-1] + a[j])``; ``a[-1] ≡ 0`` since a guided mode is
    ~0 at the transverse boundary).

    The engine's DFT monitor emits each component at its own Yee node in
    *cell-index* space (``grid.h`` ``yee_offset``: E_t1 is +½ in t1, E_t2 +½ in
    t2, H_t1 +½ in t2, H_t2 +½ in t1), so the recorded E and H tangential
    components are physically staggered by half a cell. Combining them in the
    overlap cross-products without first interpolating each to a COMMON point is a
    FIRST-ORDER error; co-locating restores SECOND-ORDER accuracy (Oskooi &
    Johnson, *Comp. Phys. Comm.* 181, 687 (2010); MEEP issues #1470/#1773). This
    is what Lumerical (monitor spatial-interpolation, default "nearest mesh cell")
    and Tidy3D (``ModeMonitor(colocate=True)``, the default) do before the
    two-term mode overlap. The collocated FDE mode needs no shift."""
    prev = np.roll(a, 1, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = 0
    prev[tuple(idx)] = 0.0
    return 0.5 * (prev + a)


def resample_profile(
    field: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
) -> np.ndarray:
    """Separable bilinear resample of ``field[iy, ix]`` (defined on the centered
    1-D grids ``src_x``/``src_y``) onto the destination coordinates
    ``dst_x``/``dst_y``, zero-filled outside the source window.

    Generalizes ``benchmarks/waveguide/run_waveguide.py:_resample`` — numpy-only
    (two passes of :func:`numpy.interp`, x then y). Returns a ``(dst_y.size,
    dst_x.size)`` array indexed ``[iy, ix]``."""
    field = np.asarray(field, dtype=np.float64)
    src_x = np.asarray(src_x, dtype=np.float64)
    src_y = np.asarray(src_y, dtype=np.float64)
    dst_x = np.asarray(dst_x, dtype=np.float64)
    dst_y = np.asarray(dst_y, dtype=np.float64)

    tmp = np.empty((field.shape[0], dst_x.size))
    for j in range(field.shape[0]):
        tmp[j] = np.interp(dst_x, src_x, field[j], left=0.0, right=0.0)
    out = np.empty((dst_y.size, dst_x.size))
    for i in range(dst_x.size):
        out[:, i] = np.interp(dst_y, src_y, tmp[:, i], left=0.0, right=0.0)
    return out


def modal_fields(
    mode: Mode,
    t1_um: np.ndarray,
    t2_um: np.ndarray,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Tuple[float, float] = (0.0, 0.0),
    thickness_axis: Optional[Axis] = None,
) -> Dict[str, np.ndarray]:
    """Assemble the scalar-limit modal transverse fields on a monitor plane.

    The mode's scalar profile is resampled onto the plane's transverse grid
    ``(t1_um, t2_um)`` (the two in-plane axes for ``axis``, in their natural
    Yee order — see :func:`mode_transmission`). The major transverse ``E`` carries
    the whole profile, the minor transverse ``E`` is zero (scalar limit), and the
    transverse ``H`` is ``(n_eff/eta0) * (z_hat x e_mode)``; ``direction="-"``
    flips ``H`` to select the backward mode.

    Parameters
    ----------
    mode:
        The frozen FDE :class:`~photonhub.plugins.modes.Mode`. Its ``.field`` is
        the major transverse-E component (``Ex`` for TE, ``Ey`` for TM).
    t1_um, t2_um:
        The plane's two transverse coordinate axes (microns), in the order
        ``_TRANSVERSE[axis]`` (so ``t1 x t2 = +axis``).
    axis:
        Propagation axis ``"x"``/``"y"``/``"z"``.
    direction:
        ``"+"`` forward (default) or ``"-"`` backward.
    n_eff:
        Optional override for the modal index used in the ``H`` reconstruction
        (per-frequency dispersion). Defaults to ``mode.n_eff``.
    center_um:
        ``(t1, t2)`` location of the waveguide axis in the plane's coordinate
        frame (microns). The mode profile (centered at its own origin) is shifted
        here before resampling. Defaults to the plane origin.
    thickness_axis:
        The simulation axis along the guide's slab thickness (the mode's HEIGHT
        / ``dl_y`` direction); must be one of the two transverse axes for
        ``axis``. The mode's WIDTH (``dl_x``) is mapped to the OTHER transverse
        axis. ``None`` (default) keeps the legacy mapping ``width->t1,
        height->t2`` — correct only when the thickness lies on the second
        transverse axis (e.g. x-propagation with a z-normal slab). For
        y-propagation of a z-normal slab the thickness is the FIRST transverse
        axis, so pass ``thickness_axis="z"`` to orient the mode correctly (else
        the profile comes out rotated 90 degrees).

    Returns
    -------
    dict
        Keys ``"e1"``, ``"e2"`` (transverse-E components along ``t1``/``t2``),
        ``"h1"``, ``"h2"`` (transverse-H), each a ``(t2.size, t1.size)`` array.
        The major-E component is whichever of ``t1``/``t2`` is the mode's major
        axis; the other E component is all zeros.
    """
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    a1, a2 = _TRANSVERSE[axis]  # the two transverse axis NAMES (a1 x a2 = +axis)
    if thickness_axis is None:
        thickness_axis = a2  # legacy: slab thickness on the 2nd transverse axis
    if thickness_axis not in (a1, a2):
        raise ValueError(
            f"thickness_axis {thickness_axis!r} must be a transverse axis "
            f"({a1!r} or {a2!r}) for propagation axis {axis!r}")
    # Physically the mode's WIDTH (dl_x) lies on the in-plane transverse axis and
    # its HEIGHT (dl_y) on the slab-normal (thickness) axis. Map width -> the
    # non-thickness axis, height -> the thickness axis (NOT the fixed x->t1,
    # y->t2, which is right only when the thickness happens to be a2).
    width_axis = a1 if thickness_axis == a2 else a2

    neff = float(mode.n_eff if n_eff is None else n_eff)

    # Mode's own centered real-space coords (microns), matching field_dataarray.
    ny, nx = mode.field.shape
    w_coords = (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um  # mode width axis
    h_coords = (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um  # mode height axis
    t1c = np.asarray(t1_um, dtype=np.float64)
    t2c = np.asarray(t2_um, dtype=np.float64)

    if width_axis == a1:  # width -> t1, height -> t2 (legacy orientation)
        wc = w_coords + center_um[0]
        hc = h_coords + center_um[1]
        profile = resample_profile(mode.field, wc, hc, t1c, t2c)  # [i_t2, i_t1]
    else:  # width -> t2, height -> t1 (e.g. y-propagation, thickness on a1)
        wc = w_coords + center_um[1]
        hc = h_coords + center_um[0]
        # width(mode-x)->t2, height(mode-y)->t1; transpose to [i_t2, i_t1].
        profile = resample_profile(mode.field, wc, hc, t2c, t1c).T

    # Major transverse-E axis: TE's major (mode Ex) lies along the WIDTH axis,
    # TM's major (mode Ey) along the HEIGHT = thickness axis. In the scalar limit
    # the minor transverse E is zero.
    major_axis = width_axis if mode.polarization != "TM" else thickness_axis
    major_is_t1 = major_axis == a1
    e1 = profile if major_is_t1 else np.zeros_like(profile)
    e2 = np.zeros_like(profile) if major_is_t1 else profile

    # h = (n_eff/eta0) * (z_hat x e), z_hat = +axis. With e = (e1, e2) in the
    # right-handed (t1, t2) frame: z_hat x (e1 t1_hat + e2 t2_hat)
    #   = e1 (z_hat x t1_hat) + e2 (z_hat x t2_hat) = e1 t2_hat - e2 t1_hat.
    sign = 1.0 if direction == "+" else -1.0
    scale = sign * neff / ETA0
    h1 = -scale * e2
    h2 = scale * e1
    return {"e1": e1, "e2": e2, "h1": h1, "h2": h2}


def _resample_complex(
    field: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
) -> np.ndarray:
    """Like :func:`resample_profile` but for a complex ``field`` — real and
    imaginary parts are resampled independently (the bilinear interpolation is
    linear, so this preserves the per-point complex value exactly on the
    source grid and interpolates the relative phase between components)."""
    field = np.asarray(field)
    re = resample_profile(field.real, src_x, src_y, dst_x, dst_y)
    if np.iscomplexobj(field) and np.any(field.imag):
        im = resample_profile(field.imag, src_x, src_y, dst_x, dst_y)
        return re + 1j * im
    return re.astype(np.complex128)


def vector_modal_fields(
    mode,
    t1_um: np.ndarray,
    t2_um: np.ndarray,
    *,
    axis: Axis,
    direction: Direction = "+",
    center_um: Tuple[float, float] = (0.0, 0.0),
    thickness_axis: Optional[Axis] = None,
) -> Dict[str, np.ndarray]:
    """Assemble the FULL-VECTOR transverse fields of a
    :class:`~photonhub.plugins.vector_modes.VectorMode` on a monitor/injection
    plane — the full-vector analogue of :func:`modal_fields`.

    Unlike :func:`modal_fields` (which carries one scalar profile in the major-E
    component and reconstructs ``H`` in the scalar limit), this resamples the
    mode's *actual* transverse-E pair ``(ex, ey)`` AND transverse-H pair
    ``(hx, hy)`` onto the plane, preserving their true component RATIO and
    relative phase. The mode's own width/height axes are mapped to the plane's
    ``(t1, t2)`` exactly as :func:`modal_fields` does (via ``thickness_axis``),
    and ``direction="-"`` flips ``H`` to select the backward mode.

    Parameters mirror :func:`modal_fields`. ``mode`` is a ``VectorMode`` (the
    six complex ``(ny, nx)`` component arrays ``ex, ey, ez, hx, hy, hz`` indexed
    ``[iy, ix]``). Returns a dict with ``"e1"``, ``"e2"``, ``"h1"``, ``"h2"``
    (transverse components along ``t1``/``t2``), each a ``(t2.size, t1.size)``
    complex array — same layout as :func:`modal_fields`.
    """
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    a1, a2 = _TRANSVERSE[axis]
    if thickness_axis is None:
        thickness_axis = a2  # legacy: slab thickness on the 2nd transverse axis
    if thickness_axis not in (a1, a2):
        raise ValueError(
            f"thickness_axis {thickness_axis!r} must be a transverse axis "
            f"({a1!r} or {a2!r}) for propagation axis {axis!r}")
    width_axis = a1 if thickness_axis == a2 else a2

    # The mode's own centered real-space coords (microns) — width along mode-x
    # (dl_x_um, carried by ex) and height along mode-y (dl_y_um, carried by ey).
    ny, nx = mode.ex.shape
    w_coords = (np.arange(nx) - (nx - 1) / 2.0) * mode.dl_x_um  # mode width / x
    h_coords = (np.arange(ny) - (ny - 1) / 2.0) * mode.dl_y_um  # mode height / y
    t1c = np.asarray(t1_um, dtype=np.float64)
    t2c = np.asarray(t2_um, dtype=np.float64)

    def to_plane(mode_field: np.ndarray) -> np.ndarray:
        """Resample a mode-frame [iy, ix] field onto the plane [i_t2, i_t1]."""
        if width_axis == a1:  # mode-x -> t1, mode-y -> t2 (legacy)
            wc = w_coords + center_um[0]
            hc = h_coords + center_um[1]
            return _resample_complex(mode_field, wc, hc, t1c, t2c)
        # mode-x -> t2, mode-y -> t1; resample then transpose to [i_t2, i_t1].
        wc = w_coords + center_um[1]
        hc = h_coords + center_um[0]
        return _resample_complex(mode_field, wc, hc, t2c, t1c).T

    # The mode's x-field (ex/hx) lies along the WIDTH axis, the y-field (ey/hy)
    # along the HEIGHT (= thickness) axis. Route each to t1/t2 accordingly.
    ex_p, ey_p = to_plane(mode.ex), to_plane(mode.ey)
    hx_p, hy_p = to_plane(mode.hx), to_plane(mode.hy)
    if width_axis == a1:  # width(mode-x) -> t1, height(mode-y) -> t2
        e1, e2, h1, h2 = ex_p, ey_p, hx_p, hy_p
    else:                 # width(mode-x) -> t2, height(mode-y) -> t1
        e1, e2, h1, h2 = ey_p, ex_p, hy_p, hx_p

    if direction == "-":  # backward mode: H -> -H
        h1, h2 = -h1, -h2
    return {"e1": e1, "e2": e2, "h1": h1, "h2": h2}


def _plane_component(
    fields: Mapping[str, xr.DataArray],
    name: str,
    freq_hz: Optional[float],
    t1: str,
    t2: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull a tangential component DataArray (e.g. ``"Ey"``), drop the singleton
    normal axis and any freq/component dims, and return it as a 2-D complex
    ``[i_t2, i_t1]`` array plus its ``(t1_coords, t2_coords)`` in microns."""
    da = fields[name]
    if "component" in da.dims:
        da = da.sel(component=name) if name in list(da.coords.get("component", [])) \
            else da.squeeze("component", drop=True)
    if "f" in da.dims:
        da = da.sel(f=freq_hz, method="nearest") if freq_hz is not None \
            else da.isel(f=0)
    # Drop the (singleton) normal axis and any other length-1 dims, keeping t1/t2.
    da = da.squeeze(drop=True)
    if set(da.dims) != {t1, t2}:
        raise ValueError(
            f"component {name!r}: after reduction dims are {tuple(da.dims)}, "
            f"expected the two transverse axes {{{t1!r}, {t2!r}}}")
    # Orient as [i_t2, i_t1] so it matches modal_fields' (t2.size, t1.size).
    da = da.transpose(t2, t1)
    vals = np.asarray(da.values)
    c1 = np.asarray(da.coords[t1].values, dtype=np.float64)
    c2 = np.asarray(da.coords[t2].values, dtype=np.float64)
    return vals, c1, c2


def _overlap_terms(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode: Mode,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    colocate: bool = True,
) -> Dict[float, Tuple[complex, float]]:
    """Per-frequency directional-power overlap terms ``{f: (a_pm, P_mode)}`` for a
    recorded plane projected onto ``mode`` — the shared kernel behind
    :func:`mode_amplitude` (``c = a_pm/P_mode``), :func:`mode_transmission`
    (``|c|² = |a_pm|²/P_mode²``) and the power readout (``|a_pm|²/P_mode``).
    ``a_pm`` is the unnormalized complex coefficient, ``P_mode`` the mode's own
    power on the plane. See :func:`mode_transmission` for the argument schema."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    t1, t2 = _TRANSVERSE[axis]
    e1_name, e2_name = f"E{t1}", f"E{t2}"
    h1_name, h2_name = f"H{t1}", f"H{t2}"
    for key in (e1_name, e2_name, h1_name, h2_name):
        if key not in sim_plane_fields:
            raise ValueError(
                f"sim_plane_fields missing {key!r}; for axis={axis!r} need the "
                f"tangential components {e1_name!r},{e2_name!r},{h1_name!r},"
                f"{h2_name!r}")

    # Determine the set of frequencies from the first E component.
    ref = sim_plane_fields[e1_name]
    if "f" in getattr(ref, "dims", ()):  # DataArray with a freq axis
        freqs = [float(f) for f in np.asarray(ref.coords["f"].values)]
    else:
        freqs = [None]  # plane carries a single (frequencyless) snapshot

    out: Dict[float, Tuple[complex, float]] = {}
    for f in freqs:
        Es1, c1, c2 = _plane_component(sim_plane_fields, e1_name, f, t1, t2)
        Es2, _, _ = _plane_component(sim_plane_fields, e2_name, f, t1, t2)
        Hs1, _, _ = _plane_component(sim_plane_fields, h1_name, f, t1, t2)
        Hs2, _, _ = _plane_component(sim_plane_fields, h2_name, f, t1, t2)

        if colocate:
            # Yee co-location (§ _colocate_to_node): shift each staggered sim
            # component to the cell node so the two-term overlap is 2nd-order.
            # Arrays are [i_t2, i_t1] (t1 = last axis, t2 = axis 0); offsets from
            # grid.h yee_offset: E_t1 +½t1, E_t2 +½t2, H_t1 +½t2, H_t2 +½t1.
            Es1 = _colocate_to_node(Es1, -1)
            Es2 = _colocate_to_node(Es2, 0)
            Hs1 = _colocate_to_node(Hs1, 0)
            Hs2 = _colocate_to_node(Hs2, -1)

        # Area element from the plane's real (possibly graded) coord spacings.
        w1 = _cell_widths(c1)            # along t1
        w2 = _cell_widths(c2)            # along t2
        dA = np.outer(w2, w1)            # [i_t2, i_t1], matches field arrays

        cen = center_um
        if cen is None:
            cen = (float(np.mean(c1)), float(np.mean(c2)))
        use_mode, use_neff = mode, n_eff
        if modes_by_freq and f is not None:
            # per-λ: project this frequency onto its OWN solved mode (profile +
            # n_eff), matching Tidy3D's per-frequency ModeMonitor decomposition.
            key = min(modes_by_freq, key=lambda k: abs(k - f))
            use_mode, use_neff = modes_by_freq[key], None
        if hasattr(use_mode, "hx"):
            # Full-vector mode: project with the mode's TRUE transverse H, not the
            # scalar-limit (n_eff/eta0)·(z_hat x e). This is the grid-consistent
            # "smooth readout" path — see benchmarks/tidy3d/SMOOTH_CONVERGENCE_PLAN.md
            # (issue #34). n_eff is intrinsic to the vector mode, so use_neff is
            # not applicable here.
            m = vector_modal_fields(use_mode, c1, c2, axis=axis,
                                    direction=direction, center_um=cen,
                                    thickness_axis=thickness_axis)
        else:
            m = modal_fields(use_mode, c1, c2, axis=axis, direction=direction,
                             n_eff=use_neff, center_um=cen,
                             thickness_axis=thickness_axis)
        e1, e2, h1, h2 = m["e1"], m["e2"], m["h1"], m["h2"]

        # n_hat-component of a cross product of transverse vectors
        # (a1, a2) x (b1, b2) = (a1 b2 - a2 b1) n_hat.
        # a_pm = (1/4) integral [ E_sim x h_mode* + e_mode* x H_sim ] . n_hat dA
        cross_eh = (Es1 * np.conj(h2) - Es2 * np.conj(h1))      # E_sim x h*
        cross_he = (np.conj(e1) * Hs2 - np.conj(e2) * Hs1)      # e* x H_sim
        a_pm = 0.25 * np.sum((cross_eh + cross_he) * dA)

        # P_mode = (1/2) integral Re( e_mode x h_mode* ) . n_hat dA.
        p_density = np.real(e1 * np.conj(h2) - e2 * np.conj(h1))
        p_mode = 0.5 * np.sum(p_density * dA)

        if p_mode == 0.0:
            raise ValueError(
                "P_mode is zero — the resampled mode has no power on this plane "
                "(check the mode window vs the plane extent and center_um).")
        out[f if f is not None else 0.0] = (complex(a_pm), float(p_mode))
    return out


def mode_amplitude(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode: Mode,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    colocate: bool = True,
) -> Dict[float, complex]:
    """Mode-resolved **complex** modal amplitude ``c(f)`` of a recorded plane.

    This is the COMPLEX coefficient that :func:`mode_transmission` squares to a
    power. Per frequency on the plane it computes the directional-power overlap

        a_pm   = (1/4) integral [ E_sim x h_mode* + e_mode* x H_sim ] . n_hat dA
        P_mode = (1/2) integral Re( e_mode x h_mode* ) . n_hat dA
        c      = a_pm / P_mode

    in the ``e^{-i omega t}`` convention, with the scalar-limit modal ``H``
    (see the module docstring). The normalization by ``P_mode`` makes a clean
    single-mode forward self-overlap read ``c == 1`` exactly (so ``|c|^2 == T``,
    the power transmission). Crucially ``c`` retains the **phase** of the modal
    projection — it advances by ``e^{-i beta L}`` along a straight guide — which
    is exactly what an S-matrix assembler needs (``S_ij = b_i / a_j``).

    The amplitude is *directional*: ``direction="+"`` projects onto the forward
    mode, ``direction="-"`` onto the backward one. A pure forward wave reads a
    near-unit forward ``c`` and a near-zero backward ``c``, and vice versa — this
    is what separates incident (forward) from scattered (backward) at a port.

    Parameters mirror :func:`mode_transmission`; see it for the
    ``sim_plane_fields`` schema and the per-argument documentation.

    Returns
    -------
    dict[float, complex]
        ``{freq_hz: c}`` for every frequency on the plane (frequencyless planes
        key on ``0.0``). ``c`` is the normalized complex modal amplitude.
    """
    return {
        f: complex(a_pm / p_mode)
        for f, (a_pm, p_mode) in _overlap_terms(
            sim_plane_fields, mode, axis=axis, direction=direction, n_eff=n_eff,
            center_um=center_um, thickness_axis=thickness_axis,
            modes_by_freq=modes_by_freq, colocate=colocate).items()
    }


def mode_transmission(
    sim_plane_fields: Mapping[str, xr.DataArray],
    mode: Mode,
    *,
    axis: Axis,
    direction: Direction = "+",
    n_eff: Optional[float] = None,
    center_um: Optional[Tuple[float, float]] = None,
    thickness_axis: Optional[Axis] = None,
    modes_by_freq: Optional[Mapping[float, Mode]] = None,
    power: bool = False,
    colocate: bool = True,
) -> Dict[float, float]:
    """Mode-resolved power transmission ``T(f)`` of a recorded plane onto ``mode``.

    Computes, per frequency on the plane, the directional-power overlap

        a_pm   = (1/4) integral [ E_sim x h_mode* + e_mode* x H_sim ] . n_hat dA
        T      = |a_pm|^2 / P_mode^2 ,           (power=False, default)
        P_mode = (1/2) integral Re( e_mode x h_mode* ) . n_hat dA

    in the ``e^{-i omega t}`` convention, with the scalar-limit modal ``H``
    (see the module docstring; the ``P_mode^2`` denominator — not ``P_mode`` —
    is the squared NORMALISED amplitude ``|c|^2``, so a clean single-mode
    self-overlap reads ``T == 1``).
    ``direction="+"`` returns forward T, ``direction="-"`` backward T.

    ``power=True`` instead returns the actual modal **power** ``|a_pm|^2 / P_mode``
    (= ``|c|^2 * P_mode``). Use this when ratioing two planes whose modes may
    DIFFER (e.g. a w1→w2 taper): ``P_out / P_in`` is then the true power
    transmission. The bare ``|c|^2`` (power=False) drops each port's ``P_mode``,
    so its ratio is only correct when both ports carry the SAME mode (it cancels);
    for unequal-width ports it is wrong (the historical taper-parity bug).

    This is exactly ``|c|^2`` of the complex amplitude from
    :func:`mode_amplitude` — use that function when you need the phase (e.g. for
    an S-matrix). Behaviour here is unchanged (back-compatible).

    Parameters
    ----------
    sim_plane_fields:
        Mapping from component name to its plane DataArray, supplying the two
        tangential ``E`` and two tangential ``H`` components for ``axis``:

        * ``axis="z"`` -> keys ``"Ex"``, ``"Ey"``, ``"Hx"``, ``"Hy"``;
        * ``axis="x"`` -> keys ``"Ey"``, ``"Ez"``, ``"Hy"``, ``"Hz"``;
        * ``axis="y"`` -> keys ``"Ez"``, ``"Ex"``, ``"Hz"``, ``"Hx"``.

        Each is an xarray ``DataArray`` in µm coords (a single-plane ``field_dft``
        slice: dims like ``('f','component','z','y','x')`` with a singleton normal
        axis are reduced automatically; a plain 2-D ``(t2, t1)`` DataArray also
        works). The two transverse axes are ``_TRANSVERSE[axis]``.
    mode:
        The frozen FDE :class:`~photonhub.plugins.modes.Mode` to project onto.
    axis:
        Propagation axis / plane normal, ``"x"``/``"y"``/``"z"``.
    direction:
        ``"+"`` forward (default) or ``"-"`` backward.
    n_eff:
        Optional override for the modal index in the ``H`` reconstruction.
    center_um:
        ``(t1, t2)`` location of the waveguide axis in the plane's coordinate
        frame (microns). If ``None`` (default) the plane's transverse coordinate
        midpoints are used, i.e. the mode is centered on the monitor.
    thickness_axis:
        Simulation axis along the guide's slab thickness; forwarded to
        :func:`modal_fields` to orient the mode (pass the slab normal, e.g.
        ``"z"``, for any non-x propagation — see that function). ``None`` keeps
        the legacy thickness-on-second-transverse-axis mapping.
    modes_by_freq:
        Optional ``{freq_hz: Mode}`` map. When given, each plane frequency is
        projected onto the mode whose key is nearest that frequency (using that
        mode's own profile *and* ``n_eff``), instead of the single frozen
        ``mode`` — the per-λ mode solve. ``mode`` is still required (used as the
        fallback for any frequencyless plane).

    Returns
    -------
    dict[float, float]
        ``{freq_hz: T}`` for every frequency present on the plane (real, >= 0;
        ``~1`` for a clean single-mode forward field, ``~0`` for the opposite
        direction).
    """
    terms = _overlap_terms(
        sim_plane_fields, mode, axis=axis, direction=direction, n_eff=n_eff,
        center_um=center_um, thickness_axis=thickness_axis,
        modes_by_freq=modes_by_freq, colocate=colocate,
    )
    if power:
        return {f: float(np.abs(a_pm) ** 2 / p_mode)
                for f, (a_pm, p_mode) in terms.items()}
    return {f: float(np.abs(a_pm) ** 2 / p_mode ** 2)
            for f, (a_pm, p_mode) in terms.items()}
