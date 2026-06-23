"""Parametric PIC component library — the "~one line per component" sugar
layer on top of the geometry primitives in :mod:`simupod.components.structures`.

Each builder returns a small :class:`Component` bundling the emitted
:class:`~simupod.Structure` geometry with its :class:`Port` s — the planes
where a mode source / mode monitor will later attach. Builders are
*position-agnostic*: they take a ``center_um`` and place geometry relative to
it, so the caller positions the device inside their (corner-origin) domain.

Coordinate convention (defaults): in-plane propagation runs along ``axis``
(default ``"x"``), the slab thickness runs along ``thickness_axis`` (default
``"z"``), and the remaining in-plane axis carries the waveguide width. This is
the standard SOI strip-waveguide layout.

>>> import simupod as ph
>>> from simupod.library import straight, ring
>>> wg = straight(length_um=10.0)                 # one Box + 2 ports
>>> res = ring(radius_um=5.0, gap_um=0.2)         # ring Cylinder + bus Box + 2 ports
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .components.base import AxisName, Vec3Um
from .components.structures import Box, Cylinder, Medium, PolySlab, Structure

__all__ = [
    "Port",
    "Component",
    "SILICON",
    "straight",
    "bend",
    "taper",
    "crossing",
    "coupler",
    "ring",
]

# Default material/geometry for the SOI strip platform (NUMERICS.md / Phase-2
# MVP). n ~= 3.5 silicon core, 220 nm slab, 450 nm strip width.
SILICON = Medium(permittivity=12.25)
DEFAULT_WIDTH_UM = 0.45
DEFAULT_THICKNESS_UM = 0.22

_AXES: Tuple[AxisName, AxisName, AxisName] = ("x", "y", "z")
_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Port:
    """A plane where a mode source / mode monitor attaches.

    ``axis`` is the local propagation axis at the port; ``width_um`` is the
    waveguide width there (sizes the transverse mode window).
    """

    name: str
    center_um: Tuple[float, float, float]
    axis: str
    width_um: float


@dataclass(frozen=True)
class Component:
    """The geometry a builder emits plus the ports it exposes."""

    structures: Tuple[Structure, ...]
    ports: Tuple[Port, ...]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _third_axis(a: AxisName, b: AxisName) -> AxisName:
    """The axis that is neither ``a`` nor ``b``."""
    for ax in _AXES:
        if ax != a and ax != b:
            return ax
    raise ValueError(f"could not find third axis distinct from {a!r}, {b!r}")


def _vec(center: Vec3Um, **offsets: float) -> Tuple[float, float, float]:
    """``center`` shifted by per-axis offsets given as ``x=``/``y=``/``z=``."""
    out = list(center)
    for ax, d in offsets.items():
        out[_INDEX[ax]] += d
    return (out[0], out[1], out[2])


def _planar_axes(
    prop_axis: AxisName, thickness_axis: AxisName
) -> Tuple[AxisName, AxisName]:
    """Return ``(width_axis, thickness_axis)`` validating they are distinct."""
    if prop_axis == thickness_axis:
        raise ValueError(
            f"propagation axis {prop_axis!r} must differ from thickness axis "
            f"{thickness_axis!r}"
        )
    width_axis = _third_axis(prop_axis, thickness_axis)
    return width_axis, thickness_axis


def _slab_box(
    *,
    center: Vec3Um,
    prop_axis: AxisName,
    width_axis: AxisName,
    thickness_axis: AxisName,
    length_um: float,
    width_um: float,
    thickness_um: float,
    medium: Medium,
) -> Structure:
    """A strip-waveguide ``Box`` of the given length/width/thickness centered
    on ``center`` with the stated axis roles."""
    size = [0.0, 0.0, 0.0]
    size[_INDEX[prop_axis]] = length_um
    size[_INDEX[width_axis]] = width_um
    size[_INDEX[thickness_axis]] = thickness_um
    return Structure(
        geometry=Box(center_um=center, size_um=(size[0], size[1], size[2])),
        medium=medium,
    )


# ---------------------------------------------------------------------------
# 1. straight
# ---------------------------------------------------------------------------


def straight(
    length_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A straight strip waveguide: one ``Box`` with input/output ports at the
    two ends along ``axis``."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    structure = _slab_box(
        center=center_um,
        prop_axis=axis,
        width_axis=width_axis,
        thickness_axis=thickness_axis,
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = length_um / 2.0
    ports = (
        Port("in", _vec(center_um, **{axis: -half}), axis, width_um),
        Port("out", _vec(center_um, **{axis: +half}), axis, width_um),
    )
    return Component(structures=(structure,), ports=ports)


# ---------------------------------------------------------------------------
# 2. bend (90-degree annular sector)
# ---------------------------------------------------------------------------


def bend(
    radius_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    thickness_axis: AxisName = "z",
) -> Component:
    """A 90-degree waveguide bend: an annular-sector ``Cylinder`` extruded
    along ``thickness_axis`` (the slab normal), with ``inner_radius`` /
    ``radius`` straddling ``radius_um +/- width_um/2`` and a pi/2 sweep.

    ``center_um`` is the bend's *center of curvature*. The cylinder sweeps the
    first quadrant in the transverse (u, v) plane, so the two arc ends point
    along the two in-plane axes. With the default ``thickness_axis="z"`` the
    transverse plane is (x, y): the start end (angle 0) sits at +u and
    propagates along that axis; the stop end (angle pi/2) sits at +v.
    """
    u_axis, v_axis = _bend_plane_axes(thickness_axis)
    geometry = Cylinder(
        axis=thickness_axis,
        center_um=center_um,
        radius_um=radius_um + width_um / 2.0,
        inner_radius_um=radius_um - width_um / 2.0,
        length_um=thickness_um,
        angle_start=0.0,
        angle_stop=math.pi / 2.0,
    )
    structure = Structure(geometry=geometry, medium=medium)
    # Arc endpoints on the centerline radius. Start end faces +u, propagating
    # along u_axis; stop end faces +v, propagating along v_axis.
    p_start = _vec(center_um, **{u_axis: radius_um})
    p_stop = _vec(center_um, **{v_axis: radius_um})
    ports = (
        Port("in", p_start, u_axis, width_um),
        Port("out", p_stop, v_axis, width_um),
    )
    return Component(structures=(structure,), ports=ports)


def _bend_plane_axes(thickness_axis: AxisName) -> Tuple[AxisName, AxisName]:
    """The (u, v) transverse axes for a cylinder extruded along
    ``thickness_axis`` (u = lower-indexed of the remaining two)."""
    remaining = [ax for ax in _AXES if ax != thickness_axis]
    return remaining[0], remaining[1]


# ---------------------------------------------------------------------------
# 3. taper (trapezoidal PolySlab)
# ---------------------------------------------------------------------------


def taper(
    length_um: float,
    width1_um: float,
    width2_um: float,
    *,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A linear width taper: a trapezoidal ``PolySlab`` (4 vertices) whose
    cross-section width grows ``width1_um`` -> ``width2_um`` along ``axis``.
    Ports at each end with the local widths."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    # PolySlab vertices are (u, v) in the two transverse axes of the EXTRUSION
    # axis (= thickness_axis): u = lower-indexed, v = higher-indexed. Here the
    # in-plane polygon lives in (prop, width) so we map those two roles onto
    # (u, v) by axis index, then emit vertices in increasing-index order.
    poly_axes = [ax for ax in _AXES if ax != thickness_axis]  # the two (u, v)
    half_len = length_um / 2.0
    # offsets of each role relative to center, keyed by axis
    half1 = width1_um / 2.0
    half2 = width2_um / 2.0
    # Build the four corners in (prop, width) space then project to (u, v).
    corners_pw = (
        (-half_len, -half1),  # input bottom
        (+half_len, -half2),  # output bottom
        (+half_len, +half2),  # output top
        (-half_len, +half1),  # input top
    )
    pw_to_axis = {axis: 0, width_axis: 1}
    vertices = []
    for corner in corners_pw:
        u = corner[pw_to_axis[poly_axes[0]]]
        v = corner[pw_to_axis[poly_axes[1]]]
        # add the center offset for these in-plane axes
        u += center_um[_INDEX[poly_axes[0]]]
        v += center_um[_INDEX[poly_axes[1]]]
        vertices.append((u, v))
    t_center = center_um[_INDEX[thickness_axis]]
    slab_bounds = (t_center - thickness_um / 2.0, t_center + thickness_um / 2.0)
    geometry = PolySlab(
        axis=thickness_axis,
        vertices_um=tuple(vertices),
        slab_bounds_um=slab_bounds,
    )
    structure = Structure(geometry=geometry, medium=medium)
    ports = (
        Port("in", _vec(center_um, **{axis: -half_len}), axis, width1_um),
        Port("out", _vec(center_um, **{axis: +half_len}), axis, width2_um),
    )
    return Component(structures=(structure,), ports=ports)


# ---------------------------------------------------------------------------
# 4. crossing (two boxes at 90 degrees)
# ---------------------------------------------------------------------------


def crossing(
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    arm_length_um: float = 3.0,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    thickness_axis: AxisName = "z",
) -> Component:
    """A waveguide crossing: two ``Box`` waveguides crossed at 90 degrees in
    the slab plane. Four ports, one per arm end."""
    a_axis, b_axis = _bend_plane_axes(thickness_axis)  # the two in-plane axes
    arm_a = _slab_box(
        center=center_um,
        prop_axis=a_axis,
        width_axis=b_axis,
        thickness_axis=thickness_axis,
        length_um=arm_length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    arm_b = _slab_box(
        center=center_um,
        prop_axis=b_axis,
        width_axis=a_axis,
        thickness_axis=thickness_axis,
        length_um=arm_length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = arm_length_um / 2.0
    ports = (
        Port(f"{a_axis}-", _vec(center_um, **{a_axis: -half}), a_axis, width_um),
        Port(f"{a_axis}+", _vec(center_um, **{a_axis: +half}), a_axis, width_um),
        Port(f"{b_axis}-", _vec(center_um, **{b_axis: -half}), b_axis, width_um),
        Port(f"{b_axis}+", _vec(center_um, **{b_axis: +half}), b_axis, width_um),
    )
    return Component(structures=(arm_a, arm_b), ports=ports)


# ---------------------------------------------------------------------------
# 5. coupler (two parallel straights, edge-to-edge gap)
# ---------------------------------------------------------------------------


def coupler(
    length_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    gap_um: float = 0.2,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis: AxisName = "x",
    thickness_axis: AxisName = "z",
) -> Component:
    """A directional coupler: two parallel straight waveguides separated
    edge-to-edge by ``gap_um`` (along the in-plane width axis). Four ports —
    two per guide at the ends."""
    width_axis, thickness_axis = _planar_axes(axis, thickness_axis)
    # center-to-center spacing = one width + the edge-to-edge gap
    offset = (width_um + gap_um) / 2.0
    top_center = _vec(center_um, **{width_axis: +offset})
    bot_center = _vec(center_um, **{width_axis: -offset})
    top = _slab_box(
        center=top_center,
        prop_axis=axis,
        width_axis=width_axis,
        thickness_axis=thickness_axis,
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    bot = _slab_box(
        center=bot_center,
        prop_axis=axis,
        width_axis=width_axis,
        thickness_axis=thickness_axis,
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = length_um / 2.0
    ports = (
        Port("top_in", _vec(top_center, **{axis: -half}), axis, width_um),
        Port("top_out", _vec(top_center, **{axis: +half}), axis, width_um),
        Port("bot_in", _vec(bot_center, **{axis: -half}), axis, width_um),
        Port("bot_out", _vec(bot_center, **{axis: +half}), axis, width_um),
    )
    return Component(structures=(top, bot), ports=ports)


# ---------------------------------------------------------------------------
# 6. ring (full-sweep annular Cylinder + bus Box)
# ---------------------------------------------------------------------------


def ring(
    radius_um: float,
    *,
    width_um: float = DEFAULT_WIDTH_UM,
    gap_um: float = 0.2,
    bus_width_um: float | None = None,
    bus_length_um: float | None = None,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    medium: Medium = SILICON,
    center_um: Vec3Um = (0.0, 0.0, 0.0),
    axis_normal: AxisName = "z",
) -> Component:
    """An all-pass ring resonator: a full-sweep annular ``Cylinder`` (inner =
    ``radius - width/2``, outer = ``radius + width/2``) plus a straight bus
    ``Box`` one ``gap_um`` away (edge-to-edge) from the ring's outer edge.

    ``axis_normal`` is the slab normal (the ring/cylinder extrusion axis).
    The bus runs along the lower-indexed in-plane axis; it is offset along the
    higher-indexed in-plane axis. Two ports on the bus (input + through)."""
    if bus_width_um is None:
        bus_width_um = width_um
    bus_axis, offset_axis = _bend_plane_axes(axis_normal)
    outer_r = radius_um + width_um / 2.0
    # ring sits centered on center_um in the slab plane
    ring_geom = Cylinder(
        axis=axis_normal,
        center_um=center_um,
        radius_um=outer_r,
        inner_radius_um=radius_um - width_um / 2.0,
        length_um=thickness_um,
        angle_start=0.0,
        angle_stop=2.0 * math.pi,
    )
    ring_structure = Structure(geometry=ring_geom, medium=medium)

    # Bus: parallel to bus_axis, offset along offset_axis so its inner edge is
    # gap_um from the ring's outer edge.
    bus_offset = outer_r + gap_um + bus_width_um / 2.0
    if bus_length_um is None:
        # span the ring diameter by default
        bus_length_um = 2.0 * outer_r
    bus_center = _vec(center_um, **{offset_axis: bus_offset})
    bus_structure = _slab_box(
        center=bus_center,
        prop_axis=bus_axis,
        width_axis=offset_axis,
        thickness_axis=axis_normal,
        length_um=bus_length_um,
        width_um=bus_width_um,
        thickness_um=thickness_um,
        medium=medium,
    )
    half = bus_length_um / 2.0
    ports = (
        Port("in", _vec(bus_center, **{bus_axis: -half}), bus_axis, bus_width_um),
        Port(
            "through",
            _vec(bus_center, **{bus_axis: +half}),
            bus_axis,
            bus_width_um,
        ),
    )
    return Component(structures=(ring_structure, bus_structure), ports=ports)
