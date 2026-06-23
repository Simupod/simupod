"""GDS layout import — turn a GDSII layout into PhotonHub structures.

A GDS file is a 2-D layout: ordered polygons each tagged by an integer
``(layer, datatype)`` pair, optionally organized into a hierarchy of cell
references (instances with translation/rotation/magnification). A photonic
device is built from that 2-D drawing by **extruding** each layer to a slab of a
fixed z-thickness filled with one material — the "layer stack".

:func:`import_gds` reads the file (via the optional ``gdstk`` dependency),
flattens any cell hierarchy into a flat polygon list, and emits one
:class:`~simupod.PolySlab` :class:`~simupod.Structure` per polygon on each
requested layer, using that layer's z-extent and medium. Polygon winding is
normalized to counter-clockwise (the orientation :class:`PolySlab` and the
rasterizer expect).

This is the client-side analogue of Tidy3D's ``Geometry.from_gds`` paired with a
``LayerStack``. It is what the GDS benchmark suite (``benchmarks/gds/``) uses to
build devices from the JPPhotonics ``fdtd-pipeline`` layouts (arXiv:2506.16665).

>>> import simupod as ph
>>> from simupod.gds import import_gds, GdsLayer
>>> si = ph.Medium(permittivity=3.478**2)
>>> structures = import_gds(
...     "crossing.gds",
...     [GdsLayer(layer=(1, 0), medium=si, zmin_um=0.0, thickness_um=0.22),
...      GdsLayer(layer=(2, 0), medium=si, zmin_um=0.0, thickness_um=0.15)],
... )
>>> sim = ph.Simulation(..., structures=structures)

Limitations (v1). Each polygon is extruded independently; polygons with holes
(even-odd fill) are not specially handled — for the strip/rib SOI layouts this
targets, every drawn shape is a simple filled region. Curved sidewalls are a
single global ``sidewall_angle`` per layer (matching ``PolySlab``); arbitrary
per-edge tapering is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .components.base import AxisName
from .components.structures import Medium, PolySlab, Structure

__all__ = ["GdsLayer", "import_gds", "read_gds_cell_names"]


@dataclass(frozen=True)
class GdsLayer:
    """One GDS ``(layer, datatype)`` mapped to an extruded slab of one medium.

    ``zmin_um`` / ``thickness_um`` give the slab extent along the extrusion axis
    (the :func:`import_gds` ``axis``, default ``z``); the slab spans
    ``[zmin_um, zmin_um + thickness_um]``. ``sidewall_angle`` (radians) and
    ``reference_plane`` are forwarded to every :class:`PolySlab` emitted for this
    layer (see :class:`~simupod.PolySlab`)."""

    layer: Tuple[int, int]
    medium: Medium
    zmin_um: float
    thickness_um: float
    sidewall_angle: float = 0.0
    reference_plane: str = "middle"

    def __post_init__(self) -> None:
        if self.thickness_um <= 0.0:
            raise ValueError(
                f"GdsLayer thickness_um must be > 0, got {self.thickness_um}"
            )

    @property
    def slab_bounds_um(self) -> Tuple[float, float]:
        return (self.zmin_um, self.zmin_um + self.thickness_um)


def _import_gdstk():
    """Import the optional ``gdstk`` GDSII reader with a helpful error."""
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "import_gds needs the optional 'gdstk' dependency to read GDSII "
            "files. Install it with `pip install gdstk` (the same reader Tidy3D "
            "and gdsfactory use)."
        ) from exc
    return gdstk


def _signed_area(points) -> float:
    """Twice the signed polygon area (shoelace); >0 for counter-clockwise."""
    n = len(points)
    acc = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return 0.5 * acc


def _normalize_ring(points) -> Optional[List[Tuple[float, float]]]:
    """Clean one polygon's point ring into a CCW list of ``(u, v)`` tuples.

    Drops a duplicated closing vertex if present and reverses clockwise rings so
    the result is counter-clockwise. Returns ``None`` for a degenerate ring
    (< 3 distinct vertices or zero area)."""
    ring = [(float(p[0]), float(p[1])) for p in points]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        return None
    area2 = _signed_area(ring)
    if area2 == 0.0:
        return None
    if area2 < 0.0:  # clockwise -> reverse to CCW
        ring.reverse()
    return ring


def _select_cell(gdstk, lib, cell_name: Optional[str], gds_path: str):
    """Pick the cell to import: the named one, or the single top-level cell."""
    if cell_name is not None:
        for cell in lib.cells:
            if cell.name == cell_name:
                return cell
        have = ", ".join(repr(c.name) for c in lib.cells)
        raise ValueError(
            f"cell {cell_name!r} not found in {gds_path}; cells: {have}"
        )
    tops = lib.top_level()
    if not tops:
        raise ValueError(f"{gds_path} contains no cells")
    if len(tops) > 1:
        names = ", ".join(repr(c.name) for c in tops)
        raise ValueError(
            f"{gds_path} has multiple top-level cells ({names}); pass "
            "cell_name= to choose one"
        )
    return tops[0]


def import_gds(
    gds_path: Union[str, Path],
    layers: Sequence[GdsLayer],
    *,
    cell_name: Optional[str] = None,
    axis: AxisName = "z",
    flatten: bool = True,
    min_area_um2: float = 0.0,
) -> Tuple[Structure, ...]:
    """Import a GDSII layout as a tuple of extruded :class:`Structure`.

    Parameters
    ----------
    gds_path:
        Path to the ``.gds`` file.
    layers:
        The layers to import, as :class:`GdsLayer` specs (each maps a GDS
        ``(layer, datatype)`` to a z-slab + medium). A GDS layer present in the
        file but absent from this list is ignored; a spec whose layer is absent
        from the file simply yields no structures.
    cell_name:
        Which cell to import. ``None`` (default) uses the file's single
        top-level cell (raising if there are several — pass a name to choose).
    axis:
        Extrusion axis = slab normal (default ``"z"``: the GDS drawing plane is
        ``(x, y)``). The two GDS coordinate columns map to the two transverse
        axes of ``axis`` in index order, so ``axis="z"`` keeps GDS ``x,y`` as
        the device ``x,y``.
    flatten:
        Resolve cell references (instances) into polygons first (default True).
        Required for hierarchical layouts; with ``False`` only the chosen cell's
        own polygons are read.
    min_area_um2:
        Drop polygons whose absolute area is below this (default 0 = keep all) —
        a guard against zero-area slivers from boolean ops.

    Returns
    -------
    tuple[Structure, ...]
        One :class:`PolySlab` structure per polygon, grouped in the given
        ``layers`` order (file order within a layer). Paint order is last-wins
        (NUMERICS.md §9); same-material overlaps from a flattened hierarchy are
        therefore harmless.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if not layers:
        raise ValueError("import_gds: pass at least one GdsLayer")

    gdstk = _import_gdstk()
    gds_path = str(gds_path)
    if not Path(gds_path).is_file():
        raise FileNotFoundError(gds_path)

    lib = gdstk.read_gds(gds_path)
    cell = _select_cell(gdstk, lib, cell_name, gds_path)
    if flatten:
        # Work on a copy so the library cell is not mutated; flatten() resolves
        # all references (applying their transforms) into this cell's polygons.
        cell = cell.copy(cell.name + "__phflat").flatten()

    # Bucket the cell's polygons by (layer, datatype) once.
    by_key: dict = {}
    for poly in cell.polygons:
        by_key.setdefault((poly.layer, poly.datatype), []).append(poly)

    structures: List[Structure] = []
    for spec in layers:
        slab_bounds = spec.slab_bounds_um
        for poly in by_key.get(spec.layer, ()):
            ring = _normalize_ring(poly.points)
            if ring is None:
                continue
            if min_area_um2 > 0.0 and abs(_signed_area(ring)) < min_area_um2:
                continue
            geometry = PolySlab(
                axis=axis,
                vertices_um=tuple(ring),
                slab_bounds_um=slab_bounds,
                sidewall_angle=spec.sidewall_angle,
                reference_plane=spec.reference_plane,
            )
            structures.append(Structure(geometry=geometry, medium=spec.medium))
    return tuple(structures)


def read_gds_cell_names(gds_path: Union[str, Path]) -> Tuple[str, ...]:
    """List every cell name in a GDS file (top-level first) — a discovery
    helper for picking ``cell_name`` / layers before :func:`import_gds`."""
    gdstk = _import_gdstk()
    gds_path = str(gds_path)
    if not Path(gds_path).is_file():
        raise FileNotFoundError(gds_path)
    lib = gdstk.read_gds(gds_path)
    tops = {c.name for c in lib.top_level()}
    ordered = [c.name for c in lib.cells if c.name in tops]
    ordered += [c.name for c in lib.cells if c.name not in tops]
    return tuple(ordered)
