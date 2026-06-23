"""Auto-mesh resolver (Track E) — ``ph.auto_grid``.

The resolver is a CLIENT-SIDE pure function turning a physical target
(steps-per-wavelength per medium + a grading ratio + wavelength + domain +
structures) into a valid GradedGridSpec (NUMERICS.md section 15). These tests
pin the acceptance contract:

  * the output passes GradedGridSpec validation (incl. GRADED_RATIO_GUARD and
    the section-15.1 invariants: coords[0]=0, strictly increasing, >= 4 nodes);
  * a finer target -> more cells and a smaller minimum spacing;
  * refinement actually concentrates cells in / around high-index structures;
  * MESH-FREEZE: identical inputs -> byte-identical coords (determinism), the
    property that keeps an adjoint objective continuous between iterations;
  * a realistic case: refine around a high-index silicon waveguide core.
"""

import math

import pytest

import simupod as ph
from simupod import auto_grid
from simupod.components.grid import (
    GRADED_RATIO_GUARD,
    GradedGridSpec,
    graded_primary_spacings,
)

C0 = 2.99792458e8


def _box(center, size, eps):
    return ph.Structure(
        geometry=ph.Box(center_um=center, size_um=size),
        medium=ph.Medium(permittivity=eps))


def _axis_spacings(spec: GradedGridSpec, axis: str):
    q = getattr(spec.coords, axis)
    return graded_primary_spacings(q)


def _max_cell_to_cell_ratio(spacings):
    """Largest adjacent cell-to-cell growth ratio (both directions)."""
    r = 1.0
    for a, b in zip(spacings[:-1], spacings[1:]):
        r = max(r, a / b, b / a)
    return r


# --------------------------------------------------------------------------- #
# Validity: the resolver returns a spec that already passed GradedGridSpec's
# own validators (the call below would have raised otherwise), and we re-assert
# the section-15.1 invariants explicitly.
# --------------------------------------------------------------------------- #

def test_returns_valid_graded_spec():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    assert isinstance(spec, GradedGridSpec)
    for axis in "xy":
        q = getattr(spec.coords, axis)
        assert q[0] == 0.0
        assert len(q) >= 4
        assert all(q[i + 1] > q[i] for i in range(len(q) - 1))
    # z was not requested -> not graded.
    assert spec.coords.z is None


def test_grading_ratio_respected_and_under_guard():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    for grading in (1.2, 1.3, 1.4):
        spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                         structures=[core], background_index=1.444,
                         steps_per_wvl=18.0, max_grading=grading, axes="xy")
        for axis in "xy":
            sp = _axis_spacings(spec, axis)
            ratio = _max_cell_to_cell_ratio(sp)
            # Cell-to-cell growth honors the requested grading (small rounding
            # slack from the 1e-7 um coordinate quantization).
            assert ratio <= grading + 1e-3, (axis, grading, ratio)
            # And the GLOBAL max/min guard the spec enforces holds with margin.
            assert max(sp) / min(sp) <= GRADED_RATIO_GUARD


# --------------------------------------------------------------------------- #
# Finer target -> more cells and a smaller minimum spacing.
# --------------------------------------------------------------------------- #

def test_finer_target_gives_more_cells_and_smaller_min_spacing():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              structures=[core], background_index=1.444,
              max_grading=1.3, axes="xy")
    coarse = auto_grid(steps_per_wvl=10.0, **kw)
    fine = auto_grid(steps_per_wvl=30.0, **kw)
    for axis in "xy":
        qc, qf = getattr(coarse.coords, axis), getattr(fine.coords, axis)
        assert len(qf) > len(qc), axis
        assert min(_axis_spacings(fine, axis)) < min(_axis_spacings(coarse, axis))


# --------------------------------------------------------------------------- #
# Refinement concentrates cells IN / AROUND the high-index structure.
# --------------------------------------------------------------------------- #

def test_cells_concentrate_in_high_index_core():
    # 0.45um silicon (n=3.5) core centered in a 2um cladding (n=1.444) domain.
    core = _box((1.0, 0.5, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    spec = auto_grid(size_um=(2.0, 1.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x")
    qx = spec.coords.x
    # Count nodes inside the core span [1.0 - 0.225, 1.0 + 0.225] vs an
    # equal-width window in the cladding far from the core.
    lo, hi = 1.0 - 0.225, 1.0 + 0.225
    in_core = sum(1 for c in qx if lo <= c <= hi)
    # A cladding window of the same 0.45um width near the edge.
    edge_lo, edge_hi = 0.0, 0.45
    in_edge = sum(1 for c in qx if edge_lo <= c <= edge_hi)
    assert in_core > in_edge, (in_core, in_edge)
    # The finest cell sits inside the core (local dl ~ lambda/(n*steps)).
    sp = _axis_spacings(spec, "x")
    finest_idx = min(range(len(sp)), key=lambda i: sp[i])
    cell_center = 0.5 * (qx[finest_idx] + qx[finest_idx + 1])
    assert lo - 0.1 <= cell_center <= hi + 0.1, cell_center
    # Local dl in the core ~ 1.31/(3.5*20) = 0.0187 um; background ~ 0.045 um.
    expected_core_dl = 1.31 / (3.5 * 20.0)
    assert min(sp) <= expected_core_dl * 1.2


def test_no_high_index_structure_gives_essentially_uniform():
    # Background-only domain (or structures at/below background index): no
    # refinement, so the mesh is ~uniform at the background spacing.
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    # Uniform up to the 1e-7 um coordinate quantization (no refinement at all).
    assert _max_cell_to_cell_ratio(sp) == pytest.approx(1.0, abs=1e-4)
    # spacing ~ lambda/(n*steps) = 1/20 = 0.05 um.
    assert min(sp) == pytest.approx(0.05, rel=0.05)


def test_low_index_structure_does_not_refine():
    # A structure with index <= background must not pull in fine cells.
    lowq = _box((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), eps=1.0)  # n=1 < bg 1.444
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[lowq], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    # Uniform up to the 1e-7 um coordinate quantization (no refinement at all).
    assert _max_cell_to_cell_ratio(sp) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# MESH-FREEZE: determinism. Identical inputs -> byte-identical coords. This is
# the property that keeps an adjoint objective continuous across iterations.
# --------------------------------------------------------------------------- #

def test_mesh_freeze_byte_identical_for_identical_inputs():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              structures=[core], background_index=1.444,
              steps_per_wvl=22.0, max_grading=1.27, axes="xy")
    a = auto_grid(**kw)
    b = auto_grid(**kw)
    # Exact equality of the coordinate tuples (and the JSON wire form).
    assert a.coords.x == b.coords.x
    assert a.coords.y == b.coords.y
    assert a.dl_um == b.dl_um
    assert a.model_dump_json() == b.model_dump_json()


def test_mesh_freeze_independent_of_structure_list_order():
    # The mesh is a pure function of the SET of (span, index) constraints, so
    # reordering structures (an optimizer might) must not move a single node.
    s1 = _box((0.6, 1.0, 1.0), (0.3, 0.3, 4.0), eps=12.25)
    s2 = _box((1.4, 1.0, 1.0), (0.3, 0.3, 4.0), eps=6.0)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              background_index=1.444, steps_per_wvl=18.0,
              max_grading=1.3, axes="x")
    a = auto_grid(structures=[s1, s2], **kw)
    b = auto_grid(structures=[s2, s1], **kw)
    assert a.coords.x == b.coords.x


def test_mesh_freeze_small_input_change_small_output_change():
    # Determinism + continuity: a tiny wavelength nudge must NOT produce a wild
    # mesh jump (the discontinuity the roadmap warns about). We just assert the
    # node count is stable and the coords move only slightly.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), structures=[core],
              background_index=1.444, steps_per_wvl=20.0,
              max_grading=1.3, axes="x")
    a = auto_grid(wavelength_um=1.310, **kw)
    b = auto_grid(wavelength_um=1.311, **kw)
    assert len(a.coords.x) == len(b.coords.x)
    drift = max(abs(p - q) for p, q in zip(a.coords.x, b.coords.x))
    assert drift < 0.01  # < one fine cell


# --------------------------------------------------------------------------- #
# Realistic case: SOI strip waveguide cross-section (matches the hand-built
# mesh in benchmarks/waveguide/waveguide.py — refine the core, coarsen cladding).
# --------------------------------------------------------------------------- #

def test_realistic_soi_waveguide_cross_section():
    LX, LY, LZ = 1.6, 1.4, 5.0
    core = _box((LX / 2, LY / 2, LZ / 2), (0.45, 0.22, LZ * 2), eps=3.5 ** 2)
    spec = auto_grid(
        size_um=(LX, LY, LZ), wavelength_um=1.31, structures=[core],
        background_index=1.444, steps_per_wvl=20.0, max_grading=1.4, axes="xy")
    # Drop it straight into a Simulation to prove the produced spec is usable.
    src = ph.PointDipole(
        center_um=(LX / 2, LY / 2, 0.7), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=C0 / 1.31e-6,
                                      fwidth_hz=0.12 * C0 / 1.31e-6))
    sim = ph.Simulation(
        size_um=(LX, LY, LZ), grid=spec, run={"n_steps": 100},
        background=ph.Background(permittivity=1.444 ** 2), structures=(core,),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10, sources=[src])
    # Graded x and y realized lengths come from the closing-node rule, not L.
    rx, ry, _ = sim._realized_um()
    assert math.isclose(rx, LX, abs_tol=0.02)
    assert math.isclose(ry, LY, abs_tol=0.02)
    # The core (~0.22um tall) is resolved by several fine cells in y.
    spy = _axis_spacings(spec, "y")
    fine_in_core = [d for q, d in zip(spec.coords.y, spy)
                    if LY / 2 - 0.11 <= q <= LY / 2 + 0.11]
    assert len(fine_in_core) >= 4
    # Cladding cells (near the y edge) are coarser than core cells.
    assert max(spy) > min(spy) * 1.5
    # Round-trips through the wire format unchanged.
    sim2 = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert sim2.grid.coords.y == spec.coords.y


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kw,match", [
    (dict(wavelength_um=0.0), "wavelength_um"),
    (dict(steps_per_wvl=0.0), "steps_per_wvl"),
    (dict(max_grading=1.0), "max_grading must be > 1"),
    (dict(max_grading=11.0), "exceeds GRADED_RATIO_GUARD"),
    (dict(background_index=0.5), "background_index"),
    (dict(axes="xq"), "subset of 'xyz'"),
    (dict(axes="xx"), "no repeats"),
    (dict(min_nodes=3), "min_nodes"),
])
def test_invalid_inputs_rejected(kw, match):
    base = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                background_index=1.0, steps_per_wvl=20.0, max_grading=1.3)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        auto_grid(**base)


def test_min_nodes_floor_for_tiny_domain():
    # A domain so small the target produces < 4 cells still yields >= 4 nodes
    # (section 15.10), uniformly subdivided.
    spec = auto_grid(size_um=(0.05, 0.05, 0.05), wavelength_um=1.31,
                     structures=[], background_index=1.0,
                     steps_per_wvl=4.0, max_grading=1.3, axes="x")
    assert len(spec.coords.x) >= 4
    assert spec.coords.x[0] == 0.0


def test_sphere_geometry_supported():
    # A high-index sphere refines around its bounding box on each axis.
    sph = ph.Structure(
        geometry=ph.Sphere(center_um=(1.0, 1.0, 1.0), radius_um=0.3),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[sph], background_index=1.0,
                     steps_per_wvl=16.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    finest_idx = min(range(len(sp)), key=lambda i: sp[i])
    qx = spec.coords.x
    cell_center = 0.5 * (qx[finest_idx] + qx[finest_idx + 1])
    assert 0.6 <= cell_center <= 1.4  # within the sphere's bounding box


# --------------------------------------------------------------------------- #
# Curved / extruded geometries: Cylinder and PolySlab now drive refinement
# (previously returned None and were silently ignored — exactly the curved
# structures where subpixel matters most). They are bounded by their enclosing
# box: a safe over-estimate of where the fine mesh is needed.
# --------------------------------------------------------------------------- #

def test_cylinder_refines_along_extrusion_and_transverse():
    # A z-extruded high-index disk (radius 0.4 um, length 1.0 um) centered in a
    # 2um cube. Its bbox is [0.6,1.4] in x,y (radial) and [0.5,1.5] in z (length).
    cyl = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=1.0),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[cyl], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    # The finest cell on a radial axis sits inside the disk radius bbox.
    for axis, (lo, hi) in (("x", (0.6, 1.4)), ("y", (0.6, 1.4)),
                           ("z", (0.5, 1.5))):
        sp = _axis_spacings(spec, axis)
        q = getattr(spec.coords, axis)
        fi = min(range(len(sp)), key=lambda i: sp[i])
        cell_center = 0.5 * (q[fi] + q[fi + 1])
        assert lo - 0.15 <= cell_center <= hi + 0.15, (axis, cell_center)
        # local dl ~ lambda/(n*steps) ~ 1.31/(3.5*18) = 0.0208 um
        assert min(sp) <= 1.31 / (3.5 * 18.0) * 1.25, (axis, min(sp))


def test_cylinder_partial_sector_uses_full_disk_bbox():
    # A 90-degree sector still refines over the FULL disk bbox (over-estimate is
    # safe — never under-refines). Just assert refinement happened transversely.
    sec = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=2.0,
                             angle_start=0.0, angle_stop=math.pi / 2),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[sec], background_index=1.0,
                     steps_per_wvl=16.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    assert _max_cell_to_cell_ratio(sp) > 1.05  # the mesh is graded, not uniform


def test_polyslab_refines_in_cross_section_and_along_extrusion():
    # A z-extruded triangle (vertices in x,y) with slab_bounds in z. The polygon
    # bbox is [0.6,1.4]x[0.7,1.3]; the slab runs z in [0.4,0.9].
    verts = ((0.6, 0.7), (1.4, 0.7), (1.0, 1.3))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                            slab_bounds_um=(0.4, 0.9)),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    for axis, (lo, hi) in (("x", (0.6, 1.4)), ("y", (0.7, 1.3)),
                           ("z", (0.4, 0.9))):
        sp = _axis_spacings(spec, axis)
        q = getattr(spec.coords, axis)
        fi = min(range(len(sp)), key=lambda i: sp[i])
        cell_center = 0.5 * (q[fi] + q[fi + 1])
        assert lo - 0.15 <= cell_center <= hi + 0.15, (axis, cell_center)


def test_polyslab_slanted_sidewall_refines_reference_plane_bbox():
    # A slanted sidewall only NARROWS the section away from the reference plane,
    # so the reference-plane vertex bbox is a safe over-estimate. Refinement
    # still concentrates around the (widest) cross-section.
    verts = ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                            slab_bounds_um=(0.4, 1.6), sidewall_angle=0.3,
                            reference_plane="bottom"),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=16.0, max_grading=1.3, axes="x")
    sp = _axis_spacings(spec, "x")
    fi = min(range(len(sp)), key=lambda i: sp[i])
    qx = spec.coords.x
    cell_center = 0.5 * (qx[fi] + qx[fi + 1])
    assert 0.5 - 0.15 <= cell_center <= 1.5 + 0.15, cell_center


def test_curved_and_box_refine_consistently():
    # A Cylinder with the same bbox as a Box must refine the same axis span (the
    # bbox-based span is geometry-agnostic). Cell counts should be comparable.
    box = _box((1.0, 1.0, 1.0), (0.8, 0.8, 0.8), eps=12.25)
    cyl = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=0.8),
        medium=ph.Medium(permittivity=12.25))
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
              background_index=1.0, steps_per_wvl=18.0,
              max_grading=1.3, axes="x")
    sbox = auto_grid(structures=[box], **kw)
    scyl = auto_grid(structures=[cyl], **kw)
    # Same x-bbox [0.6,1.4] -> identical coordinate arrays.
    assert sbox.coords.x == scyl.coords.x


# --------------------------------------------------------------------------- #
# Tidy3D-parity hardening: wavelength inference from a source, dl_min floor,
# enforced-refinement override regions.
# --------------------------------------------------------------------------- #

def test_wavelength_inferred_from_source_matches_explicit():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    lam_um = 1.31
    f0 = C0 / (lam_um * 1e-6)
    src = ph.PointDipole(
        center_um=(1.0, 1.0, 0.5), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    kw = dict(size_um=(2.0, 2.0, 4.0), structures=[core],
              background_index=1.444, steps_per_wvl=20.0,
              max_grading=1.3, axes="xy")
    explicit = auto_grid(wavelength_um=lam_um, **kw)
    inferred = auto_grid(source=src, **kw)
    # The inferred wavelength is c/f0 (== lam_um here), so coords match closely.
    assert inferred.coords.x == explicit.coords.x
    assert inferred.coords.y == explicit.coords.y


def test_wavelength_source_mutual_exclusion_and_required():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    f0 = C0 / 1.31e-6
    src = ph.PointDipole(
        center_um=(1.0, 1.0, 0.5), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    base = dict(size_um=(2.0, 2.0, 4.0), structures=[core],
                background_index=1.444, axes="x")
    # both -> error
    with pytest.raises(ValueError, match="exactly one"):
        auto_grid(wavelength_um=1.31, source=src, **base)
    # neither -> error
    with pytest.raises(ValueError, match="wavelength_um is required"):
        auto_grid(**base)


def test_dl_min_floor_caps_refinement():
    # A very high-index inclusion would normally pull dl ~ lambda/(n*steps); the
    # dl_min floor caps the minimum spacing so the cell count cannot explode.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=49.0)  # n=7
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              structures=[core], background_index=1.444,
              steps_per_wvl=20.0, max_grading=1.3, axes="x")
    free = auto_grid(**kw)
    floor = 0.03
    capped = auto_grid(dl_min_um=floor, **kw)
    # No realized cell falls below the floor (within coordinate quantization).
    sp = _axis_spacings(capped, "x")
    assert min(sp) >= floor - 1e-3, min(sp)
    # And the floor genuinely bites: fewer cells than the unfloored mesh.
    assert len(capped.coords.x) < len(free.coords.x)


def test_dl_min_must_be_positive():
    with pytest.raises(ValueError, match="dl_min_um must be > 0"):
        auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                  background_index=1.0, dl_min_um=0.0, axes="x")


def test_refine_region_override_forces_fine_mesh_in_empty_space():
    # No structures, but an override forces fine cells over [0.8, 1.2] in x.
    fine_dl = 0.01
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     refine_regions=[("x", 0.8, 1.2, fine_dl)])
    sp = _axis_spacings(spec, "x")
    q = spec.coords.x
    fi = min(range(len(sp)), key=lambda i: sp[i])
    cell_center = 0.5 * (q[fi] + q[fi + 1])
    assert 0.8 - 0.1 <= cell_center <= 1.2 + 0.1, cell_center
    # The override pulled cells well below the background spacing (0.05 um).
    assert min(sp) <= fine_dl * 1.3


def test_refine_region_respects_dl_min_floor():
    # An over-fine override is still clamped to the dl_min floor.
    floor = 0.02
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     dl_min_um=floor,
                     refine_regions=[("x", 0.8, 1.2, 0.002)])
    sp = _axis_spacings(spec, "x")
    assert min(sp) >= floor - 1e-3, min(sp)


@pytest.mark.parametrize("region,match", [
    (("q", 0.8, 1.2, 0.01), "axis must be one of"),
    (("x", 1.2, 0.8, 0.01), "hi > lo"),
    (("x", 0.8, 1.2, 0.0), "dl_um must be > 0"),
    (("x", 0.8, 1.2), r"\(axis_letter, lo_um, hi_um, dl_um\)"),
])
def test_refine_region_validation(region, match):
    with pytest.raises(ValueError, match=match):
        auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                  background_index=1.0, axes="x", refine_regions=[region])


def test_refine_region_mesh_freeze_deterministic():
    # Overrides are part of the pure-function inputs: same inputs -> same coords.
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
              background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
              axes="x", refine_regions=[("x", 0.8, 1.2, 0.01)])
    a = auto_grid(**kw)
    b = auto_grid(**kw)
    assert a.coords.x == b.coords.x


# --------------------------------------------------------------------------- #
# Simulation.with_auto_grid convenience: opt-in, derives the mesh from the
# scene (size / structures / background / source wavelength). The default
# UniformGridSpec is unchanged, so no existing scene's wire output moves.
# --------------------------------------------------------------------------- #

def test_simulation_with_auto_grid_convenience():
    from simupod.components.grid import GradedGridSpec as _GGS

    LX, LY, LZ = 2.0, 2.0, 4.0
    core = _box((LX / 2, LY / 2, LZ / 2), (0.45, 0.22, LZ * 2), eps=12.25)
    f0 = C0 / 1.31e-6
    src = ph.PointDipole(
        center_um=(LX / 2, LY / 2, 0.7), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=f0, fwidth_hz=0.1 * f0))
    sim = ph.Simulation(
        size_um=(LX, LY, LZ), grid=ph.UniformGridSpec(dl_um=0.05),
        run={"n_steps": 100}, background=ph.Background(permittivity=1.444 ** 2),
        structures=(core,), boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        pml_num_layers=10, sources=[src])
    # Default scene uses the uniform grid (no churn).
    assert isinstance(sim.grid, ph.UniformGridSpec)

    auto = sim.with_auto_grid(steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    assert isinstance(auto.grid, _GGS)
    # Original is untouched (frozen copy semantics).
    assert isinstance(sim.grid, ph.UniformGridSpec)
    # Wavelength was inferred from the source -> matches the explicit call.
    explicit = auto_grid(size_um=(LX, LY, LZ), wavelength_um=1.31,
                         structures=[core], background_index=1.444,
                         steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    assert auto.grid.coords.x == explicit.coords.x
    assert auto.grid.coords.y == explicit.coords.y
    # The resulting sim round-trips through the wire format.
    sim2 = ph.Simulation.from_wire_json(auto.to_wire_json())
    assert sim2.grid.coords.x == auto.grid.coords.x


# --------------------------------------------------------------------------- #
# Interface grid-line SNAPPING (Tidy3D AutoGrid parity, gap #10a). A primary
# node must land EXACTLY on each refining-structure interface / override edge,
# WITHOUT breaking the grading-ratio or dl_min invariants, and deterministically
# (mesh-freeze). snap_interfaces defaults to True.
# --------------------------------------------------------------------------- #

_SNAP_TOL = 1e-6  # exact-snap tolerance (one coordinate quantum is 1e-7 um)


def _nearest_node_dist(q, target):
    return min(abs(c - target) for c in q)


def test_snap_box_faces_land_on_nodes():
    # SOI core faces: x at 1.0 +/- 0.225 -> {0.775, 1.225}; y at 1.0 +/- 0.11.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="xy")
    for axis, faces in (("x", (0.775, 1.225)), ("y", (0.89, 1.11))):
        q = getattr(spec.coords, axis)
        for f in faces:
            assert _nearest_node_dist(q, f) <= _SNAP_TOL, (axis, f)


def test_snap_off_recovers_pre_snap_and_interface_falls_mid_cell():
    # With snapping OFF the interface generally does NOT land on a node — this is
    # the gap snapping closes. The snapped spec differs from the unsnapped one.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31, structures=[core],
              background_index=1.444, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    on = auto_grid(snap_interfaces=True, **kw)
    off = auto_grid(snap_interfaces=False, **kw)
    assert on.coords.x != off.coords.x  # snapping moved nodes
    # Pre-snap: at least one face is mid-cell (not on a node).
    assert max(_nearest_node_dist(off.coords.x, f)
               for f in (0.775, 1.225)) > _SNAP_TOL
    # Post-snap: both faces are on nodes.
    for f in (0.775, 1.225):
        assert _nearest_node_dist(on.coords.x, f) <= _SNAP_TOL


def test_snap_preserves_grading_ratio():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    for grading in (1.2, 1.3, 1.4):
        spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                         structures=[core], background_index=1.444,
                         steps_per_wvl=18.0, max_grading=grading, axes="xy")
        for axis in "xy":
            sp = _axis_spacings(spec, axis)
            assert _max_cell_to_cell_ratio(sp) <= grading + 1e-3, (axis, grading)
            assert max(sp) / min(sp) <= GRADED_RATIO_GUARD


def test_snap_respects_dl_min_floor():
    # Snapping must never push a cell below the dl_min floor (it abandons a
    # target it cannot reconcile rather than emitting a sub-floor cell).
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=49.0)  # n=7
    floor = 0.03
    spec = auto_grid(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
                     structures=[core], background_index=1.444,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     dl_min_um=floor)
    sp = _axis_spacings(spec, "x")
    assert min(sp) >= floor - 1e-3, min(sp)


def test_snap_deterministic_mesh_freeze():
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31, structures=[core],
              background_index=1.444, steps_per_wvl=22.0, max_grading=1.3,
              axes="xy")
    a = auto_grid(**kw)
    b = auto_grid(**kw)
    assert a.coords.x == b.coords.x
    assert a.coords.y == b.coords.y
    assert a.model_dump_json() == b.model_dump_json()


def test_snap_independent_of_structure_order():
    # Snap targets are a pure function of the SET of interfaces, so reordering
    # the structure list must not move a single node.
    s1 = _box((0.6, 1.0, 1.0), (0.3, 0.3, 4.0), eps=12.25)
    s2 = _box((1.4, 1.0, 1.0), (0.3, 0.3, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31,
              background_index=1.444, steps_per_wvl=18.0, max_grading=1.3,
              axes="x")
    a = auto_grid(structures=[s1, s2], **kw)
    b = auto_grid(structures=[s2, s1], **kw)
    assert a.coords.x == b.coords.x


def test_snap_cylinder_bbox_edges():
    # A z-extruded disk: x/y radial bbox {0.6, 1.4}; z length bbox {0.5, 1.5}.
    cyl = ph.Structure(
        geometry=ph.Cylinder(axis="z", center_um=(1.0, 1.0, 1.0),
                             radius_um=0.4, length_um=1.0),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[cyl], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    for axis, edges in (("x", (0.6, 1.4)), ("y", (0.6, 1.4)),
                        ("z", (0.5, 1.5))):
        q = getattr(spec.coords, axis)
        for e in edges:
            assert _nearest_node_dist(q, e) <= _SNAP_TOL, (axis, e)
        assert _max_cell_to_cell_ratio(_axis_spacings(spec, axis)) <= 1.3 + 1e-3


def test_snap_polyslab_boundaries():
    # Triangle bbox x {0.6, 1.4}, y {0.7, 1.3}; slab z {0.4, 0.9}.
    verts = ((0.6, 0.7), (1.4, 0.7), (1.0, 1.3))
    poly = ph.Structure(
        geometry=ph.PolySlab(axis="z", vertices_um=verts,
                            slab_bounds_um=(0.4, 0.9)),
        medium=ph.Medium(permittivity=12.25))
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.31,
                     structures=[poly], background_index=1.0,
                     steps_per_wvl=18.0, max_grading=1.3, axes="xyz")
    for axis, edges in (("x", (0.6, 1.4)), ("y", (0.7, 1.3)),
                        ("z", (0.4, 0.9))):
        q = getattr(spec.coords, axis)
        for e in edges:
            assert _nearest_node_dist(q, e) <= _SNAP_TOL, (axis, e)


def test_snap_override_region_edges():
    # An override-region edge is an interface too — snap a node onto it. Use a
    # grading wide enough that the coarsening shoulder has room for both edges.
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.4, axes="x",
                     refine_regions=[("x", 0.8, 1.2, 0.01)])
    q = spec.coords.x
    for e in (0.8, 1.2):
        assert _nearest_node_dist(q, e) <= _SNAP_TOL, e


def test_snap_best_effort_never_violates_grading():
    # When an interface CANNOT be reconciled with the grading guard (a coarsening
    # shoulder already saturated at max_grading), snapping abandons that target
    # rather than emitting an out-of-spec mesh. The result is still valid.
    spec = auto_grid(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0,
                     structures=[], background_index=1.0,
                     steps_per_wvl=20.0, max_grading=1.3, axes="x",
                     refine_regions=[("x", 0.8, 1.2, 0.01)])
    sp = _axis_spacings(spec, "x")
    # Grading invariant holds regardless of how many targets were snappable.
    assert _max_cell_to_cell_ratio(sp) <= 1.3 + 1e-3
    # The closer edge still snaps even when the farther one is abandoned.
    assert _nearest_node_dist(spec.coords.x, 0.8) <= _SNAP_TOL


def test_snap_off_matches_legacy_for_uniform_cases():
    # snap_interfaces has no effect when there are no refining interfaces (the
    # "no refinement -> uniform" cases): toggling it is a no-op there.
    kw = dict(size_um=(2.0, 2.0, 2.0), wavelength_um=1.0, structures=[],
              background_index=1.0, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    on = auto_grid(snap_interfaces=True, **kw)
    off = auto_grid(snap_interfaces=False, **kw)
    assert on.coords.x == off.coords.x


def test_snap_realized_domain_length_unchanged():
    # Snapping only moves INTERIOR nodes; the §15.1 closing node (hence the
    # realized domain length) is untouched, so the snapped and unsnapped specs
    # realize the same length.
    core = _box((1.0, 1.0, 1.0), (0.45, 0.22, 4.0), eps=12.25)
    kw = dict(size_um=(2.0, 2.0, 4.0), wavelength_um=1.31, structures=[core],
              background_index=1.444, steps_per_wvl=20.0, max_grading=1.3,
              axes="x")
    on = auto_grid(snap_interfaces=True, **kw)
    off = auto_grid(snap_interfaces=False, **kw)
    realized = lambda q: q[-1] + (q[-1] - q[-2])
    assert math.isclose(realized(on.coords.x), realized(off.coords.x),
                        abs_tol=1e-3)
