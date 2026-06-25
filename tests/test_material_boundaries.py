"""Material-aware boundary selection (mirrors Tidy3D): a dispersive medium
crossing a PML face warns and is auto-switched to the adiabatic absorber, while
a plain dielectric keeps the PML. Geometry bounding boxes + the warning + the
``with_auto_boundaries`` helper."""

import math
import warnings

import pytest

import simupod as ph
from simupod.components._bounds import geometry_bounds_um


def _disp_medium(eps=2.25):
    return ph.Medium(
        permittivity=eps,
        lorentz=ph.LorentzPole(
            resonance_frequency_hz=5.0e14, delta_eps=1.0, linewidth_hz=1.0e13
        ),
    )


def _plain_medium(eps=2.25):
    return ph.Medium(permittivity=eps)


def _sim(structures, *, boundaries=None, size_um=(4.0, 4.0, 4.0)):
    return ph.Simulation(
        size_um=size_um,
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=boundaries or ph.Boundaries(),  # default = pml on all faces
        structures=structures,
        sources=[
            ph.PointDipole(
                center_um=(2.0, 2.0, 2.0),
                polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13),
            )
        ],
        monitors=[ph.FieldSnapshotMonitor(name="final", fields=["Ez"])],
    )


# --- geometry bounding boxes ------------------------------------------------

def test_box_bounds():
    bb = geometry_bounds_um(ph.Box(center_um=(1.0, 2.0, 3.0), size_um=(2.0, 4.0, 6.0)))
    assert bb == ((0.0, 2.0), (0.0, 4.0), (0.0, 6.0))


def test_sphere_bounds():
    bb = geometry_bounds_um(ph.Sphere(center_um=(1.0, 1.0, 1.0), radius_um=0.5))
    assert bb == ((0.5, 1.5), (0.5, 1.5), (0.5, 1.5))


def test_cylinder_bounds_axial_vs_transverse():
    cyl = ph.Cylinder(
        axis="z", center_um=(1.0, 1.0, 2.0), radius_um=0.5, length_um=3.0
    )
    bb = geometry_bounds_um(cyl)
    assert bb[0] == (0.5, 1.5)            # transverse: center +/- radius
    assert bb[1] == (0.5, 1.5)
    assert bb[2] == (0.5, 3.5)            # axial: center +/- length/2


def test_polyslab_bounds_pad_for_sidewall():
    # A straight-wall slab: transverse box is the vertex hull, axial is the slab.
    straight = ph.PolySlab(
        axis="z",
        vertices_um=((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
        slab_bounds_um=(0.0, 1.0),
    )
    bb = geometry_bounds_um(straight)
    assert bb[0] == (0.0, 2.0)
    assert bb[1] == (0.0, 1.0)
    assert bb[2] == (0.0, 1.0)
    # A slanted slab dilates outward by |tan(angle)| * thickness on both
    # transverse axes (conservative outer bound).
    slant = straight.model_copy(update={"sidewall_angle": 0.2})
    pad = math.tan(0.2) * 1.0
    bbs = geometry_bounds_um(slant)
    assert bbs[0] == pytest.approx((0.0 - pad, 2.0 + pad))
    assert bbs[1] == pytest.approx((0.0 - pad, 1.0 + pad))
    assert bbs[2] == (0.0, 1.0)          # axial unchanged


# --- the construction-time warning ------------------------------------------

def test_dispersive_crossing_pml_warns():
    # A dispersive bar spanning the full z extent runs into the z PML.
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning, match="dispersive .* extends into the PML"):
        _sim([bar])


def test_plain_dielectric_crossing_pml_does_not_warn():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_plain_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning would fail
        _sim([bar])


def test_dispersive_fully_interior_does_not_warn():
    # A small dispersive cube far from every face (PML band ~ 0.6 um, 12 layers).
    cube = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.2, 0.2, 0.2)),
        medium=_disp_medium(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _sim([cube])


def test_dispersive_crossing_absorber_axis_does_not_warn():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    # z already on the absorber -> nothing to warn about on that axis. (x/y PML
    # are not crossed by this z-running bar.)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _sim([bar], boundaries=ph.Boundaries(x="pml", y="pml", z="absorber"))


def test_warning_only_for_the_crossed_axis():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning) as rec:
        _sim([bar])
    msg = str(rec[0].message)
    # Only the crossed axis (z) is named; x/y are not flagged.
    assert "on axis 'z':" in msg


# --- with_auto_boundaries ---------------------------------------------------

def test_auto_boundaries_picks_absorber_only_on_crossed_axis():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning):
        sim = _sim([bar])
    auto = sim.with_auto_boundaries()
    assert auto.boundaries.z == "absorber"   # dispersive bar crosses z
    assert auto.boundaries.x == "pml"         # bar does not reach x/y faces
    assert auto.boundaries.y == "pml"


def test_auto_boundaries_plain_dielectric_stays_pml():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_plain_medium(),
    )
    auto = _sim([bar]).with_auto_boundaries()
    assert (auto.boundaries.x, auto.boundaries.y, auto.boundaries.z) == (
        "pml", "pml", "pml")


def test_auto_boundaries_preserves_periodic_and_pec():
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    # x periodic, y pec are explicit physics -> untouched even though the auto
    # rule only ever considers open axes; z (pml) flips to absorber.
    with pytest.warns(UserWarning):
        sim = _sim([bar], boundaries=ph.Boundaries(x="periodic", y="pec", z="pml"))
    auto = sim.with_auto_boundaries()
    assert auto.boundaries.x == "periodic"
    assert auto.boundaries.y == "pec"
    assert auto.boundaries.z == "absorber"


def test_auto_boundaries_does_not_change_wire_for_plain_scene():
    # No dispersive crossing -> identical boundaries -> byte-identical wire.
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_plain_medium(),
    )
    sim = _sim([bar])
    assert sim.with_auto_boundaries().to_wire_json() == sim.to_wire_json()


def test_from_wire_does_not_warn_on_dispersive_crossing():
    # Ingesting a saved document is the user's deliberate choice (like the §16
    # subpixel default) -> the construction-time advice is skipped.
    bar = ph.Structure(
        geometry=ph.Box(center_um=(2.0, 2.0, 2.0), size_um=(0.5, 0.5, 8.0)),
        medium=_disp_medium(),
    )
    with pytest.warns(UserWarning):
        sim = _sim([bar])
    wire = sim.to_wire_json()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = ph.Simulation.from_wire_json(wire)
    assert back.boundaries.z == "pml"
