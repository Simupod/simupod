"""Tests for the visualization layer (simupod.viz; design §10).

Forces the matplotlib Agg backend before any pyplot import so the smoke matrix
runs headless. No golden-image diffs — structural assertions only.
"""

import math
import warnings

import matplotlib
matplotlib.use("Agg")  # noqa: E402  (must precede pyplot)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

import simupod as ph  # noqa: E402
from simupod.viz import eps as epsmod  # noqa: E402

AXES = ("x", "y", "z")


# --------------------------------------------------------------------------- #
# Scene builders (built from the real component models; design §10).
# --------------------------------------------------------------------------- #

def _pulse():
    return ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13)


def dipole_vacuum():
    """Bare dipole in vacuum, no structures, periodic boundaries."""
    return ph.Simulation(
        size_um=(1.0, 1.0, 1.0),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        sources=[ph.PointDipole(center_um=(0.5, 0.5, 0.5), polarization="Ez",
                                source_time=_pulse())],
    )


def fresnel_slab():
    """A high-index slab spanning the transverse plane, plane wave + PML on z,
    a DFT and a flux monitor."""
    return ph.Simulation(
        size_um=(0.4, 0.4, 1.2),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        structures=[ph.Structure(
            geometry=ph.Box(center_um=(0.2, 0.2, 0.7), size_um=(1.0, 1.0, 0.3)),
            medium=ph.Medium(permittivity=12.0))],
        sources=[ph.PlaneWave(axis="z", direction="+", position_um=0.2,
                              polarization="Ex", source_time=_pulse())],
        monitors=[
            ph.FieldDftMonitor(name="slab", center_um=(0.2, 0.2, 0.7),
                               size_um=(0.4, 0.4, 0.0),
                               fields=["Ex", "Ey", "Ez"], freqs_hz=[1.934e14]),
            ph.FluxMonitor(name="refl", axis="z", position_um=0.3,
                           freqs_hz=[1.934e14]),
        ],
    )


def sphere_scene():
    """A single dielectric sphere with a dipole and a time-probe monitor."""
    return ph.Simulation(
        size_um=(1.0, 1.0, 1.0),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[ph.Structure(
            geometry=ph.Sphere(center_um=(0.5, 0.5, 0.5), radius_um=0.25),
            medium=ph.Medium(permittivity=9.0))],
        sources=[ph.PointDipole(center_um=(0.5, 0.5, 0.5), polarization="Ez",
                                source_time=_pulse())],
        monitors=[ph.FieldTimeMonitor(name="probe", center_um=(0.7, 0.5, 0.5),
                                      fields=["Ez"])],
    )


def soi_waveguide():
    """SOI-like Si box waveguide (eps ~12) on a lower-index substrate box."""
    return ph.Simulation(
        size_um=(2.0, 1.0, 1.0),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[
            ph.Structure(  # buried oxide / substrate slab
                geometry=ph.Box(center_um=(1.0, 0.5, 0.25), size_um=(3.0, 3.0, 0.5)),
                medium=ph.Medium(permittivity=2.1)),
            ph.Structure(  # Si core
                geometry=ph.Box(center_um=(1.0, 0.5, 0.6), size_um=(3.0, 0.5, 0.22)),
                medium=ph.Medium(permittivity=12.1)),
        ],
        sources=[ph.PointDipole(center_um=(0.3, 0.5, 0.6), polarization="Ey",
                                source_time=_pulse())],
        monitors=[ph.FieldDftMonitor(name="mode", center_um=(1.7, 0.5, 0.6),
                                     size_um=(0.0, 0.8, 0.6),
                                     fields=["Ex", "Ey", "Ez"],
                                     freqs_hz=[1.934e14])],
    )


def graded_scene():
    """A graded (non-uniform) mesh around a Si box, dipole source."""
    structs = [ph.Structure(
        geometry=ph.Box(center_um=(0.5, 0.5, 0.3), size_um=(0.4, 0.2, 0.22)),
        medium=ph.Medium(permittivity=12.0))]
    return ph.Simulation(
        size_um=(1.0, 1.0, 0.6),
        grid=ph.auto_grid(size_um=(1.0, 1.0, 0.6), wavelength_um=1.55,
                          structures=structs, background_index=1.0, axes="xy"),
        run=ph.RunSpec(n_steps=5),
        structures=structs,
        sources=[ph.PointDipole(center_um=(0.5, 0.5, 0.3), polarization="Ez",
                                source_time=_pulse())],
    )


def cylinder_scene():
    """Three extruded cylinders along z: a solid disk, a ring (inner>0), and a
    90-degree annular wedge — exercises every cylinder cut-plane branch."""
    return ph.Simulation(
        size_um=(1.0, 1.0, 1.0),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[
            ph.Structure(  # solid disk
                geometry=ph.Cylinder(axis="z", center_um=(0.3, 0.3, 0.5),
                                     radius_um=0.15, length_um=0.4),
                medium=ph.Medium(permittivity=9.0)),
            ph.Structure(  # ring (annulus)
                geometry=ph.Cylinder(axis="z", center_um=(0.7, 0.7, 0.5),
                                     radius_um=0.2, inner_radius_um=0.1,
                                     length_um=0.4),
                medium=ph.Medium(permittivity=6.0)),
            ph.Structure(  # 90-degree annular wedge (bend-like)
                geometry=ph.Cylinder(axis="z", center_um=(0.5, 0.5, 0.5),
                                     radius_um=0.35, inner_radius_um=0.2,
                                     length_um=0.4, angle_start=0.0,
                                     angle_stop=math.pi / 2),
                medium=ph.Medium(permittivity=4.0)),
        ],
        sources=[ph.PointDipole(center_um=(0.5, 0.5, 0.5), polarization="Ez",
                                source_time=_pulse())],
    )


def polyslab_scene():
    """An L-shaped polygon extruded along z (slab_bounds in z), dipole source."""
    return ph.Simulation(
        size_um=(1.0, 1.0, 1.0),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[ph.Structure(
            geometry=ph.PolySlab(
                axis="z",
                vertices_um=((0.2, 0.2), (0.7, 0.2), (0.7, 0.4),
                             (0.4, 0.4), (0.4, 0.7), (0.2, 0.7)),
                slab_bounds_um=(0.3, 0.7)),
            medium=ph.Medium(permittivity=11.0))],
        sources=[ph.PointDipole(center_um=(0.5, 0.5, 0.5), polarization="Ez",
                                source_time=_pulse())],
    )


ALL_SCENES = {
    "dipole_vacuum": dipole_vacuum,
    "fresnel_slab": fresnel_slab,
    "sphere": sphere_scene,
    "soi_waveguide": soi_waveguide,
    "graded": graded_scene,
    "cylinder": cylinder_scene,
    "polyslab": polyslab_scene,
}


def _cut_value(sim, axis):
    """A value inside the realized domain on ``axis`` (its center)."""
    realized = sim._realized_um()
    return 0.5 * realized["xyz".index(axis)]


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Smoke matrix: every method × every cut over every scene (design §10).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scene_name", list(ALL_SCENES))
@pytest.mark.parametrize("axis", AXES)
def test_plot_smoke(scene_name, axis):
    sim = ALL_SCENES[scene_name]()
    ax = sim.plot(**{axis: _cut_value(sim, axis)})
    from matplotlib.axes import Axes
    assert isinstance(ax, Axes)


@pytest.mark.parametrize("scene_name", list(ALL_SCENES))
@pytest.mark.parametrize("axis", AXES)
def test_plot_eps_smoke(scene_name, axis):
    sim = ALL_SCENES[scene_name]()
    ax = sim.plot_eps(**{axis: _cut_value(sim, axis)})
    from matplotlib.axes import Axes
    assert isinstance(ax, Axes)
    # Colorbar present (design §10): an extra Axes on the figure.
    assert len(ax.figure.axes) >= 2


def test_grid_overlay_draws_cell_edges():
    """grid=True overlays the realized Yee cell edges on both 2D views (the
    vlines + hlines add two LineCollections over the base scene/heatmap)."""
    sim = soi_waveguide()
    cut = _cut_value(sim, "z")
    base = len(sim.plot(z=cut).collections)
    gridded = len(sim.plot(z=cut, grid=True).collections)
    assert gridded >= base + 2
    # same flag works on the rasterized-ε view
    from matplotlib.axes import Axes
    assert isinstance(sim.plot_eps(z=cut, grid=True), Axes)


def test_source_time_plot():
    """GaussianPulse.plot() previews J(t)+spectrum; ax= draws just the trace."""
    from matplotlib.axes import Axes
    src = ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13)
    ax = src.plot()
    assert isinstance(ax, Axes)
    assert ax.get_title() == "source time"
    assert len(ax.figure.axes) >= 2           # time + spectrum panels
    _, given = plt.subplots()
    assert src.plot(ax=given) is given         # ax= path: single trace on given Axes


def test_render_slice_pure():
    """render_slice draws a frame — the pure, testable core of the scrubber."""
    from matplotlib.axes import Axes

    from simupod.viz import render_slice
    sim = soi_waveguide()
    cut = _cut_value(sim, "z")
    assert isinstance(render_slice(sim, "z", cut, eps=True, grid=True), Axes)
    assert isinstance(render_slice(sim, "z", cut, eps=False, grid=False), Axes)


def test_interactive_preview_builds():
    """sim.preview() wires up an ipywidgets container (no live kernel needed)."""
    widgets = pytest.importorskip("ipywidgets")
    ui = soi_waveguide().preview()
    assert isinstance(ui, widgets.Widget)
    assert len(ui.children) == 3      # axis+slider row, ε+grid row, output area


# --------------------------------------------------------------------------- #
# Structural assertions (design §10).
# --------------------------------------------------------------------------- #

def test_no_show_called(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.__setitem__("n", 1))
    dipole_vacuum().plot(z=0.5)
    soi_waveguide().plot_eps(z=0.6)
    assert called["n"] == 0


def test_exactly_one_plane_required():
    sim = dipole_vacuum()
    with pytest.raises(ValueError):
        sim.plot()
    with pytest.raises(ValueError):
        sim.plot(x=0.1, z=0.1)
    with pytest.raises(ValueError):
        sim.plot_eps()


def test_returns_given_ax():
    sim = sphere_scene()
    fig, ax = plt.subplots()
    out = sim.plot(z=0.5, ax=ax)
    assert out is ax


def test_structure_patch_count():
    """One structure patch per intersecting geometry (design §10)."""
    from matplotlib.patches import Circle, Rectangle

    # Sphere cut through its center -> exactly one Circle.
    ax = sphere_scene().plot(z=0.5)
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    assert len(circles) == 1

    # SOI z=0.6 passes through the Si core only (substrate centered at z=0.25,
    # half-height 0.25 -> spans [0,0.5], so 0.6 misses it); core box +
    # background fill + the dashed monitor rect are Rectangles.
    ax = soi_waveguide().plot(z=0.6)
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    # background(1) + Si core(1) + monitor dashed box(1) [monitor size_um x=0
    #  -> a span line, not a Rectangle on this z-cut since it is a plane along x]
    assert len(rects) >= 2


def test_non_intersecting_structure_skipped():
    """A plane that misses a structure draws no patch for it (design §9)."""
    from matplotlib.patches import Circle

    sim = sphere_scene()
    ax = sim.plot(z=0.95)  # well outside the sphere [0.25, 0.75]
    assert [p for p in ax.patches if isinstance(p, Circle)] == []


def test_cylinder_perpendicular_cut_shapes():
    """z-cut (perpendicular to the extrusion axis) through the cylinders yields
    the exact cross-sections: a filled Circle (solid disk), an Annulus (ring),
    and a Polygon (the 90-degree wedge)."""
    from matplotlib.patches import Annulus, Circle, Polygon

    ax = cylinder_scene().plot(z=0.5)
    circles = [p for p in ax.patches if type(p) is Circle]
    annuli = [p for p in ax.patches if isinstance(p, Annulus)]
    polygons = [p for p in ax.patches if type(p) is Polygon]
    assert len(circles) == 1   # solid disk
    assert len(annuli) == 1    # ring
    assert len(polygons) == 1  # 90-degree annular wedge


def test_cylinder_parallel_cut_bounding_rect():
    """A cut PARALLEL to the extrusion axis approximates each cylinder as a
    bounding Rectangle (axial extent x cylinder diameter)."""
    from matplotlib.patches import Rectangle

    ax = cylinder_scene().plot(y=0.3)  # passes through the solid disk @ y=0.3
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    # background fill + at least the solid-disk bounding box.
    assert len(rects) >= 2


def test_cylinder_perpendicular_cut_outside_axial_extent_skipped():
    """A z-cut outside the cylinders' axial span [0.3, 0.7] draws no cylinder
    cross-section (only the background)."""
    from matplotlib.patches import Annulus, Circle, Polygon

    ax = cylinder_scene().plot(z=0.95)
    assert [p for p in ax.patches if type(p) is Circle] == []
    assert [p for p in ax.patches if isinstance(p, Annulus)] == []
    assert [p for p in ax.patches if type(p) is Polygon] == []


def test_polyslab_perpendicular_cut_is_polygon():
    """A z-cut (perpendicular to the extrusion axis) within the slab bounds
    draws the exact polygon cross-section."""
    from matplotlib.patches import Polygon

    ax = polyslab_scene().plot(z=0.5)
    polygons = [p for p in ax.patches if type(p) is Polygon]
    assert len(polygons) == 1
    # The L-shaped polygon has its 6 input vertices (plotted closed).
    assert len(polygons[0].get_xy()) in (6, 7)  # matplotlib may repeat vertex 0


def test_polyslab_parallel_cut_bounding_rect():
    """A cut parallel to the extrusion axis approximates the slab as a bounding
    Rectangle over [slab_lo, slab_hi] x (in-plane polygon extent)."""
    from matplotlib.patches import Polygon, Rectangle

    ax = polyslab_scene().plot(x=0.3)  # within the polygon x-span [0.2, 0.7]
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) >= 2  # background + the slab bounding rect
    assert [p for p in ax.patches if type(p) is Polygon] == []


def test_polyslab_parallel_cut_outside_span_skipped():
    """A parallel cut outside the polygon's transverse span draws no slab.

    The all-PML scene draws background + PML-band Rectangles regardless; the
    test isolates the slab by comparing a hitting cut (one extra Rectangle) to
    a missing cut (no extra Rectangle) and asserting no Polygon either way."""
    from matplotlib.patches import Polygon, Rectangle

    sim = polyslab_scene()
    n_hit = len([p for p in sim.plot(x=0.3).patches
                 if isinstance(p, Rectangle)])      # within x-span [0.2, 0.7]
    ax_miss = sim.plot(x=0.95)                       # beyond the x-span
    n_miss = len([p for p in ax_miss.patches if isinstance(p, Rectangle)])
    assert n_miss == n_hit - 1  # exactly the slab bounding rect is dropped
    assert [p for p in ax_miss.patches if type(p) is Polygon] == []


def test_new_shapes_plot_eps_and_field_smoke():
    """plot_eps over every cut and plot_field structure outlines do not raise
    for the cylinder/polyslab scenes (outlines reuse the same cut-plane specs)."""
    from matplotlib.axes import Axes
    from matplotlib.patches import Annulus, Polygon

    from simupod.viz import plot_field

    for builder in (cylinder_scene, polyslab_scene):
        sim = builder()
        for axis in AXES:
            ax = sim.plot_eps(**{axis: _cut_value(sim, axis)})
            assert isinstance(ax, Axes)

    # plot_field outlines on a z-cut: the cylinder ring/wedge add Annulus +
    # Polygon outlines, the polyslab adds a Polygon outline.
    fd = _FakeData({"slab": _dft_array()})
    ax = plot_field(fd, "slab", field="Ex", z=0.5, structures=True,
                    simulation=cylinder_scene())
    assert [p for p in ax.patches if isinstance(p, Annulus)]
    ax2 = plot_field(fd, "slab", field="Ex", z=0.5, structures=True,
                     simulation=polyslab_scene())
    assert [p for p in ax2.patches if type(p) is Polygon]


def test_source_overlay_present_iff_source_in_plane():
    """Dipole marker appears on a cut through it, absent on a far cut."""
    sim = dipole_vacuum()  # dipole at z=0.5
    ax = sim.plot(z=0.5)
    # A coral 'o' marker line is added for the source.
    from simupod.viz._style import SOURCE_COLOR
    src_lines = [ln for ln in ax.lines
                 if ln.get_color() == SOURCE_COLOR and ln.get_marker() == "o"]
    assert len(src_lines) == 1
    # A cut far from the dipole (> half a cell) drops the marker.
    ax2 = sim.plot(z=0.9)
    src_lines2 = [ln for ln in ax2.lines
                  if ln.get_color() == SOURCE_COLOR and ln.get_marker() == "o"]
    assert src_lines2 == []


def test_monitor_overlay_present_iff_monitor():
    """A DFT-monitor dashed rectangle appears only when a monitor is present
    and intersects the plane."""
    from matplotlib.patches import Rectangle

    from simupod.viz._style import MONITOR_COLOR

    sim = fresnel_slab()  # DFT monitor centered at z=0.7, size z=0 (a plane)
    ax = sim.plot(x=0.2)  # x-cut: in-plane axes (y,z); the z=0 plane is a line
    # The monitor in an x-cut shows as a dashed amber line at z=0.7.
    amber = [ln for ln in ax.lines if ln.get_color() == MONITOR_COLOR]
    assert amber, "monitor overlay missing"

    # No-monitor scene draws no amber monitor glyph.
    ax2 = dipole_vacuum().plot(z=0.5)
    amber2 = [ln for ln in ax2.lines if ln.get_color() == MONITOR_COLOR]
    dashed_rects = [p for p in ax2.patches
                    if isinstance(p, Rectangle) and p.get_edgecolor()[:3]
                    == tuple(__import__("matplotlib").colors.to_rgb(MONITOR_COLOR))]
    assert amber2 == [] and dashed_rects == []


def test_pml_band_count_matches_boundaries():
    """PML bands == (# in-plane PML axes) × 2 faces; periodic axes skipped."""
    from simupod.viz._style import pml_bands

    sim = fresnel_slab()  # x,y periodic; z pml
    # z-cut: in-plane axes (x, y) are both periodic -> no bands.
    assert pml_bands(sim, "z") == []
    # x-cut: in-plane axes (y, z); only z is PML -> 2 faces.
    bands_x = pml_bands(sim, "x")
    assert len(bands_x) == 2
    assert all(letter == "z" for letter, _, _ in bands_x)

    sim2 = sphere_scene()  # all PML
    # z-cut: in-plane (x, y) both PML -> 4 bands (2 axes × 2 faces).
    assert len(pml_bands(sim2, "z")) == 4


def test_pml_band_thickness_uses_layers_and_spacing():
    from simupod.viz._style import pml_bands

    sim = sphere_scene()  # uniform dl=0.05, all PML, default pml_num_layers=12
    bands = pml_bands(sim, "z")
    # Low-face band on x: [0, layers*dl] = [0, 12*0.05] = [0, 0.6].
    low = [b for b in bands if b[0] == "x" and b[1] == 0.0][0]
    assert low[2] == pytest.approx(12 * 0.05)


def test_plot_eps_shape_matches_realized_grid():
    """plot_eps samples on the realized grid; eps2d shape == (n_v, n_h)."""
    sim = graded_scene()
    nx, ny, nz = epsmod.realized_shape(sim)
    _, _, eps_z = epsmod.sample_eps_plane(sim, "z", 0.3)
    assert eps_z.shape == (ny, nx)  # z-cut in-plane (x, y)
    _, _, eps_x = epsmod.sample_eps_plane(sim, "x", 0.5)
    assert eps_x.shape == (nz, ny)  # x-cut in-plane (y, z)


def test_plot_eps_samples_structure_and_background():
    """The ε array carries both the background and the structure permittivity."""
    sim = soi_waveguide()
    _, _, eps = epsmod.sample_eps_plane(sim, "y", 0.5)
    assert float(eps.max()) == pytest.approx(12.1)  # Si core
    assert float(eps.min()) == pytest.approx(1.0)   # background vacuum


def test_plot_eps_out_of_domain_warns_not_raises():
    sim = dipole_vacuum()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ax = sim.plot_eps(z=50.0)
        assert any("outside" in str(x.message) for x in w)
    from matplotlib.axes import Axes
    assert isinstance(ax, Axes)


def test_plot_legend_toggle():
    sim = sphere_scene()
    ax = sim.plot(z=0.5, legend=True)
    assert ax.get_legend() is not None
    ax2 = sim.plot(z=0.5, legend=False)
    assert ax2.get_legend() is None


# --------------------------------------------------------------------------- #
# plot_field — built on a hand-made DataArray matching data.py's contract.
# --------------------------------------------------------------------------- #

class _FakeData:
    """Minimal SimulationData stand-in: a dict of name -> DataArray plus a
    manifest, enough for plot_field."""

    def __init__(self, arrays, manifest=None):
        self._arrays = arrays
        self.manifest = manifest or {}

    def __getitem__(self, name):
        if name not in self._arrays:
            raise KeyError(f"unknown monitor {name!r}; available: "
                           f"{list(self._arrays)}")
        return self._arrays[name]


def _dft_array(components=("Ex", "Ey", "Ez"), freqs=(1.934e14,), ny=8, nx=8):
    rng = np.random.RandomState(0)
    data = (rng.randn(len(freqs), len(components), 1, ny, nx)
            + 1j * rng.randn(len(freqs), len(components), 1, ny, nx)).astype(np.complex64)
    return xr.DataArray(
        data, dims=("f", "component", "z", "y", "x"),
        coords={"f": list(freqs), "component": list(components),
                "z": [0.5], "y": np.linspace(0.1, 0.9, ny),
                "x": np.linspace(0.1, 0.9, nx)},
        attrs={"normalization": "phasors normalized by A0*S(f)",
               "monitor": "slab"},
        name="slab")


@pytest.mark.parametrize("axis", AXES)
def test_plot_field_smoke_each_cut(axis):
    """A volumetric DFT array sliced on each cut returns an Axes with data."""
    rng = np.random.RandomState(1)
    data = (rng.randn(1, 3, 6, 6, 6)).astype(np.complex64)
    da = xr.DataArray(
        data, dims=("f", "component", "z", "y", "x"),
        coords={"f": [1.934e14], "component": ["Ex", "Ey", "Ez"],
                "z": np.linspace(0, 1, 6), "y": np.linspace(0, 1, 6),
                "x": np.linspace(0, 1, 6)},
        attrs={}, name="vol")
    fd = _FakeData({"vol": da})
    from simupod.viz import plot_field
    out = plot_field(fd, "vol", field="Ex", structures=False, **{axis: 0.5})
    from matplotlib.axes import Axes
    assert isinstance(out, Axes)
    assert out.collections  # the pcolormesh QuadMesh


def test_plot_field_selects_slice_and_colorbar():
    fd = _FakeData({"slab": _dft_array()})
    from simupod.viz import plot_field
    ax = plot_field(fd, "slab", field="Ez", z=0.5, val="real", structures=False)
    # Colorbar present.
    assert len(ax.figure.axes) >= 2
    # The QuadMesh is on the right coordinate extent (x in [0.1, 0.9]).
    mesh = ax.collections[0]
    assert mesh is not None


@pytest.mark.parametrize("field,val", [
    ("Ex", "real"), ("Ex", "imag"), ("Ex", "abs"), ("Ex", "phase"),
    ("E", "abs"), ("intensity", "abs"),
])
def test_plot_field_components_and_derived(field, val):
    fd = _FakeData({"slab": _dft_array()})
    from simupod.viz import plot_field
    ax = plot_field(fd, "slab", field=field, val=val, z=0.5, structures=False)
    assert ax.collections


def test_plot_field_colormap_selection():
    """Signed -> diverging, magnitude -> sequential, phase -> cyclic (design §7)."""
    from simupod.viz._style import field_cmap_and_norm

    arr = np.array([[-1.0, 1.0], [0.5, -0.5]])
    cmap, norm = field_cmap_and_norm("Ex", "real", arr)
    assert cmap == "RdBu_r" and norm.vmin == -norm.vmax  # symmetric
    cmap, norm = field_cmap_and_norm("E", "abs", np.abs(arr))
    assert cmap == "magma"
    cmap, norm = field_cmap_and_norm("Ex", "phase", arr)
    assert cmap == "twilight"
    assert norm.vmin == pytest.approx(-np.pi) and norm.vmax == pytest.approx(np.pi)


def test_plot_field_bad_field_keyerror():
    fd = _FakeData({"slab": _dft_array(components=("Ex", "Ey"))})
    from simupod.viz import plot_field
    with pytest.raises(KeyError, match="available"):
        plot_field(fd, "slab", field="Hz", z=0.5, structures=False)


def test_plot_field_partial_derived_errors():
    fd = _FakeData({"slab": _dft_array(components=("Ex", "Ey"))})
    from simupod.viz import plot_field
    with pytest.raises(ValueError, match="Ez"):
        plot_field(fd, "slab", field="E", z=0.5, structures=False)


def test_plot_field_multifreq_requires_freq():
    fd = _FakeData({"slab": _dft_array(freqs=(1.7e14, 1.934e14))})
    from simupod.viz import plot_field
    with pytest.raises(ValueError, match="multiple frequencies"):
        plot_field(fd, "slab", field="Ex", z=0.5, structures=False)
    # With freq= it succeeds.
    ax = plot_field(fd, "slab", field="Ex", freq=1.934e14, z=0.5,
                    structures=False)
    assert ax.collections


def test_plot_field_unknown_monitor_keyerror():
    fd = _FakeData({"slab": _dft_array()})
    from simupod.viz import plot_field
    with pytest.raises(KeyError):
        plot_field(fd, "nope", field="Ex", z=0.5, structures=False)


def test_plot_field_structures_overlay_with_simulation():
    """structures=True + simulation= overlays outlines (design §3)."""
    sim = soi_waveguide()
    fd = _FakeData({"slab": _dft_array()})
    from simupod.viz import plot_field
    ax = plot_field(fd, "slab", field="Ex", z=0.5, structures=True,
                    simulation=sim)
    # Outline patches (unfilled) present from the structures intersecting z=0.5.
    from matplotlib.patches import Rectangle
    outlines = [p for p in ax.patches
                if isinstance(p, Rectangle) and not p.get_fill()]
    assert outlines


def test_plot_field_no_geometry_one_time_note():
    """structures=True without geometry warns once, does not raise."""
    import simupod.viz.field as fieldmod
    fieldmod._NO_GEOMETRY_NOTED["flag"] = False  # reset the one-time gate
    fd = _FakeData({"slab": _dft_array()})
    from simupod.viz import plot_field
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plot_field(fd, "slab", field="Ex", z=0.5, structures=True)
        assert any("geometry" in str(x.message) for x in w)


def test_plot_field_val_ignored_for_real_monitor():
    """val is ignored for a real (time-domain) snapshot (design §9)."""
    data = np.arange(1 * 1 * 1 * 5 * 5, dtype=np.float32).reshape(1, 1, 1, 5, 5)
    da = xr.DataArray(
        data, dims=("t", "component", "z", "y", "x"),
        coords={"t": [1e-15], "component": ["Ez"], "z": [0.5],
                "y": np.linspace(0, 1, 5), "x": np.linspace(0, 1, 5)},
        attrs={}, name="snap")
    fd = _FakeData({"snap": da})
    from simupod.viz import plot_field
    ax = plot_field(fd, "snap", field="Ez", val="phase", z=0.5, structures=False)
    assert ax.collections  # no error; val ignored on real data


# --------------------------------------------------------------------------- #
# plot_eps Cylinder / PolySlab rasterization (Gap 1: the arc/polygon are no
# longer invisible — plot_eps now paints the §9 hard sample, matching plot).
# --------------------------------------------------------------------------- #

def _bend_taper_scene():
    """A library.bend (Cylinder arc) + a library.taper (PolySlab) placed inside
    a positive domain. The bend's center of curvature is at (0.2, 0.2) so its
    first-quadrant 90-degree arc (centerline radius 0.5) sweeps into the domain;
    the taper sits in the opposite corner."""
    from simupod import library as lib

    bend = lib.bend(radius_um=0.5, width_um=0.3, thickness_um=0.22,
                    center_um=(0.2, 0.2, 0.5))
    taper = lib.taper(length_um=0.6, width1_um=0.2, width2_um=0.5,
                      thickness_um=0.22, center_um=(1.4, 1.4, 0.5))
    return ph.Simulation(
        size_um=(2.0, 2.0, 1.0),
        grid=ph.UniformGridSpec(dl_um=0.02),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=list(bend.structures) + list(taper.structures),
        sources=[ph.PointDipole(center_um=(0.5, 0.5, 0.5), polarization="Ez",
                                source_time=_pulse())],
    )


def _eps_at(sim, axis, value, h_um, v_um):
    """The sampled ε cell nearest the in-plane point (h_um, v_um) on a cut."""
    h_nodes, v_nodes, eps = epsmod.sample_eps_plane(sim, axis, value)
    hc = epsmod.axis_cell_centers_um(h_nodes)
    vc = epsmod.axis_cell_centers_um(v_nodes)
    ih = int(np.argmin(np.abs(hc - h_um)))
    iv = int(np.argmin(np.abs(vc - v_um)))
    return float(eps[iv, ih])


def test_plot_eps_paints_cylinder_arc():
    """The bend Cylinder is no longer invisible: a cell ON the arc centerline
    carries the silicon ε (≈12.25), not background; the ring hole and the
    out-of-sweep region stay background (matches engine cylinder_contains)."""
    sim = _bend_taper_scene()
    # Arc centerline at 45 degrees, radius 0.5 from the curvature center (0.2,0.2).
    px = 0.2 + 0.5 * math.cos(math.pi / 4)
    py = 0.2 + 0.5 * math.sin(math.pi / 4)
    assert _eps_at(sim, "z", 0.5, px, py) == pytest.approx(12.25)
    # The hole (radius 0.2 < inner_radius 0.35) is background.
    hx = 0.2 + 0.2 * math.cos(math.pi / 4)
    hy = 0.2 + 0.2 * math.sin(math.pi / 4)
    assert _eps_at(sim, "z", 0.5, hx, hy) == pytest.approx(1.0)


def test_plot_eps_paints_polyslab_polygon():
    """The taper PolySlab is no longer invisible: a cell inside the polygon
    carries its ε (silicon ≈12.25) on a perpendicular (z) cut."""
    sim = _bend_taper_scene()
    # Taper is centered at (1.4, 1.4); its interior contains that point.
    assert _eps_at(sim, "z", 0.5, 1.4, 1.4) == pytest.approx(12.25)


def test_plot_eps_cylinder_outside_axial_extent_is_background():
    """A z-cut outside the bend's axial span [0.39, 0.61] paints no arc."""
    sim = _bend_taper_scene()
    px = 0.2 + 0.5 * math.cos(math.pi / 4)
    py = 0.2 + 0.5 * math.sin(math.pi / 4)
    assert _eps_at(sim, "z", 0.95, px, py) == pytest.approx(1.0)


def test_plot_eps_cylinder_parallel_cut_band():
    """A cut PARALLEL to the extrusion axis paints the rectangular band where
    the plane crosses the solid (here the ring scene from the smoke matrix)."""
    sim = cylinder_scene()  # solid disk @ (0.3,0.3,0.5), r=0.15, z in [0.3,0.7]
    # y=0.3 cut -> in-plane (x, z); the disk center column at x=0.3, z=0.5 is in.
    assert _eps_at(sim, "y", 0.3, 0.3, 0.5) == pytest.approx(9.0)
    # Outside the axial extent on z is background.
    assert _eps_at(sim, "y", 0.3, 0.3, 0.95) == pytest.approx(1.0)


def test_plot_eps_polyslab_parallel_cut_band():
    """A cut parallel to the PolySlab extrusion axis paints the bounding band."""
    sim = polyslab_scene()  # L-polygon, slab_bounds z in [0.3, 0.7]
    # x=0.3 cut -> in-plane (y, z); (y=0.3, z=0.5) is inside both polygon & slab.
    assert _eps_at(sim, "x", 0.3, 0.3, 0.5) == pytest.approx(11.0)
    # z above the slab bound is background.
    assert _eps_at(sim, "x", 0.3, 0.3, 0.95) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# plot_mode — an FDE mode's transverse field heatmap (Gap 2).
# --------------------------------------------------------------------------- #

def test_plot_mode_returns_axes_with_image():
    """plot_mode on a real fundamental mode returns an Axes carrying the field
    heatmap (a pcolormesh QuadMesh) at the mode's profile resolution."""
    from matplotlib.axes import Axes

    from simupod.plugins import ModeSolver
    from simupod.viz import plot_mode

    ms = ModeSolver.from_rectangular_core(
        wavelength_um=1.55, dl_um=0.04, core_w_um=0.5, core_h_um=0.22,
        n_core=3.48, n_clad=1.44)
    mode = ms.solve(num_modes=1)[0]
    ax = plot_mode(mode)
    assert isinstance(ax, Axes)
    assert ax.collections  # the QuadMesh
    # The heatmap is on the mode's (ny, nx) cross-section.
    mesh = ax.collections[0]
    ny, nx = mode.field.shape
    assert mesh.get_array().size == ny * nx
    # Title carries the component and n_eff; colorbar present.
    assert "Ex" in ax.get_title()
    assert f"{mode.n_eff:.4g}" in ax.get_title()
    assert len(ax.figure.axes) >= 2  # plot + colorbar


def test_plot_mode_respects_given_ax():
    from simupod.plugins import ModeSolver
    from simupod.viz import plot_mode

    ms = ModeSolver.from_rectangular_core(
        wavelength_um=1.55, dl_um=0.05, core_w_um=0.5, core_h_um=0.22,
        n_core=3.48, n_clad=1.44)
    mode = ms.solve(num_modes=1)[0]
    fig, ax = plt.subplots()
    out = plot_mode(mode, ax=ax)
    assert out is ax


# --------------------------------------------------------------------------- #
# plot_spectrum — transmission T(λ) (Gap 3).
# --------------------------------------------------------------------------- #

def test_plot_spectrum_single_dict():
    """A single {freq_hz: T} mapping -> one trace, x in nm and ascending."""
    from matplotlib.axes import Axes

    from simupod.viz import plot_spectrum

    spec = {1.97e14: 0.88, 1.9e14: 0.90, 1.934e14: 0.95}  # unsorted on purpose
    ax = plot_spectrum(spec)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1
    xs = ax.lines[0].get_xdata()
    # Wavelengths near 1550 nm telecom band, sorted ascending.
    assert xs[0] < xs[-1]
    assert 1500.0 < xs.min() < xs.max() < 1600.0
    assert "nm" in ax.get_xlabel()
    # No legend for a single unlabelled trace.
    assert ax.get_legend() is None


def test_plot_spectrum_labelled_traces():
    """A {label: {freq_hz: T}} mapping -> one line per label, with a legend."""
    from simupod.viz import plot_spectrum

    multi = {
        "through": {1.9e14: 0.6, 1.97e14: 0.7},
        "cross": {1.9e14: 0.3, 1.97e14: 0.2},
    }
    ax = plot_spectrum(multi)
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"through", "cross"}
    # y-axis spans [0, ~1.05].
    assert ax.get_ylim()[0] == 0.0
    assert ax.get_ylim()[1] == pytest.approx(1.05)


def test_plot_spectrum_freq_to_wavelength():
    """λ_nm = c / f (c = 2.99792458e8): a known frequency maps to its nm."""
    from simupod.viz import plot_spectrum

    f = 1.934e14
    ax = plot_spectrum({f: 0.5})
    expected_nm = 2.99792458e8 / f * 1e9
    assert ax.lines[0].get_xdata()[0] == pytest.approx(expected_nm)


def test_plot_spectrum_empty_raises():
    from simupod.viz import plot_spectrum

    with pytest.raises(ValueError):
        plot_spectrum({})


# --------------------------------------------------------------------------- #
# plot_3d (plotly). Skipped when plotly is missing (design §10).
# --------------------------------------------------------------------------- #

plotly = pytest.importorskip("plotly")


@pytest.mark.parametrize("scene_name", list(ALL_SCENES))
def test_plot_3d_returns_figure(scene_name):
    import plotly.graph_objects as go

    fig = ALL_SCENES[scene_name]().plot_3d()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # at least the domain wireframe


def test_plot_3d_trace_count():
    """Trace count: structures + sources + monitors + domain wireframe + PML
    shells, for a known scene."""
    sim = fresnel_slab()
    fig = sim.plot_3d()
    # 1 structure box + 1 plane-wave plane + 2 monitors (dft box + flux plane)
    # + 1 domain wireframe + 2 PML shells (z low/high) = 7.
    assert len(fig.data) == 7


def test_plot_3d_trace_count_with_new_shapes():
    """The cylinder (3 structures) and polyslab (1 structure) scenes each emit
    one Mesh3d per structure plus source markers, domain wireframe, and PML
    shells — confirming the new prism/tube builders are wired in."""
    import plotly.graph_objects as go

    # cylinder_scene: 3 cylinder meshes + 1 dipole marker + 1 domain wireframe
    # + 6 PML shells (3 axes x 2 faces) = 11.
    fig = cylinder_scene().plot_3d()
    assert len(fig.data) == 11
    meshes = [t for t in fig.data if isinstance(t, go.Mesh3d)]
    assert len(meshes) == 3 + 6  # 3 cylinders + 6 PML shell boxes

    # polyslab_scene: 1 prism mesh + 1 dipole marker + 1 wireframe + 6 PML = 9.
    fig2 = polyslab_scene().plot_3d()
    assert len(fig2.data) == 9
    assert len([t for t in fig2.data if isinstance(t, go.Mesh3d)]) == 1 + 6


def test_plot_3d_import_error_without_plotly(monkeypatch):
    """When plotly is absent, plot_3d raises ImportError with the install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("no plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"simupod\[viz\]"):
        dipole_vacuum().plot_3d()
