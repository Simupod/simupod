"""Phase 2 geometry: PolySlab + annular Cylinder pydantic models, wire
round-trip through the discriminated union, and an end-to-end run through the
real phsolver (NUMERICS.md §17)."""

import math

import pytest

import simupod as ph
from simupod.runners.local import find_solver


def _cyl(**kw):
    d = dict(axis="z", center_um=(1.0, 1.0, 1.0), radius_um=0.5, length_um=2.0)
    d.update(kw)
    return ph.Cylinder(**d)


def _poly(**kw):
    d = dict(
        axis="z",
        vertices_um=((0.2, 0.9), (1.8, 0.9), (1.8, 1.1), (0.2, 1.1)),
        slab_bounds_um=(0.89, 1.11),
    )
    d.update(kw)
    return ph.PolySlab(**d)


class TestCylinderModel:
    def test_defaults_solid_full_circle(self):
        c = _cyl()
        assert c.inner_radius_um == 0.0
        assert c.angle_start == 0.0
        assert math.isclose(c.angle_stop, 2 * math.pi)

    def test_ring_requires_inner_lt_outer(self):
        _cyl(radius_um=2.0, inner_radius_um=1.5)  # ok: a ring
        with pytest.raises(ValueError):
            _cyl(radius_um=1.0, inner_radius_um=1.0)  # inner == outer
        with pytest.raises(ValueError):
            _cyl(radius_um=1.0, inner_radius_um=2.0)  # inner > outer

    def test_sweep_range(self):
        _cyl(angle_start=0.0, angle_stop=math.pi / 2)  # ok: a 90deg bend
        with pytest.raises(ValueError):
            _cyl(angle_start=1.0, angle_stop=1.0)  # zero sweep
        with pytest.raises(ValueError):
            _cyl(angle_start=0.0, angle_stop=7.0)  # > 2pi


class TestPolySlabModel:
    def test_min_three_vertices(self):
        with pytest.raises(ValueError):
            ph.PolySlab(
                axis="z", vertices_um=((0, 0), (1, 0)), slab_bounds_um=(0, 1)
            )

    def test_slab_bounds_ordered(self):
        with pytest.raises(ValueError):
            _poly(slab_bounds_um=(1.11, 0.89))

    def test_sidewall_angle_range(self):
        _poly(sidewall_angle=0.3)  # ok
        with pytest.raises(ValueError):
            _poly(sidewall_angle=math.pi / 2)  # >= pi/2


def test_geometry_wire_roundtrip():
    sim = ph.Simulation(
        size_um=(2, 2, 2),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=3),
        structures=[
            ph.Structure(
                geometry=_cyl(), medium=ph.Medium(permittivity=12.25)
            ),
            ph.Structure(
                geometry=_poly(sidewall_angle=0.2, reference_plane="bottom"),
                medium=ph.Medium(permittivity=12.25),
            ),
        ],
        sources=[
            ph.PointDipole(
                center_um=(1, 1, 1),
                polarization="Ez",
                source_time=ph.GaussianPulse(
                    freq0_hz=1.934e14, fwidth_hz=4e13
                ),
            )
        ],
    )
    back = ph.Simulation.model_validate_json(sim.model_dump_json())
    assert back == sim
    # the discriminated union preserves the concrete geometry types
    assert back.structures[0].geometry.type == "cylinder"
    assert back.structures[1].geometry.type == "polyslab"
    assert back.structures[1].geometry.reference_plane == "bottom"


@pytest.mark.skipif(find_solver() is None, reason="no built phsolver")
def test_geometry_runs_through_real_solver(tmp_path):
    """A solid cylinder + a tapered PolySlab rasterize and run end-to-end
    through the engine (proves the §17 parse + rasterization path)."""
    sim = ph.Simulation(
        size_um=(2, 2, 2),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=10),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[
            ph.Structure(
                geometry=ph.Cylinder(
                    axis="z", center_um=(1, 1, 1), radius_um=0.4, length_um=2.0
                ),
                medium=ph.Medium(permittivity=12.25),
            ),
            ph.Structure(
                geometry=ph.PolySlab(
                    axis="z",
                    vertices_um=(
                        (0.2, 0.9),
                        (1.8, 0.9),
                        (1.8, 1.1),
                        (0.2, 1.1),
                    ),
                    slab_bounds_um=(0.89, 1.11),
                    sidewall_angle=0.2,
                ),
                medium=ph.Medium(permittivity=12.25),
            ),
        ],
        sources=[
            ph.PointDipole(
                center_um=(1, 1, 1),
                polarization="Ez",
                source_time=ph.GaussianPulse(
                    freq0_hz=1.934e14, fwidth_hz=4e13
                ),
            )
        ],
        monitors=[ph.FieldSnapshotMonitor(name="final", fields=["Ez"])],
    )
    data = ph.run_local(sim, output_dir=tmp_path / "out")
    assert "final" in data
