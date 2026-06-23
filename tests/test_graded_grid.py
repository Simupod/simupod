"""GradedGridSpec (NUMERICS.md section 15) — model, validators, wire round-trip."""

import math

import pytest

import photonhub as ph
from photonhub import (
    GaussianPulse,
    GradedAxisCoords,
    GradedGridSpec,
    PointDipole,
    Simulation,
)
from photonhub.components.grid import graded_primary_spacings


def _stretched(n, dl0=0.025, ratio=1.04):
    q = [0.0]
    z = 0.0
    for k in range(1, n):
        z += dl0 * (ratio ** k)
        q.append(round(z, 7))
    return tuple(q)


def _graded_sim(coords: GradedAxisCoords, **grid_kw):
    return Simulation(
        size_um=(0.8, 0.8, 2.0),
        grid=GradedGridSpec(dl_um=0.05, coords=coords, **grid_kw),
        run={"n_steps": 20},
        boundaries={"x": "periodic", "y": "periodic", "z": "pml"},
        pml_num_layers=8,
        sources=[PointDipole(center_um=(0.4, 0.4, 0.5), polarization="Ex",
                             source_time=GaussianPulse(freq0_hz=3e14,
                                                       fwidth_hz=6e13))],
    )


def test_graded_grid_builds_and_round_trips():
    sim = _graded_sim(GradedAxisCoords(z=_stretched(28)))
    assert sim.schema_version == "1.12.0-alpha.1"
    wire = sim.to_wire_dict()
    assert wire["grid"]["type"] == "graded"
    assert "z" in wire["grid"]["coords"]
    # Omitted axes are not serialized (exclude_none).
    assert "x" not in wire["grid"]["coords"]
    sim2 = Simulation.from_wire_json(sim.to_wire_json())
    assert isinstance(sim2.grid, GradedGridSpec)
    assert sim2.grid.coords.z == sim.grid.coords.z


def test_realized_length_is_closing_node():
    z = _stretched(28)
    sim = _graded_sim(GradedAxisCoords(z=z))
    # Graded z realized length = closing node = q[-1] + last spacing.
    dq = graded_primary_spacings(z)
    expected_z = z[-1] + dq[-1]
    assert math.isclose(sim._realized_um()[2], expected_z, rel_tol=1e-12)
    # x/y omitted -> uniform realized from size/dl.
    assert math.isclose(sim._realized_um()[0], 16 * 0.05, rel_tol=1e-12)


def test_at_least_one_axis_required():
    with pytest.raises(ValueError, match="lists no axis"):
        GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords())


def test_too_few_nodes_rejected():
    with pytest.raises(ValueError, match=">= 4"):
        GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(x=(0.0, 0.05, 0.1)))


def test_must_start_at_origin():
    with pytest.raises(ValueError, match="must start at 0"):
        GradedGridSpec(dl_um=0.05,
                       coords=GradedAxisCoords(x=(0.1, 0.2, 0.3, 0.4)))


def test_strictly_increasing_required():
    with pytest.raises(ValueError, match="strictly increasing"):
        GradedGridSpec(dl_um=0.05,
                       coords=GradedAxisCoords(x=(0.0, 0.05, 0.05, 0.1)))


def test_ratio_guard():
    # max/min spacing ratio 100 >> 10 guard.
    with pytest.raises(ValueError, match="ratio"):
        GradedGridSpec(dl_um=0.05,
                       coords=GradedAxisCoords(x=(0.0, 0.001, 0.002, 0.102)))


def test_ratio_at_guard_boundary_accepted():
    # Exactly at the guard (10x) is allowed.
    GradedGridSpec(dl_um=0.05,
                   coords=GradedAxisCoords(x=(0.0, 0.01, 0.02, 0.12)))


def test_two_axes_graded():
    sim = _graded_sim(GradedAxisCoords(x=_stretched(16), z=_stretched(28)))
    wire = sim.to_wire_dict()
    assert set(wire["grid"]["coords"]) == {"x", "z"}


def _real_solver():
    from photonhub.runners.local import find_solver
    try:
        return find_solver()
    except ph.SolverRunError:
        return None


@pytest.mark.skipif(_real_solver() is None,
                    reason="no phsolver binary found (build the engine first)")
def test_graded_sim_runs_end_to_end(tmp_path):
    """Full stack: a graded-z sim built in Python runs through the real
    phsolver and its manifest carries the §15.10 coordinate arrays."""
    from photonhub.components import FieldTimeMonitor
    z = _stretched(28)
    sim = Simulation(
        size_um=(0.8, 0.8, 2.0),
        grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(z=tuple(z))),
        run={"n_steps": 60},
        boundaries={"x": "periodic", "y": "periodic", "z": "pml"},
        pml_num_layers=8,
        sources=[PointDipole(center_um=(0.4, 0.4, 0.5), polarization="Ex",
                             source_time=GaussianPulse(freq0_hz=3e14,
                                                       fwidth_hz=6e13))],
        monitors=(FieldTimeMonitor(name="p", center_um=(0.4, 0.4, 0.6),
                                   fields=("Ex",)),),
    )
    data = ph.run_local(sim, output_dir=tmp_path / "out", timeout=300)
    grid = data.manifest["grid"]
    assert "coords_um" in grid and len(grid["coords_um"]["z"]) == 28
    # Graded realized z = closing node, NOT shape*dl (= 1.4).
    assert grid["size_um"][2] < 1.4
    probe = data["p"]
    import numpy as np
    assert np.isfinite(probe.values).all()
    assert float(np.max(np.abs(probe.values))) > 0.0
