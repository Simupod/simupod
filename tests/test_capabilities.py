"""Capability gating (simupod.capabilities): representable-but-unsupported
features fail at MODEL CONSTRUCTION with an "available in <version>" message,
never at engine submission — and the client's pinned view of the engine's
``--capabilities`` manifest cannot silently drift from the real binary.
"""

import pytest
from pydantic import ValidationError

import simupod as ph
from simupod import capabilities as caps
from simupod.components.grid import GradedAxisCoords, GradedGridSpec

from conftest import make_pw_sim, make_sim


def _graded_z(n=28, dl0=0.025, ratio=1.04):
    """A strictly-increasing graded coordinate array (origin 0), as §15.1."""
    q, z = [0.0], 0.0
    for k in range(1, n):
        z += dl0 * (ratio ** k)
        q.append(round(z, 7))
    return tuple(q)


# --- the standardized message --------------------------------------------- #

def test_unavailable_builds_a_clear_value_error():
    err = caps.unavailable("graded_plane_wave")
    assert isinstance(err, ValueError)
    assert isinstance(err, caps.UnavailableFeature)
    text = str(err)
    assert "not available in schema v1" in text
    assert "available in" in text
    assert "15.9" in text
    assert err.key == "graded_plane_wave"


# --- the construction-time gate ------------------------------------------- #

def test_graded_grid_plus_plane_wave_rejected_at_construction():
    # Plane wave along z, transverse x/y periodic (required), z graded + PML.
    with pytest.raises(ValidationError) as exc:
        make_pw_sim(
            size_um=(0.2, 0.2, 1.0),
            grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(z=_graded_z())),
        )
    msg = str(exc.value)
    assert "not available in schema v1" in msg
    assert "graded" in msg


def test_transverse_only_graded_plane_wave_also_rejected():
    # Injection axis z stays UNIFORM; only the transverse x grades. §15.9 would
    # eventually permit this, but the engine rejects any graded+plane-wave combo
    # today — the client must mirror the engine, not the aspirational rule.
    with pytest.raises(ValidationError) as exc:
        make_pw_sim(
            size_um=(1.0, 0.2, 1.0),
            grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(x=_graded_z())),
        )
    assert "not available in schema v1" in str(exc.value)


def test_uniform_plane_wave_still_builds():
    # Regression: the gate must not reject a perfectly valid uniform plane wave.
    sim = make_pw_sim()
    assert any(isinstance(s, ph.PlaneWave) for s in sim.sources)


def test_graded_grid_plus_point_dipole_still_builds():
    # Only plane waves are gated on a graded grid; a dipole launch is fine
    # (the silicon-waveguide path depends on exactly this).
    sim = make_sim(
        size_um=(0.2, 0.2, 1.0),
        grid=GradedGridSpec(dl_um=0.05, coords=GradedAxisCoords(z=_graded_z())),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        pml_num_layers=8,
        monitors=[],
    )
    assert isinstance(sim.grid, GradedGridSpec)


# --- the live engine manifest (the "wired to --capabilities" link) -------- #

def test_engine_manifest_matches_the_pinned_set():
    drift = caps.engine_feature_drift()
    if drift is None:
        pytest.skip("no phsolver binary available to read --capabilities")
    assert drift == set(), (
        "phsolver --capabilities drifted from "
        "capabilities.ENGINE_ADVERTISED_FEATURES; update the pin (and the "
        f"client gates) in lockstep. Symmetric difference: {sorted(drift)}")


def test_engine_capabilities_reports_schema_major():
    info = caps.engine_capabilities()
    if info is None:
        pytest.skip("no phsolver binary available")
    assert info.get("schema_major") == caps.SCHEMA_MAJOR
