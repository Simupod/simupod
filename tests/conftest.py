import os
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]   # photonhub/ project dir
REPO_ROOT = PACKAGE_ROOT.parent                       # PhotonHub repo root
EXAMPLE_SPEC = REPO_ROOT / "examples" / "dipole_vacuum.json"
FRESNEL_SPEC = REPO_ROOT / "examples" / "fresnel_slab.json"


@pytest.fixture
def example_spec_path() -> Path:
    if not EXAMPLE_SPEC.is_file():
        pytest.skip(f"golden example not found: {EXAMPLE_SPEC}")
    return EXAMPLE_SPEC


@pytest.fixture
def fresnel_spec_path() -> Path:
    if not FRESNEL_SPEC.is_file():
        pytest.skip(f"golden example not found: {FRESNEL_SPEC}")
    return FRESNEL_SPEC


@pytest.fixture
def subprocess_env() -> dict:
    """Environment for `python -m simupod.schema ...` subprocesses so the
    in-tree package is importable without installation."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def make_sim(**overrides):
    import simupod as ph

    kwargs = dict(
        size_um=(0.2, 0.2, 0.2),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        # This is a 4-cell-per-axis domain; the schema-1.12 default (PML on all
        # faces) cannot fit a 12-layer slab, so pin periodic explicitly. (Old
        # default was periodic too, so make_sim's wire output is unchanged.)
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="periodic"),
        sources=[
            ph.PointDipole(
                center_um=(0.1, 0.1, 0.1),
                polarization="Ez",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13),
            )
        ],
        monitors=[
            ph.FieldTimeMonitor(name="probe", center_um=(0.15, 0.1, 0.1), fields=["Ez"]),
            ph.FieldSnapshotMonitor(name="final", fields=["Ez", "Hx"]),
        ],
    )
    kwargs.update(overrides)
    return ph.Simulation(**kwargs)


def make_pw_sim(**overrides):
    """Tiny plane-wave simulation (NUMERICS.md section 13): propagation
    along z, transverse axes periodic (required), PML on z."""
    import simupod as ph

    kwargs = dict(
        size_um=(0.2, 0.2, 0.4),
        grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        boundaries=ph.Boundaries(x="periodic", y="periodic", z="pml"),
        sources=[
            ph.PlaneWave(
                axis="z",
                direction="+",
                position_um=0.1,
                polarization="Ex",
                source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=3.0e13),
            )
        ],
        monitors=[],
    )
    kwargs.update(overrides)
    return ph.Simulation(**kwargs)


@pytest.fixture
def tiny_sim():
    return make_sim()
