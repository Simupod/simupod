"""Unit tests for the component-benchmark per-wavelength scoring.

Pure numpy — no engine, no tidy3d — so this gates the benchmark's loss function
(max|ΔT|, rms, bias, pass/fail, interpolation, skipped ports) without a solver.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SC_PATH = REPO_ROOT / "benchmarks" / "components" / "scoring.py"


def _load_scoring():
    if not _SC_PATH.is_file():
        pytest.skip(f"scoring.py not found at {_SC_PATH}")
    spec = importlib.util.spec_from_file_location("components_scoring", _SC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_scoring()

LAM = [1260.0, 1280.0, 1300.0, 1320.0, 1340.0, 1360.0]


def test_exact_match_zero_error():
    T = [0.99, 0.98, 0.97, 0.98, 0.99, 1.00]
    m = sc.score_port(LAM, T, LAM, T)
    assert m["max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert m["rms_abs"] == pytest.approx(0.0, abs=1e-12)
    assert m["bias"] == pytest.approx(0.0, abs=1e-12)
    assert m["n_points"] == len(LAM)


def test_constant_offset_bias_and_max():
    ref = [0.90] * len(LAM)
    ph = [0.91] * len(LAM)  # +0.01 everywhere
    m = sc.score_port(LAM, ph, LAM, ref)
    assert m["max_abs"] == pytest.approx(0.01)
    assert m["rms_abs"] == pytest.approx(0.01)
    assert m["bias"] == pytest.approx(+0.01)
    assert m["max_rel"] == pytest.approx(0.01 / 0.90, rel=1e-6)


def test_max_located_and_signed_bias():
    ref = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    ph = [0.50, 0.50, 0.50, 0.50, 0.50, 0.44]  # -0.06 only at the last point
    m = sc.score_port(LAM, ph, LAM, ref)
    assert m["max_abs"] == pytest.approx(0.06)
    assert m["max_abs_at_nm"] == pytest.approx(1360.0)
    assert m["bias"] == pytest.approx(-0.01)  # -0.06/6


def test_interpolation_onto_ph_grid():
    # reference sampled on a coarser, offset grid still scores on the ph grid.
    ref_lam = [1260.0, 1360.0]
    ref_T = [0.0, 1.0]  # linear ramp
    ph_lam = [1260.0, 1310.0, 1360.0]
    ph_T = [0.0, 0.5, 1.0]  # matches the ramp exactly
    m = sc.score_port(ph_lam, ph_T, ref_lam, ref_T)
    assert m["max_abs"] == pytest.approx(0.0, abs=1e-12)


def test_no_extrapolation_outside_reference_span():
    ref_lam = [1300.0, 1320.0]
    ref_T = [0.5, 0.5]
    ph_lam = [1260.0, 1310.0, 1360.0]  # only 1310 is inside
    ph_T = [9.9, 0.5, 9.9]  # the outside points would be huge errors if scored
    m = sc.score_port(ph_lam, ph_T, ref_lam, ref_T)
    assert m["n_points"] == 1
    assert m["max_abs"] == pytest.approx(0.0, abs=1e-12)


def test_score_component_pass_fail_and_skip():
    ref = {"out": (LAM, [0.99] * len(LAM)),
           "through": (LAM, [0.70] * len(LAM))}
    ph = {"out": (LAM, [0.991] * len(LAM)),       # within 0.2%
          "through": (LAM, [0.74] * len(LAM)),    # 4% off -> fails
          "cross": (LAM, [0.01] * len(LAM))}      # no reference -> skipped
    passed, ports = sc.score_component(ph, ref, tol_abs=0.002, gate="max")
    assert not passed                              # one port fails -> component fails
    assert ports["out"]["pass"] is True
    assert ports["through"]["pass"] is False
    assert ports["cross"]["status"].startswith("skipped")


def test_score_component_all_pass():
    ref = {"out": (LAM, [0.99] * len(LAM))}
    ph = {"out": (LAM, [0.9905] * len(LAM))}       # 0.05% < 0.2%
    passed, ports = sc.score_component(ph, ref, tol_abs=0.002, gate="max")
    assert passed and ports["out"]["pass"]


def test_band_center_metric():
    # ref flat 0.99; ph matches at centre (1310) but has big band-edge error.
    ref_T = [0.99] * len(LAM)
    ph_T = [0.90, 0.99, 0.99, 0.99, 0.99, 0.90]  # edges off 9%, centre exact
    m = sc.score_port(LAM, ph_T, LAM, ref_T, center_nm=1310.0, center_halfwidth_nm=25.0)
    assert m["max_abs"] == pytest.approx(0.09)          # full-band sees the edges
    assert m["center_abs"] == pytest.approx(0.0, abs=1e-9)  # centre is clean


def test_center_gate_passes_when_edges_fail():
    # The realistic gate: pass on band-centre even though the full band is off.
    ref = {"out": (LAM, [0.99] * len(LAM))}
    ph = {"out": (LAM, [0.90, 0.99, 0.99, 0.99, 0.99, 0.90])}
    p_center, ports_c = sc.score_component(ph, ref, tol_abs=0.002,
                                           center_nm=1310.0, gate="center")
    p_max, ports_m = sc.score_component(ph, ref, tol_abs=0.002, gate="max")
    assert p_center and ports_c["out"]["pass"]      # centre clean -> pass
    assert not p_max and not ports_m["out"]["pass"]  # full-band edges -> fail
