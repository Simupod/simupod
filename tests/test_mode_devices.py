"""Mode source + mode monitor end-to-end (NUMERICS.md §18 + the mode-overlap
transmission readout): build an FDE mode, inject it into a straight silicon
waveguide, and recover its transmission. The skipped integration test is the
Phase-2 MVP headline — shapes in -> mode-resolved transmission out."""

import math

import pytest

import simupod as ph
from simupod import auto_grid
from simupod.components.grid import (
    GradedAxisCoords,
    GradedGridSpec,
    graded_primary_spacings,
)
from simupod.plugins import ModeSolver, mode_monitor, mode_source, transmission
from simupod.plugins.mode_devices import _axis_cell_centers
from simupod.runners.local import find_solver

F0 = 1.934e14  # Hz, ~1.55 um
DL = 0.04  # um
N_CORE = 3.5
N_CLAD = 1.444


def _waveguide_sim(*, lz_um=4.0, wt_um=1.6):
    """A straight SOI strip: Si core (0.45 x 0.22 um) along z in SiO2, uniform
    grid, PML on all axes. No source yet (callers add the mode source)."""
    core = ph.Structure(
        geometry=ph.Box(
            center_um=(wt_um / 2, wt_um / 2, lz_um / 2),
            size_um=(0.45, 0.22, lz_um * 4),  # full z (through the z PMLs)
        ),
        medium=ph.Medium(permittivity=N_CORE**2),
    )
    return dict(
        size_um=(wt_um, wt_um, lz_um),
        grid=ph.UniformGridSpec(dl_um=DL),
        run=ph.RunSpec(n_steps=1200),
        background=ph.Background(permittivity=N_CLAD**2),
        boundaries=ph.Boundaries(x="pml", y="pml", z="pml"),
        structures=[core],
    )


def _te0_mode():
    solver = ModeSolver.from_rectangular_core(
        wavelength_um=1.55,
        dl_um=DL,
        core_w_um=0.45,
        core_h_um=0.22,
        n_core=N_CORE,
        n_clad=N_CLAD,
    )
    return solver.solve(num_modes=1, polarization="TE", n_guess=3.0)[0]


def test_mode_source_builder_produces_valid_modesource():
    base = _waveguide_sim()
    mode = _te0_mode()
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)
    src = mode_source(
        # a sim shell just to read the grid (needs a source to construct, so
        # build the dict and hand the eventual sim's grid/size via a stand-in)
        ph.Simulation(
            **base,
            sources=[
                ph.PointDipole(
                    center_um=(0.8, 0.8, 0.8),
                    polarization="Ex",
                    source_time=pulse,
                )
            ],
        ),
        mode,
        axis="z",
        position_um=0.8,
        source_time=pulse,
    )
    assert isinstance(src, ph.ModeSource)
    assert src.nu * src.nv == len(src.profile)
    assert src.polarization in ("Ex", "Ey")  # tangential to z
    assert math.isclose(src.n_eff, mode.n_eff)
    assert max(abs(p) for p in src.profile) == pytest.approx(1.0)  # peak-norm


def test_mode_source_wire_roundtrip():
    base = _waveguide_sim()
    mode = _te0_mode()
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)
    shell = ph.Simulation(
        **base,
        sources=[
            ph.PointDipole(
                center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse
            )
        ],
    )
    src = mode_source(shell, mode, axis="z", position_um=0.8, source_time=pulse)
    sim = ph.Simulation(**base, sources=[src])
    back = ph.Simulation.model_validate_json(sim.model_dump_json())
    assert back.sources[0].type == "mode_source"
    assert back == sim


@pytest.mark.skipif(find_solver() is None, reason="no built phsolver")
def test_straight_waveguide_mode_transmission_is_near_one():
    """MVP headline: inject the TE0 mode, measure mode-resolved transmission
    between an input and an output plane. A clean lossless straight guide passes
    the mode forward with T ≈ 1, and the backward (reflected) mode is small."""
    base = _waveguide_sim(lz_um=4.0)
    mode = _te0_mode()
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)

    shell = ph.Simulation(
        **base,
        sources=[
            ph.PointDipole(
                center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse
            )
        ],
    )
    src = mode_source(shell, mode, axis="z", position_um=0.8, source_time=pulse)
    mm_in = mode_monitor(
        shell, mode, axis="z", position_um=1.2, freqs_hz=[F0], name="in"
    )
    mm_out = mode_monitor(
        shell, mode, axis="z", position_um=3.2, freqs_hz=[F0], name="out"
    )
    sim = ph.Simulation(
        **base,
        sources=[src],
        monitors=[mm_in.field_monitor, mm_out.field_monitor],
    )

    data = ph.run_local(sim)
    t = transmission(mm_out, mm_in, data)
    f = next(iter(t))
    t_fwd = t[f]
    # Reflection: backward mode power at the input vs forward there.
    r = mm_in.mode_power(data, direction="-")[f] / mm_in.mode_power(data)[f]

    assert math.isfinite(t_fwd)
    # Lossless straight guide: near-unity forward transmission.
    assert 0.8 < t_fwd < 1.1, f"forward transmission not ~1: {t_fwd}"
    # Low back-reflection into the mode.
    assert r < 0.1, f"unexpectedly high modal reflection: {r}"


@pytest.mark.skipif(find_solver() is None, reason="no built phsolver")
def test_straight_waveguide_transmission_spectrum_is_near_one():
    """Broadband (multi-frequency) variant of the headline check — the
    acceptance assembly's core capability: ONE Gaussian-pulse run yields the
    whole transmission spectrum via the engine's running DFT (a phasor per
    requested frequency, NUMERICS.md §12). A clean straight guide reads T ≈ 1 at
    every point. This is the fast regression behind
    ``benchmarks/acceptance/run_acceptance.py``; that script sweeps the full
    1310–1410 nm / 101-pt band over the six library components."""
    base = _waveguide_sim(lz_um=4.0)
    mode = _te0_mode()
    # three points spanning a ±5 % band; one broadband pulse covers them all.
    freqs = [0.95 * F0, F0, 1.05 * F0]
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)

    shell = ph.Simulation(
        **base,
        sources=[
            ph.PointDipole(
                center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse
            )
        ],
    )
    src = mode_source(shell, mode, axis="z", position_um=0.8, source_time=pulse)
    mm_in = mode_monitor(
        shell, mode, axis="z", position_um=1.2, freqs_hz=freqs, name="in"
    )
    mm_out = mode_monitor(
        shell, mode, axis="z", position_um=3.2, freqs_hz=freqs, name="out"
    )
    sim = ph.Simulation(
        **base,
        sources=[src],
        monitors=[mm_in.field_monitor, mm_out.field_monitor],
    )

    data = ph.run_local(sim)
    t = transmission(mm_out, mm_in, data)  # {freq_hz: T} across the band

    assert len(t) == len(freqs), f"expected {len(freqs)} spectral points, got {len(t)}"
    p_fwd = mm_in.mode_power(data)
    p_bwd = mm_in.mode_power(data, direction="-")
    for f in freqs:
        key = min(t, key=lambda k: abs(k - f))  # tolerate fp on the freq labels
        assert math.isfinite(t[key])
        assert 0.8 < t[key] < 1.1, f"T(f={f:.3e}) not ~1: {t[key]}"
        assert p_bwd[key] / p_fwd[key] < 0.1, f"high reflection at f={f:.3e}"


# --------------------------------------------------------------------------- #
# Transverse-graded mode source (automesh enablement). The §18 aux line is 1-D
# along the propagation axis at the scalar dl, but the TF/SF corrections inject
# the transverse profile PER PLANE CELL — so the transverse plane may grade while
# the propagation axis stays uniform. This is what lets ``auto_grid`` drive a
# waveguide device: the oxide cladding / PML coarsen while the Si core stays fine.
# --------------------------------------------------------------------------- #


def _transverse_graded(base, *, snap=False):
    """``base`` with x,y (transverse to the z propagation) graded by auto_grid
    around the Si core; z stays uniform at DL for the aux line. steps_per_wvl is
    tuned so the core cell ~ DL (min spacing ~ DL, dt ~ the uniform run's)."""
    core = base["structures"][0]
    g = auto_grid(size_um=base["size_um"], wavelength_um=1.55, structures=[core],
                  background_index=N_CLAD, steps_per_wvl=1.55 / (N_CORE * DL),
                  max_grading=1.3, axes="xy", refine_pad_um=0.4,
                  snap_interfaces=snap)
    grid = GradedGridSpec(dl_um=DL,
                          coords=GradedAxisCoords(x=g.coords.x, y=g.coords.y))
    return {**base, "grid": grid}


def _graded_z_coords(lz_um, dl):
    """A genuinely non-uniform z (the propagation axis) spanning ~lz_um — used
    to check the engine rejects a GRADED propagation axis for a mode source."""
    cells, s = [], dl
    while sum(cells) < lz_um:
        cells.append(s)
        s = min(s * 1.08, dl * 1.6)
    scale = lz_um / sum(cells)
    coords, acc = [0.0], 0.0
    for c in cells[:-1]:
        acc += c * scale
        coords.append(round(acc, 7))
    return tuple(coords)


def test_axis_cell_centers_graded_returns_dual_node_midpoints():
    import numpy as np
    base = _transverse_graded(_waveguide_sim())
    sim = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(0.8, 0.8, 0.8), polarization="Ex",
        source_time=ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13))])
    qx = sim.grid.coords.x
    dqx = graded_primary_spacings(qx)
    cx = _axis_cell_centers(sim, "x")
    # cell i center = q[i] + dq[i]/2 (the §15.2 dual node), one per cell.
    assert len(cx) == len(qx)
    assert np.allclose(cx, np.asarray(qx) + np.asarray(dqx) / 2.0)
    # genuinely non-uniform (the cladding coarsened away from the core).
    assert max(dqx) > min(dqx) * 1.3
    # a uniform axis (z, the propagation axis) still uses (i+0.5)*dl.
    assert np.allclose(np.diff(_axis_cell_centers(sim, "z")), DL)


def test_mode_source_on_transverse_graded_grid_matches_cell_counts():
    base = _transverse_graded(_waveguide_sim())
    mode = _te0_mode()
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)
    shell = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse)])
    # This call used to raise ("graded mode injection is deferred"); now it
    # resamples the mode onto the graded transverse cells.
    src = mode_source(shell, mode, axis="z", position_um=0.8, source_time=pulse)
    assert isinstance(src, ph.ModeSource)
    # nu, nv match the GRADED transverse cell counts (x, y), not a uniform count.
    assert src.nu == len(base["grid"].coords.x)
    assert src.nv == len(base["grid"].coords.y)
    assert src.nu * src.nv == len(src.profile)


@pytest.mark.skipif(find_solver() is None, reason="no built phsolver")
def test_transverse_graded_straight_guide_transmission_near_one():
    """The headline check on an AUTOMESH grid: inject TE0 on a transverse-graded
    mesh (x,y graded, z uniform) and recover near-unity forward transmission. A
    profile sampled at the wrong (uniform) cell positions would launch a
    mismatched field and read T far from 1; this passing is the proof the graded
    cell-center sampling + the engine's per-cell injection are consistent."""
    base = _transverse_graded(_waveguide_sim(lz_um=4.0))
    # shutoff-terminate so the run reaches field decay regardless of dt.
    base = {**base, "run": ph.RunSpec(n_steps=8000, shutoff=1e-5)}
    mode = _te0_mode()
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)
    shell = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse)])
    src = mode_source(shell, mode, axis="z", position_um=0.8, source_time=pulse)
    mm_in = mode_monitor(shell, mode, axis="z", position_um=1.2, freqs_hz=[F0], name="in")
    mm_out = mode_monitor(shell, mode, axis="z", position_um=3.2, freqs_hz=[F0], name="out")
    sim = ph.Simulation(**base, sources=[src],
                        monitors=[mm_in.field_monitor, mm_out.field_monitor])
    data = ph.run_local(sim)
    t = transmission(mm_out, mm_in, data)
    t_fwd = t[next(iter(t))]
    assert math.isfinite(t_fwd)
    assert 0.8 < t_fwd < 1.1, f"transverse-graded forward transmission not ~1: {t_fwd}"


@pytest.mark.skipif(find_solver() is None, reason="no built phsolver")
def test_mode_source_graded_propagation_axis_is_rejected():
    """The §18 aux line is 1-D along the propagation axis at the scalar dl, so a
    GRADED propagation axis is rejected by the engine (only the transverse plane
    may grade)."""
    from simupod.runners.local import SolverRunError

    base = _waveguide_sim()
    lz = base["size_um"][2]
    grid = GradedGridSpec(dl_um=DL,
                          coords=GradedAxisCoords(z=_graded_z_coords(lz, DL)))
    base = {**base, "grid": grid, "run": ph.RunSpec(n_steps=50)}
    mode = _te0_mode()
    pulse = ph.GaussianPulse(freq0_hz=F0, fwidth_hz=4e13)
    shell = ph.Simulation(**base, sources=[ph.PointDipole(
        center_um=(0.8, 0.8, 0.8), polarization="Ex", source_time=pulse)])
    src = mode_source(shell, mode, axis="z", position_um=0.8, source_time=pulse)
    sim = ph.Simulation(**base, sources=[src])
    with pytest.raises(SolverRunError, match="propagation axis"):
        ph.run_local(sim)
