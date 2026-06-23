# photonhub

Python client for the PhotonHub FDTD solver. The pydantic models in
`simupod.components` are the single source of truth for the simulation
JSON wire format (`schemas/GOVERNANCE.md`); `schemas/simulation_v1.json` is
generated from them via `python -m simupod.schema emit`.

```python
import simupod as ph

sim = ph.Simulation(
    size_um=(4.0, 4.0, 4.0),
    grid=ph.UniformGridSpec(dl_um=0.05),
    run=ph.RunSpec(run_time_s=8.0e-14),
    sources=[
        ph.PointDipole(
            center_um=(2.0, 2.0, 2.0),
            polarization="Ez",
            source_time=ph.GaussianPulse(freq0_hz=1.934e14, fwidth_hz=4.0e13),
        )
    ],
    monitors=[
        ph.FieldTimeMonitor(name="probe", center_um=(3.0, 2.0, 2.0), fields=["Ez"]),
        ph.FieldSnapshotMonitor(name="final", fields=["Ex", "Ey", "Ez"]),
    ],
)

data = ph.run_local(sim)        # finds phsolver, runs it, parses outputs
probe = data["probe"]           # xarray.DataArray, dims ('t', 'component')
```

## What you can do today

Shipped surface as of schema **v1.6.0-alpha.1** (validated non-dispersive solver
core plus the silicon-PIC MVP — see `docs/quickstart.md` and the notebook
gallery under `examples/`):

- **Run a simulation:** `ph.run_local(sim)` (subprocess + file protocol;
  `device="cpu"|"gpu"|"gpu:N"`), `ph.run_async(sim)` and `ph.Batch(...)` for many
  sims in flight.
- **Know the cost first:** `ph.cost_estimate(sim)` returns a dollar estimate
  (Tcell-step billing) *before* you press run.
- **Geometry:** `Box`, `Sphere`, `Cylinder` (full, or an annular sector / ring
  via `inner_radius_um` + `angle_start` / `angle_stop`), and `PolySlab` (an
  extruded polygon — e.g. a taper).
- **Sources:** `PointDipole`, `PlaneWave` (normal-incidence TF/SF), and
  `ModeSource` (waveguide-eigenmode TF/SF injection), driven by a `GaussianPulse`
  source time.
- **Monitors:** `FieldTimeMonitor`, `FieldSnapshotMonitor`, `FieldDftMonitor`,
  and `FluxMonitor` — fp64 DFT field and flux power.
- **Component library:** `ph.library.straight / bend / taper / crossing /
  coupler / ring` — each returns a `Component` (structures + ports) in ~one line.
- **Mode-resolved transmission:** the `simupod.plugins` pipeline —
  `ModeSolver` → `mode_source` → `mode_monitor` → `transmission(out, in, data)`
  returns `{freq_hz: T}`. Broadband: one Gaussian-pulse run yields the whole
  `T(λ)` spectrum via the running DFT. One-mode-at-a-time power transmission, not
  an S-matrix.
- **Meshing:** `UniformGridSpec`, or `GradedGridSpec` + `auto_grid` for a
  cells-per-λ graded mesh.
- **Mode solving:** the FDE `ModeSolver` plugin (semi-vectorial, straight
  waveguides) for `n_eff` and mode profiles.
- **Visualization:** `Simulation.plot` / `plot_eps` (draws Box / Sphere /
  Cylinder / PolySlab) / `plot_3d`, `SimulationData.plot_field`, and the
  module-level `simupod.viz.plot_mode` (FDE mode heatmap) and
  `simupod.viz.plot_spectrum` (`T` vs λ).
- **Export:** HDF5 converter for the parsed results.
- **Correctness aids:** capability gating rejects unsupported features at
  construction; optional volume-fraction subpixel smoothing (Box / Cylinder /
  PolySlab, default-off).

### Limits (today)

This is a validated **non-dispersive** solver core plus the silicon-PIC MVP:

- Media are **non-dispersive** (relative permittivity + Ohmic conductivity).
- Mode-resolved transmission is **one-mode-at-a-time power**, not a full
  S-matrix.
- The mode-source **GPU** path compiles but its numerical CPU↔GPU equivalence is
  pending hardware verification; the CPU path is the validated reference.

Net: shapes go in and **mode-resolved transmission** comes out — solve the
eigenmode, inject it with `ModeSource`, ratio two `mode_monitor` planes, and plot
`T(λ)`. Dispersive media, GDS import, and a full S-matrix are Phase 3.

## Learn more

- `docs/quickstart.md` — end-to-end first run.
- `examples/` — 7-notebook gallery covering the shipped surface.
