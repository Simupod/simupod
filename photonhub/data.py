"""Load a phsolver output directory into xarray.

Output contract (NUMERICS.md section 6, engine/include/phcore/output.h): one
raw little-endian float32 ``.bin`` per monitor plus ``manifest.json``::

    {
      "manifest_version": "1",          # OUTPUT contract version (gate here)
      "schema_version": "1.1.0-alpha.1",  # echo of the INPUT spec version
      "monitors": [
        {"name": "probe", "type": "field_time", "file": "probe.bin",
         "dtype": "float32", "shape": [n_samples, n_components],
         "dims": ["sample", "component"], "components": ["Ez"],
         "sample_steps": [1, 2, 3], "dt_s": 1.234e-17},
        {"name": "final", "type": "field_snapshot", "file": "final.bin",
         "dtype": "float32", "shape": [n_samples, n_comp, nz, ny, nx],
         "dims": ["sample", "component", "z", "y", "x"],
         "components": ["Ex", "Ez"], "sample_steps": [1600],
         "dt_s": 1.234e-17},
        {"name": "slab", "type": "field_dft", "file": "slab.bin",
         "dtype": "float32", "shape": [n_freqs, n_comp, nz, ny, nx, 2],
         "dims": ["freq", "component", "z", "y", "x", "complex"],
         "components": ["Ex", "Hy"], "freqs_hz": [1.934e14],
         "origin_cells": [i0, j0, k0],   # optional; region low corner (x,y,z)
         "dt_s": 1.234e-17},
        {"name": "reflection", "type": "flux", "file": "reflection.bin",
         "dtype": "float32", "shape": [n_freqs], "dims": ["freq"],
         "axis": "z", "freqs_hz": [1.784e14, 1.934e14], "dt_s": 1.234e-17}
      ],
      "run": {"n_steps": 1000, "steps_run": 1000, "dt_s": 1.234e-17,
              "wall_seconds": 1.2, "mcells_per_s": 800.0, "aborted": false,
              "abort_reason": "", "shut_off": false},
      "grid": {"shape": [nx, ny, nz], "dl_um": 0.05, "size_um": [...]},
      "provenance": {"solver_version": "...", "device_name": "...", ...}
    }

Frequency-domain monitors (NUMERICS.md section 12) are emitted as float32
``[re, im]`` pairs in binary order ``[freq][component][k][j][i][re,im]``
(de-pitched) and reconstructed here as ``complex64`` DataArrays with dims
``('f', 'component', 'z', 'y', 'x')``; flux monitors are one float32 power
per frequency, dims ``('f',)``, positive toward +axis. Both carry the
section-12 normalization — phasors divided by ``A0 * S(f)`` (the first
wire-order source's amplitude times its unit-amplitude analytic spectrum),
flux therefore scaled by ``1/|A0*S(f)|^2`` — recorded in each DataArray's
``normalization`` attr.

Aborted runs (NUMERICS.md section 7: ``divergence`` / ``non_finite_energy``)
still write a complete manifest with truncated monitor data; loading one
emits a ``UserWarning`` and surfaces ``aborted`` / ``abort_reason`` here and
in every DataArray's attrs, so partial fields are never silently presented
as healthy data. (``run_local`` raises before loading; the direct
``SimulationData(path)`` post-mortem path warns instead so diverged runs
stay inspectable.)

Snapshot binaries are de-pitched (rows of exactly nx), sample-major order
``[sample][component][k][j][i]``. ``sample_steps`` stores ``step = s + 1``,
so E-field sample times are ``step * dt_s`` (H lags by ``dt_s / 2``; raw
non-colocated Yee values in Phase 0).
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import xarray as xr

_TIME_DIMS = ("t", "component")
_SNAPSHOT_DIMS = ("t", "component", "z", "y", "x")
_DFT_DIMS = ("f", "component", "z", "y", "x")
_FLUX_DIMS = ("f",)

# NUMERICS.md section 12 normalization, surfaced on every frequency-domain
# DataArray so absolute-magnitude use is never silent.
_DFT_NORMALIZATION = (
    "phasors normalized by A0*S(f): the first wire-order source's amplitude "
    "times the unit-amplitude analytic pulse spectrum (NUMERICS.md section "
    "12, e^{-i omega t} convention)")
_FLUX_NORMALIZATION = (
    "power normalized by 1/|A0*S(f)|^2 (shared normalized phasors, "
    "NUMERICS.md section 12) — NOT absolute watts; positive values flow "
    "toward +axis")


class SimulationData:
    """Lazy, dict-like view of one solver output directory.

    ``data["probe"]`` returns an :class:`xarray.DataArray`:

    - time series: dims ``('t', 'component')``, ``t`` in seconds;
    - snapshots: dims ``('t', 'component', 'z', 'y', 'x')``, spatial
      coordinates in microns (Yee-node base coordinates, ``i * dl_um``).
    """

    def __init__(self, path: Union[str, Path]):
        path = Path(path)
        # Source can be a raw-output directory / manifest.json, OR a single
        # HDF5 file (photonhub.hdf5) / a directory holding one. HDF5 reuses
        # every reconstruction path below: only the raw-blob read differs.
        self.manifest_path: Optional[Path] = None
        self._h5_path: Optional[Path] = None
        self.manifest: dict = self._open(path)

        # Output-contract version gate (distinct from the input-spec echo in
        # "schema_version"); absent in older draft manifests => assume v1.
        mv = str(self.manifest.get("manifest_version", "1"))
        if mv.split(".", 1)[0] != "1":
            raise ValueError(
                f"unsupported manifest_version {mv!r} in {self._source}; "
                "this reader supports major version 1 only"
            )

        self._run: dict = dict(self.manifest.get("run", {}))
        if self._run.get("aborted"):
            warnings.warn(
                "loading output of an ABORTED run (reason: "
                f"{self._run.get('abort_reason') or 'unknown'}); monitor data "
                "may be partial or non-finite",
                UserWarning, stacklevel=2)

        self._entries: Dict[str, dict] = {}
        for entry in self.manifest.get("monitors", []):
            name = entry.get("name")
            if not name:
                raise ValueError(f"manifest monitor entry without a name: {entry}")
            self._entries[name] = entry
        self._cache: Dict[str, xr.DataArray] = {}

    @property
    def _source(self) -> Path:
        """The file this view was loaded from (the .h5 or the manifest.json),
        for error messages."""
        return self._h5_path or self.manifest_path

    def _open(self, path: Path) -> dict:
        """Resolve ``path`` to a manifest dict, choosing the raw-directory or
        HDF5 backend and setting output_dir / manifest_path / _h5_path."""
        if path.is_file() and path.suffix == ".h5":
            return self._open_h5(path)
        # A directory holding an .h5 but no manifest.json is the HDF5 case too.
        if path.is_dir() and not (path / "manifest.json").is_file():
            h5s = sorted(path.glob("*.h5"))
            if h5s:
                return self._open_h5(h5s[0])
        self.manifest_path = (path if path.name == "manifest.json"
                              else path / "manifest.json")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json or .h5 found at: {path}")
        self.output_dir = self.manifest_path.parent
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _open_h5(self, h5_path: Path) -> dict:
        import h5py

        self._h5_path = h5_path
        self.output_dir = h5_path.parent
        with h5py.File(h5_path, "r") as f:
            fmt = f.attrs.get("format")
            if fmt is not None and not str(fmt).startswith("photonhub-hdf5-1"):
                raise ValueError(
                    f"unsupported HDF5 format {fmt!r} in {h5_path}; this reader "
                    "supports photonhub-hdf5-1")
            mj = f.attrs.get("manifest_json")
            if mj is None:
                raise ValueError(
                    f"{h5_path} is not a PhotonHub HDF5 file (no manifest_json "
                    "attribute); convert one with photonhub.convert_to_hdf5")
            return json.loads(mj)

    @property
    def dt_s(self) -> float:
        dt = self.manifest.get("run", {}).get("dt_s")
        if dt is None:
            for entry in self._entries.values():
                if "dt_s" in entry:
                    dt = entry["dt_s"]
                    break
        if dt is None:
            raise ValueError(f"manifest has no 'dt_s' key: {self._source}")
        return float(dt)

    @property
    def provenance(self) -> dict:
        return dict(self.manifest.get("provenance", {}))

    @property
    def run(self) -> dict:
        """The manifest's run block (n_steps, dt_s, wall_seconds,
        mcells_per_s, aborted, abort_reason)."""
        return dict(self._run)

    @property
    def aborted(self) -> bool:
        """True when the solver aborted this run (NUMERICS.md section 7)."""
        return bool(self._run.get("aborted", False))

    @property
    def abort_reason(self) -> Optional[str]:
        """The section-7 reason string (``divergence`` /
        ``non_finite_energy``), or None for a healthy run."""
        reason = self._run.get("abort_reason")
        return str(reason) if reason else None

    @property
    def shut_off(self) -> bool:
        """True when the run ended early via NUMERICS.md section 7 auto-shutoff
        (field energy decayed below ``run.shutoff`` of peak) — a clean finish,
        not an abort."""
        return bool(self._run.get("shut_off", False))

    @property
    def steps_run(self) -> Optional[int]:
        """Steps actually taken (``<= run.n_steps``; fewer when auto-shutoff or
        an abort ended the run early). None for older manifests without the
        key."""
        n = self._run.get("steps_run")
        return int(n) if n is not None else None

    @property
    def monitor_names(self) -> List[str]:
        return list(self._entries)

    def keys(self) -> List[str]:
        return self.monitor_names

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, name: str) -> xr.DataArray:
        if name not in self._cache:
            if name not in self._entries:
                raise KeyError(
                    f"unknown monitor {name!r}; available: {self.monitor_names}"
                )
            self._cache[name] = self._load(self._entries[name])
        return self._cache[name]

    def __repr__(self) -> str:
        return f"SimulationData({str(self.output_dir)!r}, monitors={self.monitor_names})"

    # -- visualization (photonhub.viz; docs/viz-layer-design.md) ------------

    def plot_field(self, monitor, field="Ex", x=None, y=None, z=None, *,
                   freq=None, val="real", structures=True, simulation=None,
                   ax=None, cmap=None, **kw):
        """Heatmap of a field component on a 2D slice of ``self[monitor]``.

        Thin delegation: the rendering lives in :func:`photonhub.viz.plot_field`
        (imported lazily so matplotlib loads only when a plot is requested).
        ``field`` is Ex..Hz or a derived 'E'/'intensity'/'H'; ``freq=`` is
        required for a multi-frequency DFT monitor; ``val`` selects
        real/imag/abs/phase for complex data. Returns a matplotlib ``Axes``."""
        from .viz import plot_field as _plot_field
        return _plot_field(self, monitor, field=field, x=x, y=y, z=z, freq=freq,
                           val=val, structures=structures, simulation=simulation,
                           ax=ax, cmap=cmap, **kw)

    # -- internals ----------------------------------------------------------

    def _grid_dl_um(self) -> float:
        grid = self.manifest.get("grid", {})
        if "dl_um" in grid:
            return float(grid["dl_um"])
        raise ValueError(f"manifest grid block has no 'dl_um': {grid}")

    def _axis_coord_um(self, axis: str, origin: int, n: int,
                       stride: int = 1) -> np.ndarray:
        """Yee-node base coordinates (microns) for `n` recorded cells starting at
        cell `origin` along `axis`, sampled every `stride` cells (NUMERICS.md
        section 12 interval_space; stride 1 = every cell). Graded runs (section
        15.10) carry the per-axis 'coords_um' arrays in the manifest; uniform
        runs fall back to the i * dl_um rule."""
        coords = self.manifest.get("grid", {}).get("coords_um")
        if coords is not None and axis in coords:
            q = np.asarray(coords[axis], dtype=np.float64)
            return q[origin:origin + n * stride:stride]
        return (float(origin) + np.arange(n, dtype=np.float64) * stride) \
            * self._grid_dl_um()

    def _load(self, entry: dict) -> xr.DataArray:
        name = entry["name"]
        dtype = entry.get("dtype", "float32")
        if dtype != "float32":
            raise ValueError(f"monitor {name!r}: unsupported dtype {dtype!r}")
        # Exactly the keys output.cpp emits — no aliases for never-shipped
        # draft manifests.
        kind = entry.get("type")
        if kind in ("field_time", "field_snapshot"):
            return self._load_time_domain(entry)
        if kind == "field_dft":
            return self._load_field_dft(entry)
        if kind == "flux":
            return self._load_flux(entry)
        raise ValueError(f"monitor {name!r}: unknown type {kind!r}")

    def _read_raw(self, entry: dict, expected: int) -> np.ndarray:
        """The monitor's raw float32 array (from the .bin, or the HDF5
        ``/monitors/<name>`` dataset), flat and length-checked against the
        manifest shape. Both backends store the identical little-endian bytes,
        so reconstruction downstream is bit-identical."""
        name = entry["name"]
        if self._h5_path is not None:
            import h5py
            with h5py.File(self._h5_path, "r") as f:
                raw = np.asarray(f["monitors"][name][...], dtype="<f4").ravel()
            source = f"{self._h5_path} [/monitors/{name}]"
        else:
            bin_path = self.output_dir / entry["file"]
            raw = np.fromfile(bin_path, dtype="<f4")
            source = str(bin_path)
        if raw.size != expected:
            raise ValueError(
                f"monitor {name!r}: {source} holds {raw.size} "
                f"float32 values, manifest shape "
                f"{tuple(int(n) for n in entry['shape'])} needs {expected}"
            )
        return raw

    def _freqs_hz(self, entry: dict) -> np.ndarray:
        name = entry["name"]
        freqs = entry.get("freqs_hz")
        if not freqs:
            raise ValueError(
                f"monitor {name!r}: frequency-domain manifest entry has no "
                "'freqs_hz'"
            )
        return np.asarray([float(f) for f in freqs], dtype=np.float64)

    def _common_attrs(self, name: str, kind: str) -> dict:
        return {
            "monitor": name,
            "kind": kind,
            "dt_s": self.dt_s,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason or "",
            "provenance": self.provenance,
        }

    def _load_time_domain(self, entry: dict) -> xr.DataArray:
        name = entry["name"]
        shape = tuple(int(n) for n in entry["shape"])
        components = list(entry["components"])
        sample_steps = [int(s) for s in entry["sample_steps"]]

        kind = entry.get("type")
        if kind == "field_time":
            kind, dims = "time_series", _TIME_DIMS
        else:
            kind, dims = "snapshot", _SNAPSHOT_DIMS
        manifest_dims = entry.get("dims")
        if manifest_dims is not None:
            # The engine names the leading dim "sample" (NUMERICS.md section
            # 6); the xarray dim is "t" with coordinates in seconds.
            normalized = ("t",) + tuple(manifest_dims[1:])
            if tuple(manifest_dims[:1]) != ("sample",) or normalized != dims:
                raise ValueError(
                    f"monitor {name!r}: manifest dims {manifest_dims} != expected {list(dims)}"
                )
        if len(shape) != len(dims):
            raise ValueError(
                f"monitor {name!r}: shape {shape} has {len(shape)} dims, expected {len(dims)}"
            )
        if shape[0] != len(sample_steps) or shape[1] != len(components):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(sample_steps)} samples x {len(components)} components"
            )

        data = self._read_raw(entry, int(np.prod(shape))).reshape(shape)

        dt = self.dt_s
        coords = {
            "t": ("t", np.asarray(sample_steps, dtype=np.float64) * dt,
                  {"units": "s", "long_name": "E-field sample time (step * dt)"}),
            "component": list(components),
        }
        if kind == "snapshot":
            for axis, n in zip(("z", "y", "x"), shape[2:]):
                coords[axis] = (axis, self._axis_coord_um(axis, 0, n),
                                {"units": "um"})

        attrs = self._common_attrs(name, kind)
        attrs["sample_steps"] = sample_steps
        return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs, name=name)

    def _load_field_dft(self, entry: dict) -> xr.DataArray:
        """NUMERICS.md section 12 field_dft: float32 [re, im] pairs in binary
        order [freq][component][k][j][i][re,im], reconstructed as complex64
        with dims ('f', 'component', 'z', 'y', 'x'). Spatial coordinates are
        Yee cell base coordinates (index * dl_um, offset by the optional
        'origin_cells' region corner). The section-12 snapping rule makes
        origin_cells EXACT for every listed component (the validator rejects
        specs whose per-component snaps disagree); per-component Yee
        half-cell offsets (NUMERICS.md section 1.1) are still not applied,
        as for snapshots."""
        name = entry["name"]
        shape = tuple(int(n) for n in entry["shape"])
        components = list(entry["components"])
        freqs = self._freqs_hz(entry)

        if len(shape) != 6 or shape[-1] != 2:
            raise ValueError(
                f"monitor {name!r}: field_dft shape {shape} must be "
                "[freq, component, z, y, x, 2]"
            )
        manifest_dims = entry.get("dims")
        if manifest_dims is not None:
            # Exactly the dim names output.cpp emits — no aliases. The
            # engine names the leading dim "freq"; the xarray dim is "f"
            # with coordinates in Hz. The trailing [re, im] pair dim (named
            # "complex" by the engine's output.cpp) is consumed by the
            # complex64 reconstruction.
            normalized = ("f",) + tuple(manifest_dims[1:-1])
            if (tuple(manifest_dims[:1]) != ("freq",)
                    or tuple(manifest_dims[-1:]) != ("complex",)
                    or normalized != _DFT_DIMS):
                raise ValueError(
                    f"monitor {name!r}: manifest dims {manifest_dims} != "
                    "expected ['freq', 'component', 'z', 'y', 'x', 'complex']"
                )
        if shape[0] != len(freqs) or shape[1] != len(components):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(freqs)} freqs x {len(components)} components"
            )

        raw = self._read_raw(entry, int(np.prod(shape)))
        # Adjacent [re, im] float32 pairs are exactly numpy's complex64
        # memory layout — a view, not a lossy round trip through complex128.
        data = raw.view("<c8").reshape(shape[:-1])

        origin = entry.get("origin_cells", (0, 0, 0))
        if len(origin) != 3:
            raise ValueError(
                f"monitor {name!r}: origin_cells {origin} must be the region "
                "low corner as [i0, j0, k0] (x, y, z cell indices)"
            )
        # §12 interval_space: per-axis spatial stride (x, y, z). Absent => every
        # cell (stride 1). Strides the reconstructed coordinates so a decimated
        # plane reads back at the right physical positions.
        stride = entry.get("interval_space", (1, 1, 1))
        if len(stride) != 3:
            raise ValueError(
                f"monitor {name!r}: interval_space {stride} must be "
                "[sx, sy, sz] (x, y, z strides)"
            )
        coords = {
            "f": ("f", freqs, {"units": "Hz"}),
            "component": components,
        }
        # origin_cells / interval_space are (x, y, z); the spatial block is
        # (z, y, x), so reverse both alongside it.
        for axis, n, o, s in zip(("z", "y", "x"), shape[2:5],
                                 reversed(list(origin)), reversed(list(stride))):
            coords[axis] = (axis, self._axis_coord_um(axis, int(o), n, int(s)),
                            {"units": "um"})

        attrs = self._common_attrs(name, "field_dft")
        attrs["freqs_hz"] = [float(f) for f in freqs]
        attrs["normalization"] = _DFT_NORMALIZATION
        return xr.DataArray(data, dims=_DFT_DIMS, coords=coords, attrs=attrs,
                            name=name)

    def _load_flux(self, entry: dict) -> xr.DataArray:
        """NUMERICS.md section 12 flux: one float32 power per frequency
        (fp64-accumulated in the engine), dims ('f',), positive toward
        +axis, carrying the shared 1/|A0*S(f)|^2 normalization."""
        name = entry["name"]
        shape = tuple(int(n) for n in entry["shape"])
        freqs = self._freqs_hz(entry)

        if len(shape) != 1:
            raise ValueError(
                f"monitor {name!r}: flux shape {shape} must be [freq]"
            )
        manifest_dims = entry.get("dims")
        if manifest_dims is not None and tuple(manifest_dims) != ("freq",):
            # Exactly the dim name output.cpp emits — no aliases.
            raise ValueError(
                f"monitor {name!r}: manifest dims {manifest_dims} != "
                "expected ['freq']"
            )
        if shape[0] != len(freqs):
            raise ValueError(
                f"monitor {name!r}: shape {shape} inconsistent with "
                f"{len(freqs)} freqs"
            )

        data = self._read_raw(entry, shape[0])

        attrs = self._common_attrs(name, "flux")
        attrs["freqs_hz"] = [float(f) for f in freqs]
        attrs["normalization"] = _FLUX_NORMALIZATION
        # output.cpp emits "axis" unconditionally for flux entries (it throws
        # on a spec lookup miss): without it the sign convention of the
        # reported power is unrecoverable from the artifact.
        if "axis" not in entry:
            raise ValueError(
                f"monitor {name!r}: flux manifest entry has no 'axis' (the "
                "plane normal; required — output.h manifest contract)"
            )
        attrs["axis"] = entry["axis"]
        coords = {"f": ("f", freqs, {"units": "Hz"})}
        return xr.DataArray(data, dims=_FLUX_DIMS, coords=coords, attrs=attrs,
                            name=name)
