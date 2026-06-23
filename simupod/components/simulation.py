"""Top-level simulation model — the root of the wire format."""

import warnings
from typing import Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from ..capabilities import unavailable
from ..cost import CostEstimate, estimate_cost
from .base import MAX_INT32, FrozenModel, PositiveUm, SubpixelMethodName
from .grid import (
    GridSpecType,
    graded_primary_spacings,
    realized_cells,
    snapped_plane_index,
)
from .medium import Background, Boundaries
from .monitors import FluxMonitor, MonitorType
from .run import RunSpec
from .sources import PlaneWave, SourceType
from .structures import Structure

SCHEMA_VERSION = "1.12.0-alpha.1"
SUPPORTED_SCHEMA_MAJOR = 1

_AXES = "xyz"


class Simulation(FrozenModel):
    """Complete simulation description. Serializes 1:1 to the JSON wire
    format consumed by ``phsolver`` (schemas/GOVERNANCE.md).

    The cross-field validators here are best-effort early feedback mirroring
    the engine's checks where they are cheap and unambiguous; ``phsolver
    validate`` remains authoritative (notably for Yee snapping at exact
    boundaries and for the plane-wave/PML intersection rule)."""

    schema_version: str = SCHEMA_VERSION
    size_um: Tuple[PositiveUm, PositiveUm, PositiveUm]
    grid: GridSpecType
    run: RunSpec
    background: Background = Background()
    # NUMERICS.md section 11: layer count for every "pml" boundary axis. The
    # engine default is 12; an UNSET value is omitted from the wire format
    # (see to_wire_dict) so Phase-0 documents round-trip byte-identically and
    # remain consumable by schema-1.0 parsers that reject unknown keys.
    pml_num_layers: int = Field(default=12, ge=4, le=MAX_INT32)
    # NUMERICS.md §11 CPML profile (Roden–Gedney) tuning knobs. The defaults
    # reproduce the historically-hardcoded profile BIT-FOR-BIT, so an UNSET
    # value is omitted from the wire format (see _wire_exclude) and the engine
    # applies the same constants — documents from earlier minors round-trip
    # byte-identically and stay consumable by parsers that reject unknown keys.
    #   pml_m         polynomial grading order (>= 1)
    #   pml_kappa_max real-stretch peak at the wall (>= 1)
    #   pml_alpha_max CFS frequency-shift peak in S/m (>= 0)
    # The default alpha_max (0.24 S/m) is a deliberately SMALL nudge: at optical
    # frequencies alpha << omega*eps0, so it costs no in-band reflectionlessness
    # yet guards the DC/late-time tail. Raising layers + kappa_max + alpha_max
    # together is the "stabilized" recipe for a grazing/long-run/dispersive scene
    # that diverges — ``with_stabilized_pml`` builds such a copy.
    pml_m: float = Field(default=3.0, ge=1.0)
    pml_kappa_max: float = Field(default=3.0, ge=1.0)
    pml_alpha_max: float = Field(default=0.24, ge=0.0)
    # NUMERICS.md §21 adiabatic-absorber knobs (apply to every "absorber" axis).
    # The absorber is a graded electric-conductivity ramp, NOT a stretched-
    # coordinate PML — the robustness fallback for the cases that make a PML
    # diverge (a structure crossing the boundary, dispersive/gain media at the
    # edge). It needs more layers than the PML for comparable reflection (40 vs
    # 12) because, being impedance-unmatched, its reflection falls only
    # polynomially with thickness. Additive-optional: an UNSET value is omitted
    # from the wire (see _wire_exclude) so earlier-minor parsers accept the
    # document and golden specs round-trip byte-identically.
    #   absorber_num_layers  slab thickness in cells, both faces (>= 4)
    #   absorber_m           polynomial conductivity grading order (>= 1)
    absorber_num_layers: int = Field(default=40, ge=4, le=MAX_INT32)
    absorber_m: float = Field(default=3.0, ge=1.0)
    # NUMERICS.md §16: volume-fraction subpixel smoothing of the rasterized
    # permittivity. False (default) = the §9 last-wins point sample, bit-exact
    # with prior schema minors; an UNSET value is omitted from the wire format
    # (see _wire_exclude) so documents stay byte-identical and consumable by
    # parsers from earlier minors that reject unknown keys. Box interfaces are
    # smoothed on a uniform grid; spheres and graded axes keep the hard sample
    # in v1 (engine reference_solver.cpp; the §16 deferral).
    subpixel: bool = False
    # NUMERICS.md §16.5/§16.8: which smoothing to apply when ``subpixel`` is True.
    # "volume" (default) = isotropic volume average, bit-identical to schema
    # < 1.7.0; "tensor" = diagonal anisotropic KFJ on box interfaces. Omitted from
    # the wire when unset (see _wire_exclude) so earlier-minor parsers still
    # accept the document.
    subpixel_method: SubpixelMethodName = "volume"
    structures: Tuple[Structure, ...] = ()
    boundaries: Boundaries = Boundaries()
    # NUMERICS.md §20: optional symmetry plane on each axis' MINIMUM face.
    # 0 = none; -1 = odd / electric (a PEC mirror: tangential E pinned, normal E
    # free — the common case for a TE-like mode); +1 = even / magnetic (PMC
    # mirror: tangential E free, the cross-plane H read is the odd-H image).
    # PMC is available on x and y; z-axis PMC (+1 on z) is deferred and rejected.
    # When symmetry[a] != 0 the axis is non-periodic and boundaries[a] governs
    # the FAR (max) face only (pml or pec); the PML on that axis is built
    # one-sided so the min/symmetry face reflects. You supply the reduced (half)
    # domain with the structure's mirror plane on that face. Additive-optional:
    # an all-zero symmetry is omitted from the wire (see _wire_exclude), so
    # earlier-minor parsers and golden specs round-trip byte-identically.
    symmetry: Tuple[int, int, int] = (0, 0, 0)
    sources: Tuple[SourceType, ...] = Field(min_length=1)
    monitors: Tuple[MonitorType, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def _supported_major_version(cls, v: str) -> str:
        # Mirrors the engine's schema_major() gate (engine/src/core/
        # resolve.cpp): leading dotted component, digits only, must be the
        # supported major — GOVERNANCE.md rule 2 hangs migrations on it, and
        # an unsupported spec must fail at construction, not at submission.
        head = v.split(".", 1)[0]
        if not (head.isascii() and head.isdigit()) or int(
                head) != SUPPORTED_SCHEMA_MAJOR:
            raise ValueError(
                f"unsupported schema_version {v!r}: this client supports "
                f"major version {SUPPORTED_SCHEMA_MAJOR} only (other majors "
                "require a migration; see simupod.migrations)"
            )
        return v

    @field_validator("monitors")
    @classmethod
    def _unique_monitor_names(cls, v):
        names = [m.name for m in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"monitor names must be unique; duplicates: {dupes}")
        return v

    @field_validator("symmetry")
    @classmethod
    def _symmetry_values(cls, v):
        # NUMERICS.md §20: each axis is -1 (odd/electric), 0 (none), or +1
        # (even/magnetic). +1 is range-valid here but rejected as unsupported in
        # _reject_unavailable_features (a whole-feature verdict, like the engine).
        for a, s in enumerate(v):
            if s not in (-1, 0, 1):
                raise ValueError(
                    f"symmetry[{a}] ('{_AXES[a]}'): must be -1 (electric/PEC), "
                    f"0 (none), or +1 (magnetic/PMC), got {s}"
                )
        return v

    def _axis_coords_um(self, axis_index: int):
        """The graded coordinate array (microns) for an axis, or None when
        that axis is uniform (UniformGridSpec, or a GradedGridSpec axis not
        listed in ``coords``)."""
        coords = getattr(self.grid, "coords", None)
        if coords is None:
            return None
        return getattr(coords, "xyz"[axis_index])

    def _realized_um(self) -> Tuple[float, float, float]:
        dl = self.grid.dl_um
        out = []
        for i, L in enumerate(self.size_um):
            q = self._axis_coords_um(i)
            if q is None:
                out.append(realized_cells(L, dl) * dl)
            else:
                # NUMERICS.md section 15.1: realized length = closing node
                # q[n-1] + (replicate-last spacing).
                out.append(q[-1] + graded_primary_spacings(q)[-1])
        return tuple(out)

    def with_auto_grid(
        self,
        *,
        wavelength_um: Optional[float] = None,
        steps_per_wvl: float = 20.0,
        **auto_grid_kwargs,
    ) -> "Simulation":
        """Return a COPY of this simulation whose ``grid`` is replaced by an
        auto-meshed :class:`GradedGridSpec` derived from this scene — its
        domain ``size_um``, ``structures``, ``background`` index, and (if
        ``wavelength_um`` is omitted) the wavelength inferred from the first
        source. A convenience wrapper over :func:`simupod.auto_grid`; extra
        keyword arguments (``max_grading``, ``axes``, ``dl_min_um``,
        ``refine_regions``, ...) pass straight through.

        Opt-in only: the default :class:`UniformGridSpec` is unchanged, so no
        existing scene's wire output moves. Use this when you want Tidy3D-style
        per-medium refinement without hand-building coordinate arrays::

            sim = sim.with_auto_grid(steps_per_wvl=20)
        """
        from .grid import auto_grid as _auto_grid  # local: avoid import cycle

        bg_index = float(self.background.permittivity) ** 0.5
        src = self.sources[0] if self.sources else None
        spec = _auto_grid(
            size_um=tuple(self.size_um),
            wavelength_um=wavelength_um,
            source=None if wavelength_um is not None else src,
            structures=self.structures,
            background_index=bg_index,
            steps_per_wvl=steps_per_wvl,
            **auto_grid_kwargs,
        )
        return self.model_copy(update={"grid": spec})

    def with_stabilized_pml(
        self,
        *,
        num_layers: int = 20,
        kappa_max: float = 5.0,
        alpha_frac: float = 0.02,
    ) -> "Simulation":
        """Return a COPY with a more numerically STABLE CPML profile: more
        layers, a higher real-stretch peak ``kappa_max``, and — the lever the
        default profile deliberately keeps small — a RAISED CFS ``alpha_max``
        (NUMERICS.md §11). The complex frequency shift is what moves the PML
        pole off DC and cures the late-time / grazing-incidence /
        dispersive-medium divergences that more layers alone do not fix; it
        costs a few percent of in-band absorption, paid back by the extra
        layers. Reach for it when the default 12-layer profile leaks or drifts.

        ``alpha_max`` is set GRID-AWARE as ``alpha_frac * sigma_max`` with the
        §11 optimal ``sigma_max = 0.8*(m+1)/(eta0*dl)``, so the per-step CFS
        damping is independent of the mesh (a fixed absolute alpha would weaken
        on finer grids). ``alpha_frac`` ~ 0.02 follows the Gedney rule (a few %
        of sigma). Opt-in only — the default profile is unchanged, so no
        existing scene's wire output moves::

            sim = sim.with_stabilized_pml()              # 20 layers, kappa 5, alpha on
            sim = sim.with_stabilized_pml(num_layers=32, alpha_frac=0.04)

        (Renamed from the earlier ``with_stable_pml``, which raised only layers
        and kappa — never alpha, so it missed the actual stability lever.)
        """
        # §11 optimal conductivity at this grid's base spacing (eta0 = mu0*c0).
        # dl_um is the base spacing for both uniform and graded grids; on a
        # graded mesh the absolute alpha_max is referenced to it.
        eta0 = 1.25663706212e-6 * 2.99792458e8
        dl_m = self.grid.dl_um * 1e-6
        sigma_max = 0.8 * (self.pml_m + 1.0) / (eta0 * dl_m)
        return self.model_copy(update={
            "pml_num_layers": num_layers,
            "pml_kappa_max": kappa_max,
            "pml_alpha_max": alpha_frac * sigma_max,
        })

    def with_absorber(self, *, num_layers: int = 40) -> "Simulation":
        """Return a COPY with every face set to the adiabatic absorber
        (NUMERICS.md §21) instead of a PML. Use this when a structure crosses
        the domain boundary or a dispersive/gain medium touches the edge — the
        cases where a stretched-coordinate PML can diverge. The absorber trades
        some reflection (≈ -28 dB at the default 40 layers, vs the PML's -68 dB)
        for robustness; add layers if you need it tighter::

            sim = sim.with_absorber()                 # 40-layer absorber, 6 faces
            sim = sim.with_absorber(num_layers=60)
        """
        return self.model_copy(update={
            "boundaries": Boundaries(x="absorber", y="absorber", z="absorber"),
            "absorber_num_layers": num_layers,
        })

    @model_validator(mode="after")
    def _reject_unavailable_features(self) -> "Simulation":
        # Capability gating (simupod.capabilities): a feature the client can
        # express but the v1 engine cannot run must fail HERE, at construction,
        # with an "available in <version>" message — never at phsolver
        # submission. Defined first among the model validators so this
        # whole-feature verdict wins over positional nitpicks (a center bound,
        # a flux-plane index) on a scene that can never run anyway.
        #
        # Graded grid + plane wave: §15.9 restricts plane-wave injection to a
        # uniform axis, and the engine currently rejects ANY graded+plane-wave
        # combination (engine reference_solver.cpp). Mirror that exactly — even
        # a transverse-only graded axis is rejected, matching the engine rather
        # than §15.9's finer (not-yet-implemented) rule, so the client never
        # accepts a scene the solver will refuse.
        is_graded = any(self._axis_coords_um(a) is not None for a in range(3))
        if is_graded and any(isinstance(s, PlaneWave) for s in self.sources):
            raise unavailable("graded_plane_wave")
        # NUMERICS.md §20.4: even / magnetic (PMC) is supported on x and y; the
        # z axis reads a stored ghost plane (not the branchless bwd path), so
        # z-axis PMC is deferred — the engine rejects symmetry.z == +1.
        if self.symmetry[2] == 1:
            raise unavailable("magnetic_symmetry_z")
        return self

    @model_validator(mode="after")
    def _symmetry_plane_rules(self) -> "Simulation":
        # NUMERICS.md §20.2, mirrored at construction (cheap, unambiguous): a
        # symmetry axis must be non-periodic (boundaries governs the far face),
        # and a symmetry plane is incompatible with a plane-wave source (its
        # TF/SF aux line spans the full transverse plane).
        if not any(s != 0 for s in self.symmetry):
            return self
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        for a, s in enumerate(self.symmetry):
            if s != 0 and kinds[a] == "periodic":
                raise ValueError(
                    f"symmetry[{a}] ('{_AXES[a]}'): a symmetry axis cannot be "
                    f"periodic; set boundaries.{_AXES[a]} to 'pml', 'absorber', "
                    "or 'pec' for the far face (NUMERICS.md §20.2)"
                )
        if any(isinstance(s, PlaneWave) for s in self.sources):
            raise ValueError(
                "a symmetry plane cannot be combined with a plane-wave source "
                "(NUMERICS.md §20.2)"
            )
        return self

    @model_validator(mode="after")
    def _centers_inside_realized_domain(self) -> "Simulation":
        # Best-effort early feedback mirroring the engine's domain check
        # (engine/src/core/resolve.cpp): centers must lie inside the REALIZED
        # domain n_axis * dl (NUMERICS.md section 1 — the n >= 4 floor and
        # half-away rounding can make it differ from size_um). The engine
        # computes in meters, so `phsolver validate` remains authoritative at
        # exact boundaries. Structures are exempt: geometry may extend beyond
        # the domain (NUMERICS.md section 9).
        realized = self._realized_um()
        domain = (f"[0, {realized[0]:.9g}] x [0, {realized[1]:.9g}] x "
                  f"[0, {realized[2]:.9g}] um (realized)")

        def check(center, label: str) -> None:
            for c, r in zip(center, realized):
                if not (0.0 <= c <= r):
                    raise ValueError(
                        f"{label}.center_um {tuple(center)} is outside the "
                        f"domain {domain}"
                    )

        for i, s in enumerate(self.sources):
            center = getattr(s, "center_um", None)  # plane waves have none
            if center is not None:
                check(center, f"sources[{i}]")
            elif isinstance(s, PlaneWave):
                axis = _AXES.index(s.axis)
                if not (0.0 <= s.position_um <= realized[axis]):
                    raise ValueError(
                        f"sources[{i}].position_um {s.position_um} ({s.axis} "
                        f"axis) is outside the domain {domain}"
                    )
        for m in self.monitors:
            center = getattr(m, "center_um", None)  # snapshots/flux have none
            if center is not None:
                check(center, f"monitor '{m.name}'")
        return self

    @model_validator(mode="after")
    def _plane_wave_transverse_axes_periodic(self) -> "Simulation":
        # NUMERICS.md section 13 validator, mirrored exactly (no float math,
        # safe to enforce strictly): a normal-incidence plane wave requires
        # both transverse axes periodic.
        kinds = (self.boundaries.x, self.boundaries.y, self.boundaries.z)
        for i, s in enumerate(self.sources):
            if not isinstance(s, PlaneWave):
                continue
            for t, kind in enumerate(kinds):
                if _AXES[t] != s.axis and kind != "periodic":
                    raise ValueError(
                        f"sources[{i}] (plane_wave along {s.axis}): transverse "
                        f"axis '{_AXES[t]}' must be periodic, got '{kind}' "
                        "(NUMERICS.md section 13)"
                    )
        return self

    @model_validator(mode="after")
    def _resolve_subpixel_default(self, info) -> "Simulation":
        # D2 (NUMERICS.md §16): default-ON subpixel smoothing for the common
        # case. When ``subpixel`` is NOT set explicitly, enable the diagonal-KFJ
        # ``tensor`` average (matching Tidy3D's subpixel-on posture, the more
        # accurate out-of-box choice) for a NON-dispersive scene, and fall back
        # to OFF for a dispersive one (the subpixel × Lorentz-ADE late-time
        # instability — engine/docs/subpixel-dispersion-instability.md). The
        # resolved value is marked "set" so it serialises on the wire (the engine
        # field default is off), keeping CPU and GPU runs consistent with this
        # choice. An explicit ``subpixel`` is always respected verbatim; an
        # explicit subpixel-ON dispersive scene only gets a warning, never an
        # override.
        #
        # This is a CONSTRUCTION-time convenience only. When INGESTING an existing
        # wire document (``from_wire_json`` passes context ``wire_ingest``), the
        # absence of ``subpixel`` means the engine default (off) — flipping it
        # would break byte-identical round-trip of older docs — so the resolution
        # is skipped and the field keeps whatever the document stated (or its
        # unset default).
        if (info.context or {}).get("wire_ingest"):
            return self
        dispersive = any(
            getattr(s.medium, "lorentz", None) is not None
            for s in self.structures
        )
        if "subpixel" not in self.model_fields_set:
            if not dispersive:
                # Enable + mark set so it serialises (engine field default = off).
                object.__setattr__(self, "subpixel", True)
                self.__pydantic_fields_set__.add("subpixel")
                if "subpixel_method" not in self.model_fields_set:
                    object.__setattr__(self, "subpixel_method", "tensor")
                    self.__pydantic_fields_set__.add("subpixel_method")
            # Dispersive: leave ``subpixel`` at its unset default (off, omitted
            # from the wire = the engine default) — the auto-fallback.
        elif self.subpixel and dispersive:
            warnings.warn(
                "subpixel smoothing is enabled on a dispersive (Lorentz) scene; "
                "the subpixel × ADE update can diverge at fine grids "
                "(engine/docs/subpixel-dispersion-instability.md). Prefer "
                "subpixel=False for dispersive runs.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _flux_planes_inside_domain(self) -> "Simulation":
        # Best-effort mirror of the engine's flux-plane bound (NUMERICS.md
        # section 12: snapped plane index 1 <= kp <= n_axis - 1); phsolver
        # remains authoritative at exact half-cell positions.
        dl = self.grid.dl_um
        for m in self.monitors:
            if not isinstance(m, FluxMonitor):
                continue
            axis = _AXES.index(m.axis)
            # Graded axis: the coordinate-based plane snap (NUMERICS.md
            # section 15.6) is the engine's; skip the uniform-dl best-effort
            # check here (phsolver validate remains authoritative).
            if self._axis_coords_um(axis) is not None:
                continue
            n = realized_cells(self.size_um[axis], dl)
            kp = snapped_plane_index(m.position_um, dl)
            if not (1 <= kp <= n - 1):
                raise ValueError(
                    f"monitor '{m.name}': flux plane at position_um "
                    f"{m.position_um} snaps to {m.axis}-plane index {kp}, "
                    f"outside the interior range [1, {n - 1}] "
                    "(NUMERICS.md section 12)"
                )
        return self

    @classmethod
    def from_wire_json(cls, text: Union[str, bytes]) -> "Simulation":
        """Strictly-typed ingestion of wire JSON, matching the engine's
        nlohmann typing exactly: JSON int -> float fields is accepted,
        string -> number and float -> int are rejected. Use this (not lax
        ``model_validate_json``) when consuming sim.json files."""
        # context wire_ingest: do NOT apply the D2 construction-time subpixel
        # default to a parsed document — absent means the engine default (off),
        # so older docs round-trip byte-identically (see _resolve_subpixel_default).
        return cls.model_validate_json(text, strict=True,
                                       context={"wire_ingest": True})

    def _wire_exclude(self):
        # Omit additive-optional fields that were never explicitly set so
        # older documents round-trip byte-identically and stay consumable by
        # earlier-minor parsers that reject unknown keys; the engine applies
        # the same defaults. pml_num_layers entered the wire in schema 1.1.0
        # (default 12); run.shutoff in 1.3.0 (default 1e-5, NUMERICS.md §7).
        exclude: dict = {}
        if "pml_num_layers" not in self.model_fields_set:
            exclude["pml_num_layers"] = True
        # The §11 CPML profile knobs entered the wire in schema 1.8.0 (defaults
        # m=3 / kappa_max=3 / alpha_max=0.24, bit-identical to the prior
        # hardcoded profile); omit each when unset so earlier-minor parsers
        # accept the document and golden specs round-trip byte-identically.
        for _f in ("pml_m", "pml_kappa_max", "pml_alpha_max"):
            if _f not in self.model_fields_set:
                exclude[_f] = True
        # The §21 absorber knobs entered the wire in schema 1.12.0 (defaults
        # 40 layers / m=3); omit each when unset so earlier-minor parsers accept
        # the document and golden specs round-trip byte-identically.
        for _f in ("absorber_num_layers", "absorber_m"):
            if _f not in self.model_fields_set:
                exclude[_f] = True
        # subpixel entered the wire in schema 1.4.0 (default false, NUMERICS.md
        # §16); omit it when unset so 1.3-and-earlier parsers still accept the
        # document and golden specs round-trip byte-identically.
        if "subpixel" not in self.model_fields_set:
            exclude["subpixel"] = True
        # subpixel_method entered the wire in schema 1.7.0 (default "volume",
        # NUMERICS.md §16.5); omit when unset so earlier-minor parsers accept the
        # document and golden specs round-trip byte-identically.
        if "subpixel_method" not in self.model_fields_set:
            exclude["subpixel_method"] = True
        # symmetry entered the wire in schema 1.11.0 (NUMERICS.md §20); omit when
        # all-zero (the no-symmetry default) so earlier-minor parsers accept the
        # document and golden specs round-trip byte-identically.
        if self.symmetry == (0, 0, 0):
            exclude["symmetry"] = True
        if "shutoff" not in self.run.model_fields_set:
            exclude["run"] = {"shutoff"}
        return exclude or None

    def to_wire_dict(self) -> dict:
        """Canonical JSON-level dict: defaults materialized, unset optionals
        (the unused run_time_s/n_steps key, an unset pml_num_layers) omitted."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True,
                               exclude=self._wire_exclude())

    def to_wire_json(self, indent: int = 2) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True,
                                    exclude=self._wire_exclude(), indent=indent)

    # -- Visualization (simupod.viz; design doc docs/viz-layer-design.md) ---
    # Thin delegations: the rendering logic lives entirely in simupod.viz so
    # these pydantic models stay clean. Imported lazily so matplotlib is only
    # loaded when a plot is actually requested.

    def plot(self, x=None, y=None, z=None, *, ax=None, legend=True, **kw):
        """2D analytic cross-section of the scene on a cut plane (exactly one
        of x/y/z, in microns). Returns a matplotlib ``Axes``. See
        :func:`simupod.viz.plot`."""
        from ..viz import plot as _plot
        return _plot(self, x=x, y=y, z=z, ax=ax, legend=legend, **kw)

    def plot_eps(self, x=None, y=None, z=None, *, ax=None, cmap=None, **kw):
        """Rasterized permittivity heatmap (the §9 hard sample the solver
        takes) on a cut plane. Returns a matplotlib ``Axes``. See
        :func:`simupod.viz.plot_eps`."""
        from ..viz import plot_eps as _plot_eps
        return _plot_eps(self, x=x, y=y, z=z, ax=ax, cmap=cmap, **kw)

    def plot_3d(self, **kw):
        """Interactive 3D geometry as a plotly ``Figure`` (requires the
        ``photonhub[viz]`` extra). See :func:`simupod.viz.plot_3d`."""
        from ..viz import plot_3d as _plot_3d
        return _plot_3d(self, **kw)

    def cost_estimate(
        self,
        *,
        rate_usd_per_tcell_step: float = ...,
        throughput_gcells_per_s: float = ...,
    ) -> "CostEstimate":
        """Pure-Python dollar / memory / output / wall-time estimate (the
        plan's "estimate in dollars before you press run"). See
        :func:`simupod.cost.estimate_cost`. Cell count, dt and step count
        match the engine's resolve.cpp, so the dollar figure tracks what
        ``phsolver`` will run; it is exact for a full-duration run (auto-shutoff
        can only make it cheaper)."""
        # Forward only explicitly-passed overrides so the single source of the
        # default rate/throughput stays in simupod.cost.
        kwargs = {}
        if rate_usd_per_tcell_step is not ...:
            kwargs["rate_usd_per_tcell_step"] = rate_usd_per_tcell_step
        if throughput_gcells_per_s is not ...:
            kwargs["throughput_gcells_per_s"] = throughput_gcells_per_s
        return estimate_cost(self, **kwargs)
