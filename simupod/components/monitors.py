"""Field monitors (NUMERICS.md sections 6 and 12).

Monitor names must be filename-safe (the engine writes ``<name>.bin``) and —
a constraint JSON Schema cannot express across array items — unique within a
simulation (enforced by ``Simulation``).
"""

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator

from .base import (
    MAX_INT32,
    AxisName,
    FieldComponentName,
    FreqHz,
    FrozenModel,
    MonitorName,
    NonNegativeUm,
    Vec3Um,
)


class FieldTimeMonitor(FrozenModel):
    """Scalar time-series probe at the Yee node nearest ``center_um``. Samples
    are raw, non-colocated Yee values; H lags E by dt/2."""

    type: Literal["field_time"] = "field_time"
    name: MonitorName
    center_um: Vec3Um
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    interval_steps: int = Field(default=1, ge=1, le=MAX_INT32)


class FieldSnapshotMonitor(FrozenModel):
    """Full-domain dump of selected components. ``interval_steps = 0`` (the
    default) records only the final step."""

    type: Literal["field_snapshot"] = "field_snapshot"
    name: MonitorName
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    interval_steps: int = Field(default=0, ge=0, le=MAX_INT32)


class FieldDftMonitor(FrozenModel):
    """Running-DFT field monitor over a box region (NUMERICS.md section 12):
    fp64 accumulation every step over the full run, raw Yee-located phasors,
    normalized by the first wire-order source's ``A0 * S(f)``. ``size_um``
    components may be 0 (plane/line/point regions); the region is snapped per
    component to that component's Yee sublattice, and the engine validator
    REJECTS boxes whose per-component snaps disagree (the output carries one
    shape/origin per monitor). When ``fields`` mixes Yee offsets along an
    axis, place that axis' box faces strictly between an integer cell
    boundary and the next half-cell plane — canonically ``(k + 0.25) * dl``,
    which every component snaps to cell ``k`` with quarter-cell fp margin."""

    type: Literal["field_dft"] = "field_dft"
    name: MonitorName
    center_um: Vec3Um
    size_um: Tuple[NonNegativeUm, NonNegativeUm, NonNegativeUm]
    fields: Tuple[FieldComponentName, ...] = Field(min_length=1)
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)
    # Per-axis spatial sampling stride (schema 1.11.0, additive/optional — the
    # Tidy3D interval_space). None (default) records every cell; (sx, sy, sz)
    # decimates the recorded region along each axis (output cell i -> snapped
    # cell + i*stride), cutting field-monitor output for large planes/volumes.
    # Each stride >= 1. Omitted from the wire when unset (older engines/readers
    # round-trip unchanged); the data layer strides the coordinates to match.
    interval_space: Optional[Tuple[int, int, int]] = None

    @field_validator("interval_space")
    @classmethod
    def _strides_positive(cls, v):
        if v is not None and any(s < 1 for s in v):
            raise ValueError(
                f"interval_space strides must be >= 1 (1 = every cell), got {v}"
            )
        return v


class FluxMonitor(FrozenModel):
    """Poynting-flux monitor over the full plane perpendicular to ``axis`` at
    ``position_um``, snapped to a plane index ``1 <= kp <= n_axis - 1``
    (NUMERICS.md section 12). Positive values mean power toward +axis; the
    reported power carries the ``1/|A0*S(f)|^2`` normalization of the shared
    phasors, so it is not absolute watts."""

    type: Literal["flux"] = "flux"
    name: MonitorName
    axis: AxisName
    position_um: float
    freqs_hz: Tuple[FreqHz, ...] = Field(min_length=1)


MonitorType = Annotated[
    Union[FieldTimeMonitor, FieldSnapshotMonitor, FieldDftMonitor, FluxMonitor],
    Field(discriminator="type"),
]
