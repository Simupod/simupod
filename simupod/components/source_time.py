"""Source time dependences (NUMERICS.md section 5)."""

import math
import warnings
from typing import Annotated, Literal, Optional, Sequence, Union

from pydantic import Field

from .base import FrozenModel

# Free-space speed of light (m/s), identical to the engine's kC0
# (engine/include/phcore/types.h) and to grid.py / cost.py, so the wavelength
# <-> frequency conversion used by :meth:`GaussianPulse.for_band` round-trips
# bit-comparably with the rest of the client.
_C0_M_PER_S = 2.99792458e8


class GaussianPulse(FrozenModel):
    """J(t) = amplitude * exp(-(t-t0)^2/(2 tau^2)) * cos(2 pi freq0 (t-t0) + phase),
    tau = 1/(2 pi fwidth), t0 = offset * tau.

    The frequency-domain envelope is a Gaussian whose standard deviation in Hz
    is exactly ``fwidth_hz`` (NUMERICS.md §12). Two facts drive source tuning:
    the run cannot auto-shutoff before the source stops injecting at
    ``(offset+8)*tau`` (NUMERICS.md §7), so settling scales as ``1/fwidth``; and
    too broad a pulse puts spectral weight on DC and goes unstable. The
    :meth:`for_band` constructor tunes ``freq0_hz``/``fwidth_hz`` for both, and
    :attr:`source_end_time_s` / :attr:`dc_amplitude` expose them for auditing.
    """

    type: Literal["gaussian_pulse"] = "gaussian_pulse"
    freq0_hz: float = Field(gt=0)
    fwidth_hz: float = Field(gt=0)
    offset: float = Field(default=5.0, ge=0)
    phase: float = 0.0

    # --- settling / spectral diagnostics (mirror engine source_time.h) ---

    @property
    def tau_s(self) -> float:
        """Gaussian width in seconds, ``1/(2 pi fwidth_hz)`` (engine pulse_tau)."""
        return 1.0 / (2.0 * math.pi * self.fwidth_hz)

    @property
    def t0_s(self) -> float:
        """Pulse-centre time, ``offset * tau`` (engine t0)."""
        return self.offset * self.tau_s

    @property
    def source_end_time_s(self) -> float:
        """Instant the source is considered done injecting, ``(offset+8)*tau``
        (engine ``source_end_time``). Auto-shutoff (NUMERICS.md §7) may not
        terminate the run before this time, so it is the settling floor on the
        step count — inversely proportional to ``fwidth_hz``."""
        return (self.offset + 8.0) * self.tau_s

    def spectral_amplitude(self, freq_hz: float) -> float:
        """Relative spectral envelope at ``freq_hz`` (peak 1.0 at ``freq0_hz``):
        ``exp(-(freq-freq0)^2/(2 fwidth^2))`` — the dominant sideband of the
        analytic pulse spectrum (engine ``gaussian_pulse_spectrum``)."""
        d = (freq_hz - self.freq0_hz) / self.fwidth_hz
        return math.exp(-0.5 * d * d)

    @property
    def dc_amplitude(self) -> float:
        """Relative spectral weight the pulse places on DC,
        ``spectral_amplitude(0)`` = ``exp(-(freq0/fwidth)^2/2)``. A large value
        is the over-broadband instability tell (even a few percent is risky);
        :meth:`for_band` keeps this far below 1."""
        return self.spectral_amplitude(0.0)

    # --- band -> pulse tuning (settling-aware) ---

    @classmethod
    def for_band(
        cls,
        *,
        freqs_hz: Optional[Sequence[float]] = None,
        wavelengths_um: Optional[Sequence[float]] = None,
        freq0_hz: Optional[float] = None,
        band_sigmas: float = 1.0,
        offset: float = 5.0,
        phase: float = 0.0,
        min_dc_sigmas: float = 4.0,
    ) -> "GaussianPulse":
        """Pick ``freq0_hz``/``fwidth_hz`` to cover a measurement band with the
        SHORTEST settling that band justifies (NUMERICS.md §5/§7).

        Exactly one of ``freqs_hz`` or ``wavelengths_um`` defines the band (only
        its extremes matter). ``freq0_hz`` defaults to the band centre
        ``(fmin+fmax)/2``. ``fwidth_hz`` is set so the further band edge sits at
        ``band_sigmas`` standard deviations from ``freq0``: ``band_sigmas=1``
        (default) puts the edges at ~61% spectral amplitude, the broadest pulse
        — hence the shortest :attr:`source_end_time_s` — that covers the band
        near-flat without spilling much energy outside it. Lower ``band_sigmas``
        broadens the pulse further (settling shrinks, the band flattens toward
        the peak) at the cost of spectral density spread outside the band (lower
        in-band SNR) and DC margin; higher ``band_sigmas`` narrows it,
        concentrating spectral density near ``freq0`` (higher central SNR) but
        rolling off the band edges and settling slower.

        ``fwidth`` is capped at ``freq0 / min_dc_sigmas`` so DC stays at least
        ``min_dc_sigmas`` (default 4 → DC amplitude ~3e-4) standard deviations
        from ``freq0``; past this the broadband drive dumps DC into its lowest
        carrier and the run goes unstable. A band too wide for that cap is
        clamped (narrower fwidth, longer settling, under-driven edges) with a
        warning — such a band wants a broadband mode source (``num_freqs``),
        not a single Gaussian.

        Returns a :class:`GaussianPulse`; read :attr:`source_end_time_s` on it
        to see the resulting settling floor.
        """
        if (freqs_hz is None) == (wavelengths_um is None):
            raise ValueError("for_band: pass exactly one of freqs_hz or wavelengths_um")
        if band_sigmas <= 0.0:
            raise ValueError("for_band: band_sigmas must be > 0")
        if min_dc_sigmas <= 0.0:
            raise ValueError("for_band: min_dc_sigmas must be > 0")

        if wavelengths_um is not None:
            lams = [float(w) for w in wavelengths_um]
            if not lams or any(w <= 0.0 for w in lams):
                raise ValueError("for_band: wavelengths_um must be positive")
            freqs = [_C0_M_PER_S / (w * 1e-6) for w in lams]
        else:
            freqs = [float(f) for f in freqs_hz]
            if not freqs or any(f <= 0.0 for f in freqs):
                raise ValueError("for_band: freqs_hz must be positive")
        fmin, fmax = min(freqs), max(freqs)
        if fmax <= fmin:
            raise ValueError(
                "for_band: need a non-degenerate band (>= 2 distinct "
                "frequencies); a single frequency has no width — set fwidth_hz "
                "explicitly"
            )

        f0 = (fmin + fmax) / 2.0 if freq0_hz is None else float(freq0_hz)
        if not (fmin <= f0 <= fmax):
            raise ValueError(
                f"for_band: freq0_hz ({f0:.4g}) must lie inside the band "
                f"[{fmin:.4g}, {fmax:.4g}] Hz"
            )
        half_band = max(f0 - fmin, fmax - f0)  # cover the further edge
        fwidth = half_band / band_sigmas
        fwidth_dc_cap = f0 / min_dc_sigmas
        if fwidth > fwidth_dc_cap:
            warnings.warn(
                f"for_band: band half-width {half_band:.4g} Hz at "
                f"band_sigmas={band_sigmas:g} wants fwidth {fwidth:.4g} Hz, "
                f"placing DC only {f0 / fwidth:.2f} sigma from freq0 "
                f"({f0:.4g} Hz) — clamping to {fwidth_dc_cap:.4g} Hz "
                f"(min_dc_sigmas={min_dc_sigmas:g}) to stay stable; band edges "
                "will be under-driven. Use a broadband mode source (num_freqs) "
                "for a band this wide.",
                stacklevel=2,
            )
            fwidth = fwidth_dc_cap
        return cls(freq0_hz=f0, fwidth_hz=fwidth, offset=offset, phase=phase)


# Single member today; new time dependences slot into the union.
SourceTimeType = Annotated[Union[GaussianPulse], Field(discriminator="type")]
