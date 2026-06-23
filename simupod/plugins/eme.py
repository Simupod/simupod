"""Minimal eigenmode-expansion (EME) propagator — CPU, frequency-domain.

This is a small, self-contained EME engine built on top of the full-vector FDE
mode solver (:class:`simupod.plugins.vector_modes.VectorModeSolver`). It models
a z-varying device as a **staircase of z-invariant cross-sections** ("sections"),
solves the guided eigenmodes of each, mode-matches at the interfaces, propagates
the modal amplitudes through each section, and cascades the per-interface and
per-section scattering matrices into one device S-matrix.

Method
======
Within section *k* the transverse field is expanded in the local eigenmodes
``(e_m, h_m)`` of that cross-section. With the carried ``exp(i(omega t - beta z))``
convention a **forward** mode propagates as ``exp(-i beta_m z)`` and a **backward**
mode as ``exp(+i beta_m z)``, ``beta_m = k0 * n_eff,m``.

**Interface (mode matching).** Modes are normalized within each section to unit
**unconjugated reciprocity self-overlap** ``<e_m, h_m> = integral (e_m x h_m) .
z_hat dA = 1``, under which distinct modes of one section are orthogonal
(``<e_m, h_n> = delta_mn``). The step between a left section L and a right section
R is described by the single cross-overlap matrix

    G_mn = <e_Rn, h_Lm> = integral (ex_Rn hy_Lm - ey_Rn hx_Lm) dA   (m: left, n: right) .

Enforcing tangential E-continuity (projected onto the left h-modes) and tangential
H-continuity (projected onto the right e-modes) yields a system in ``G`` and its
transpose whose solution is the **power-conserving** interface S-matrix

    S21 = 2 (I + G^T G)^-1 G^T          S22 = (I + G^T G)^-1 (I - G^T G)
    S11 = 2 G (I + G^T G)^-1 G^T - I    S12 = 2 G (I + G^T G)^-1 ,

with ``[out_L; out_R] = S [in_L; in_R]``. Because the same overlap ``G`` drives
both projections, the block matrix ``[[S11, S12], [S21, S22]]`` is **unitary by
construction** (energy-conserving within the guided basis) for any ``G`` — a
mismatched step scatters power between guided modes and reflects, but never gains
or loses it. For **identical** sections ``G = I`` (orthonormality), so ``S21 = I``
and ``S11 = 0``: a matched step is a clean pass-through. This is a built-in
correctness invariant — a straight (constant-cross-section) "taper" returns
``T == 1``, ``R == 0`` to machine precision, and any cascade of lossless sections
has ``T_total + R_total == 1`` exactly.

**Propagation.** A uniform section of length ``L`` adds only a diagonal modal
phase (no inter-mode coupling): ``S = [[0, Φ], [Φ, 0]]`` with
``Φ = diag(exp(-i k0 n_eff,m L - k0 k_eff,m L))`` — the second term is modal
attenuation (``k_eff > 0`` is loss; zero for a straight lossless mode).

**Cascade.** Per-interface and per-section S-matrices are combined with the
**Redheffer star product**, which (unlike a transfer-matrix product) stays
numerically stable in the presence of evanescent / below-cutoff modes.

S-matrix block convention
=========================
Each S-matrix is the 4-tuple ``(S11, S12, S21, S22)`` of ``N x N`` complex arrays
relating outgoing to incoming modal amplitudes::

    [out_L]   [S11 S12] [in_L]
    [out_R] = [S21 S22] [in_R]

so ``S11`` is left-port reflection, ``S21`` is left->right transmission, etc. The
amplitudes are in the unit-``<m,m>`` modal basis, so ``|S_ij|**2`` is a power
ratio and ``S21[a, b]`` is the field coupling from input mode ``b`` to output
mode ``a``.

Scope and limitations (this is a *minimal* prototype)
=====================================================
* **Guided modes only -> no radiation loss.** The FDE solver returns only the
  discrete guided modes (``n_clad < n_eff < n_core``); the radiation continuum is
  not represented. Since the interface S-matrix is unitary on that basis, the
  device is modelled as **strictly lossless** — ``T_total + R_total == 1`` always,
  and the only loss channels are inter-guided-mode conversion and reflection.
  Radiation loss (which a full-vector FDTD *does* see) is therefore absent, so
  EME slightly *over*-predicts the transmission of a real device. This is exact
  for a lossless single-mode guide and accurate for an adiabatic device (where
  radiation is negligible) — the regime where guided-mode EME is the right tool.
* **Common transverse grid.** All sections must be solved on the *same*
  ``(ny, nx)`` grid and spacing (build them with a fixed ``window_*``/``dl`` via
  :meth:`VectorModeSolver.from_rectangular_core`) so the overlap integrals are
  same-mesh sums. Cross-mesh overlap (resampling) is future work.
* **Fixed modal basis.** Every interface is truncated to the common number of
  modes ``N = min`` over the sections; there is no auto mode-tracking / CVCS
  sub-cell interpolation (also future work).

See ``benchmarks/eme/taper_eme.py`` for a convergence-vs-Tidy3D validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .vector_modes import VectorMode, VectorModeSolver

__all__ = [
    "Section",
    "EMEResult",
    "interface_smatrix",
    "propagation_smatrix",
    "star_product",
    "cascade",
    "run_eme",
    "waveguide_section",
]

#: An S-matrix as the block 4-tuple ``(S11, S12, S21, S22)``.
SMatrix = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class Section:
    """One z-invariant cross-section of an EME device.

    Attributes
    ----------
    modes:
        The guided eigenmodes of this cross-section (full-vector
        :class:`VectorMode`), as returned by ``VectorModeSolver.solve`` — sorted
        by descending ``n_eff`` (fundamental first). Every section in a device
        must share the same transverse grid.
    length_um:
        Physical length of the section along propagation (microns). ``0.0`` marks
        a **port / semi-infinite lead** — it contributes its interface with its
        neighbour but no propagation phase.
    """

    modes: Sequence[VectorMode]
    length_um: float = 0.0


@dataclass
class EMEResult:
    """Device scattering matrix from :func:`run_eme`.

    The blocks are ``N x N`` (``N`` = the common modal basis size actually used);
    the input/output port bases are the modes of the first/last section.
    """

    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray
    n_modes: int

    @property
    def transmission(self) -> float:
        """Fundamental-to-fundamental power transmission ``|S21[0, 0]|**2``."""
        return float(np.abs(self.s21[0, 0]) ** 2)

    @property
    def reflection(self) -> float:
        """Fundamental-to-fundamental power reflection ``|S11[0, 0]|**2``."""
        return float(np.abs(self.s11[0, 0]) ** 2)

    def transmitted_power(self, input_mode: int = 0) -> float:
        """Total power transmitted into *all* output modes from ``input_mode``."""
        return float(np.sum(np.abs(self.s21[:, input_mode]) ** 2))

    def reflected_power(self, input_mode: int = 0) -> float:
        """Total power reflected into *all* input-side modes from ``input_mode``."""
        return float(np.sum(np.abs(self.s11[:, input_mode]) ** 2))

    def energy_balance(self, input_mode: int = 0) -> float:
        """``T_total + R_total`` for ``input_mode`` — ``1`` for a lossless device
        in a complete basis; ``< 1`` by the power that couples to the (unmodelled)
        radiation continuum."""
        return self.transmitted_power(input_mode) + self.reflected_power(input_mode)


def _transverse(
    modes: Sequence[VectorMode], dl_x_um: float, dl_y_um: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack the transverse fields of ``modes`` as ``(N, npts)`` complex arrays,
    **bi-orthonormalized** under the unconjugated reciprocity product so that
    ``integral (e_m x h_n) . z_hat dA == delta_mn``.

    Two steps: (1) scale each mode by ``1/sqrt(<m,m>)`` (an overall amplitude
    normalization that preserves the physical e:h ratio and removes the arbitrary
    global phase), then (2) **Lowdin-orthonormalize** the basis — apply the
    symmetric transform ``T = P^-1/2`` (``P_mn = <e_m, h_n>`` the within-section
    Gram) to both ``e`` and ``h``. The discretized FDE modes are only
    approximately reciprocity-orthogonal (the residual shrinks with resolution);
    orthonormalizing makes the matched-interface pass-through and the interface
    unitarity *exact* at any resolution, rather than accurate only to the mesh.
    Lowdin's transform is the closest orthonormal set to the originals, so the
    fundamental stays the fundamental to ``O(off-diagonal^2)``.

    Raises if the Gram is not positive definite — a near-degenerate or
    under-resolved basis (e.g. a near-cutoff mode); reduce ``num_modes`` or refine
    the grid. (Lossless straight modes only: the Gram is real-symmetric here;
    lossy/bent bases are out of scope for this minimal prototype.)
    """
    dA = dl_x_um * dl_y_um
    ex = np.array([m.ex.ravel() for m in modes], dtype=complex)
    ey = np.array([m.ey.ravel() for m in modes], dtype=complex)
    hx = np.array([m.hx.ravel() for m in modes], dtype=complex)
    hy = np.array([m.hy.ravel() for m in modes], dtype=complex)
    self_overlap = np.sum(ex * hy - ey * hx, axis=1) * dA  # (N,), complex
    if np.any(np.abs(self_overlap) < 1e-30):
        raise ValueError(
            "a mode has ~zero reciprocity self-overlap; cannot normalize "
            "(degenerate or non-propagating eigenpair in the basis)"
        )
    scale = (1.0 / np.sqrt(self_overlap))[:, None]
    ex, ey, hx, hy = ex * scale, ey * scale, hx * scale, hy * scale
    if len(modes) > 1:
        gram = (ex @ hy.T - ey @ hx.T) * dA  # P_mn = <e_m, h_n>
        gram = 0.5 * (gram + gram.T)  # symmetrize discretization asymmetry
        evals, evecs = np.linalg.eigh(gram)
        if evals.min() <= 1e-8 * abs(evals.max()):
            raise ValueError(
                "ill-conditioned modal basis (near-degenerate or under-resolved "
                f"mode: Gram eigenvalues {np.round(evals.real, 4)}); reduce "
                "num_modes or refine the transverse grid"
            )
        t_inv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T
        ex, ey = t_inv_sqrt @ ex, t_inv_sqrt @ ey
        hx, hy = t_inv_sqrt @ hx, t_inv_sqrt @ hy
    return ex, ey, hx, hy


def interface_smatrix(
    left_modes: Sequence[VectorMode],
    right_modes: Sequence[VectorMode],
    dl_x_um: float,
    dl_y_um: float,
) -> SMatrix:
    """Mode-matching scattering matrix for the step from ``left_modes`` to
    ``right_modes`` (both on the same transverse grid, same count ``N``).

    Uses the Gram-matrix Galerkin mode match (see the module docstring), so for
    identical mode sets it returns the pass-through ``([0], [I], [I], [0])`` to
    machine precision — even when the basis is not orthonormal.

    Returns the block 4-tuple ``(S11, S12, S21, S22)``.
    """
    n_left = len(left_modes)
    n_right = len(right_modes)
    if n_left != n_right:
        raise ValueError(
            f"interface needs equal mode counts, got {n_left} (left) != "
            f"{n_right} (right); truncate to a common basis first"
        )
    n = n_left
    exL, eyL, hxL, hyL = _transverse(left_modes, dl_x_um, dl_y_um)
    exR, eyR, hxR, hyR = _transverse(right_modes, dl_x_um, dl_y_um)
    dA = dl_x_um * dl_y_um
    # Cross-overlap G_mn = <e_Rn, h_Lm> = integral (ex_Rn hy_Lm - ey_Rn hx_Lm) dA
    # (m: left test/mode, n: right mode).
    g = (hyL @ exR.T - hxL @ eyR.T) * dA  # (N, N)
    gt = g.T
    gtg = gt @ g
    inv = np.linalg.inv(np.eye(n, dtype=complex) + gtg)  # (I + G^T G)^-1, SPD
    s21 = 2.0 * (inv @ gt)
    s22 = inv @ (np.eye(n, dtype=complex) - gtg)
    s11 = 2.0 * (g @ inv @ gt) - np.eye(n, dtype=complex)
    s12 = 2.0 * (g @ inv)
    return s11, s12, s21, s22


def propagation_smatrix(modes: Sequence[VectorMode], length_um: float) -> SMatrix:
    """Diagonal propagation S-matrix for a uniform section of ``length_um``.

    Forward and backward both pick up ``Φ = diag(exp(-i k0 n_eff L - k0 k_eff L))``
    with no inter-mode coupling; reflection blocks are zero.
    """
    n = len(modes)
    lam_um = modes[0].wavelength_um
    k0 = 2.0 * np.pi / lam_um  # 1/µm
    n_eff = np.array([m.n_eff for m in modes], dtype=float)
    k_eff = np.array([m.k_eff for m in modes], dtype=float)
    phi = np.exp(-1j * k0 * n_eff * length_um - k0 * k_eff * length_um)
    phase = np.diag(phi).astype(complex)
    zero = np.zeros((n, n), dtype=complex)
    return zero, phase.copy(), phase.copy(), zero.copy()


def star_product(sa: SMatrix, sb: SMatrix) -> SMatrix:
    """Redheffer star product ``Sa ⋆ Sb`` (``Sa`` on the left, ``Sb`` on the
    right), connecting ``Sa``'s right port to ``Sb``'s left port.

    Numerically stable for evanescent modes (the inverted factors are
    ``I - reflection*reflection``, well-conditioned for small reflections).
    """
    a11, a12, a21, a22 = sa
    b11, b12, b21, b22 = sb
    n = a11.shape[0]
    eye = np.eye(n, dtype=complex)
    m_b = np.linalg.inv(eye - b11 @ a22)  # (I - B11 A22)^-1
    m_a = np.linalg.inv(eye - a22 @ b11)  # (I - A22 B11)^-1
    c11 = a11 + a12 @ m_b @ b11 @ a21
    c12 = a12 @ m_b @ b12
    c21 = b21 @ m_a @ a21
    c22 = b22 + b21 @ m_a @ a22 @ b12
    return c11, c12, c21, c22


def cascade(segments: Sequence[SMatrix]) -> SMatrix:
    """Fold a left-to-right sequence of S-matrices with the star product."""
    if not segments:
        raise ValueError("cascade needs at least one S-matrix segment")
    total = segments[0]
    for seg in segments[1:]:
        total = star_product(total, seg)
    return total


def run_eme(sections: Sequence[Section], n_modes: Optional[int] = None) -> EMEResult:
    """Cascade a sequence of :class:`Section`s into one device S-matrix.

    Parameters
    ----------
    sections:
        Cross-sections in propagation order. The first and last define the
        input/output ports (typically length-``0`` leads). All must share the
        same transverse grid ``(ny, nx)`` and spacing.
    n_modes:
        Cap on the modal basis size. The basis used is
        ``min(n_modes, min_k len(sections[k].modes))`` — every section is
        truncated to that many most-confined modes so the interface matrices are
        square. ``None`` (default) uses the largest common basis available.

    Returns
    -------
    EMEResult
        The device scattering matrix and convenience power ratios.
    """
    if len(sections) < 2:
        raise ValueError("run_eme needs at least two sections (in and out ports)")
    counts = [len(s.modes) for s in sections]
    if min(counts) < 1:
        raise ValueError("every section must carry at least one mode")
    n = min(counts)
    if n_modes is not None:
        n = min(n, int(n_modes))
    if n < 1:
        raise ValueError("n_modes must be >= 1")

    ref = sections[0].modes[0]
    shape0 = ref.shape
    dl_x = ref.dl_x_um
    dl_y = ref.dl_y_um
    for k, sec in enumerate(sections):
        for m in sec.modes[:n]:
            if m.shape != shape0:
                raise ValueError(
                    f"section {k} mode grid {m.shape} != port grid {shape0}; "
                    "all sections must share one transverse grid"
                )
            if not (
                np.isclose(m.dl_x_um, dl_x) and np.isclose(m.dl_y_um, dl_y)
            ):
                raise ValueError(
                    f"section {k} grid spacing differs from the port spacing"
                )

    segments: List[SMatrix] = []
    for k, sec in enumerate(sections):
        modes = list(sec.modes[:n])
        if k > 0:
            prev = list(sections[k - 1].modes[:n])
            segments.append(interface_smatrix(prev, modes, dl_x, dl_y))
        if sec.length_um > 0.0:
            segments.append(propagation_smatrix(modes, sec.length_um))

    s11, s12, s21, s22 = cascade(segments)
    return EMEResult(s11=s11, s12=s12, s21=s21, s22=s22, n_modes=n)


def waveguide_section(
    *,
    wavelength_um: float,
    dl_um: float,
    core_w_um: float,
    core_h_um: float,
    n_core: float,
    n_clad: float,
    window_w_um: float,
    window_h_um: float,
    num_modes: int,
    length_um: float = 0.0,
    neff_margin: float = 0.0,
    subpixel: bool = True,
    subpixel_method: str = "tensor",
) -> Section:
    """Solve a centered rectangular-core cross-section and wrap it as a
    :class:`Section`.

    Pass the **same** ``wavelength_um``, ``dl_um``, ``window_w_um`` and
    ``window_h_um`` for every section of a device so they share one transverse
    grid (required by :func:`run_eme`). Only ``core_w_um`` / ``core_h_um`` should
    vary between sections.

    ``neff_margin`` drops modes whose ``n_eff`` is within this margin of the
    cladding index — i.e. **near-cutoff** modes. Those are poorly resolved (large
    evanescent tails into the window walls) and break the within-section
    orthonormality the interface relies on, so a small margin (e.g. ``0.05``)
    keeps the modal basis clean. Default ``0.0`` keeps every guided mode the
    solver returns.
    """
    solver = VectorModeSolver.from_rectangular_core(
        wavelength_um=wavelength_um,
        dl_um=dl_um,
        core_w_um=core_w_um,
        core_h_um=core_h_um,
        n_core=n_core,
        n_clad=n_clad,
        window_w_um=window_w_um,
        window_h_um=window_h_um,
        subpixel=subpixel,
        subpixel_method=subpixel_method,  # type: ignore[arg-type]
    )
    modes = solver.solve(num_modes=num_modes)
    if neff_margin > 0.0:
        modes = tuple(m for m in modes if m.n_eff > n_clad + neff_margin)
    if not modes:
        raise ValueError(
            f"no guided modes for core_w={core_w_um} um at lambda={wavelength_um} "
            f"um (neff_margin={neff_margin} may be too strict)"
        )
    return Section(modes=modes, length_um=length_um)
