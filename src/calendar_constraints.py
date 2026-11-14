"""Calendar (across-maturity) no-arbitrage conditions (paper, Section 6 and
Appendix A), used by the joint surface QP of experiment exp_3.

Three regions of the (x, T) plane require a separate treatment because the
two neighbouring slices' body domains [x_l, x_h] generally differ:

1. Body-body overlap: both slices are represented by their exact B-spline
   body there. Delta c = c_{n+1} - c_n is itself a piecewise polynomial once
   both bodies are expressed on a common, refined knot vector (by knot
   insertion, Boehm's algorithm); non-negativity of all its refined
   coefficients is a sufficient condition for Delta c(x) >= 0 everywhere in
   the overlap (B-spline convex-hull property), which is the paper's
   approach. Context7 (scipy docs): ``scipy.interpolate.insert`` implements
   Boehm's knot-insertion algorithm directly; applying it to a vector-valued
   (identity-coefficient) BSpline turns the insertion into a plain linear
   map, reused here exactly as the D_1, D_2 derivative maps are reused in
   src/bspline_body.py.
2. Body-wing band: only one slice's body extends past the other slice's last
   seam. There the wing side is compared against the body side on a grid,
   with the paper's analytic cubic-interpolation error bound (eq. 19-20)
   added as a safety margin so the inequality remains a valid, if slightly
   conservative, no-arbitrage sufficient condition even though the wing is
   evaluated pointwise rather than through a coefficient comparison.
3. Deep wing-wing region beyond both seams: summation-by-parts on the shared
   atom dictionary (Appendix A), using that the SAME dictionary atoms (same
   mu_i) serve every maturity, so only the per-maturity weights a_i change.

Custom implementation throughout: no optimisation or options library
provides calendar-arbitrage-free spline calibration; cvxpy/Clarabel only
solve the resulting QP (as in src/calibration.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, insert

from src.bspline_body import BodyBasis
from src.wing_dictionary import WingDictionary


def _insert_all(t: np.ndarray, C: np.ndarray, p: int, new_knots: np.ndarray, tol: float = 1e-9):
    cur_t, cur_c = t, C
    for v in np.sort(new_knots):
        if np.any(np.abs(cur_t - v) < tol):
            continue
        spl = BSpline(cur_t, cur_c, p)
        out = insert(float(v), spl, m=1)
        cur_t, cur_c = out.t, out.c
    return cur_t, cur_c


def _greville(t: np.ndarray, p: int, n_coef: int) -> np.ndarray:
    return np.array([np.mean(t[j + 1: j + p + 1]) for j in range(n_coef)])


@dataclass
class BodyOverlap:
    """Linear maps M_n, M_n1 (same number of rows) such that, on the body-body
    overlap of two adjacent maturities, (M_n1 @ d_n1 - M_n @ d_n) >= 0 is a
    sufficient no-calendar-arbitrage condition."""

    M_n: np.ndarray
    M_n1: np.ndarray
    lo: float
    hi: float


def body_body_overlap(basis_n: BodyBasis, basis_n1: BodyBasis, subdivisions: int) -> BodyOverlap | None:
    """Merge the two slices' knots on their domain overlap and cut every
    merged interval into ``subdivisions`` pieces (paper, Section 6)."""
    p = basis_n.degree
    assert basis_n1.degree == p, "calendar body-body condition requires matching spline degree"
    lo, hi = max(basis_n.x_l, basis_n1.x_l), min(basis_n.x_h, basis_n1.x_h)
    if hi <= lo:
        return None

    def interior_in_range(basis: BodyBasis) -> np.ndarray:
        interior = basis.t[p + 1: len(basis.t) - p - 1]
        return interior[(interior > lo + 1e-12) & (interior < hi - 1e-12)]

    merged_breaks = np.unique(np.concatenate([[lo, hi], interior_in_range(basis_n), interior_in_range(basis_n1)]))
    fine_pts: list[float] = []
    for a, b in zip(merged_breaks[:-1], merged_breaks[1:]):
        fine_pts.extend(np.linspace(a, b, subdivisions + 1)[:-1].tolist())
    fine_pts.append(float(merged_breaks[-1]))
    fine = np.unique(np.array(fine_pts))
    new_pts = fine[(fine > lo + 1e-12) & (fine < hi - 1e-12)]

    t_n_new, C_n = _insert_all(basis_n.t, np.eye(basis_n.n_coef), p, new_pts)
    t_n1_new, C_n1 = _insert_all(basis_n1.t, np.eye(basis_n1.n_coef), p, new_pts)
    n_n = len(t_n_new) - p - 1
    n_n1 = len(t_n1_new) - p - 1
    greville_n = _greville(t_n_new, p, n_n)
    greville_n1 = _greville(t_n1_new, p, n_n1)

    def safe_idx(t_new: np.ndarray, n_coef: int) -> np.ndarray:
        return np.array([j for j in range(n_coef) if t_new[j] >= lo - 1e-9 and t_new[j + p + 1] <= hi + 1e-9])

    idx_n = safe_idx(t_n_new, n_n)
    idx_n1 = safe_idx(t_n1_new, n_n1)

    # Match by Greville abscissa: shared interior knot structure in (lo, hi)
    # makes safe control points of n and n1 coincide exactly in location;
    # boundary-adjacent points whose support reaches a clamped edge of only
    # one of the two slices will not find a match and are conservatively
    # dropped (they are covered by the body's own endpoint constraints).
    matched_n, matched_n1 = [], []
    for j in idx_n:
        g = greville_n[j]
        k = idx_n1[np.argmin(np.abs(greville_n1[idx_n1] - g))] if len(idx_n1) else None
        if k is not None and abs(greville_n1[k] - g) < 1e-7:
            matched_n.append(j)
            matched_n1.append(k)
    if not matched_n:
        return None
    return BodyOverlap(M_n=C_n[matched_n], M_n1=C_n1[matched_n1], lo=lo, hi=hi)


def body_body_calendar_grid_rows(basis_n: BodyBasis, basis_n1: BodyBasis, n_grid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The 'grid-based' baseline for comparison with the coefficient-based
    condition above: plain evaluation of both bodies at ``n_grid`` points of
    the overlap, with no coefficient merge and no safety margin (paper,
    Section 8, "grid-based calendar imposition")."""
    lo, hi = max(basis_n.x_l, basis_n1.x_l), min(basis_n.x_h, basis_n1.x_h)
    if hi <= lo:
        return None
    x_grid = np.linspace(lo, hi, n_grid)
    return basis_n.design_matrix(x_grid, nu=0), basis_n1.design_matrix(x_grid, nu=0)


def _atom_third_derivative_bound(dictionary: WingDictionary, k_lo: float, k_hi: float, k_s: float, n_sample: int = 50) -> np.ndarray:
    """max |h_i'''(k)| on [k_lo, k_hi] via a central-difference estimate of the
    (analytically available) second derivative's slope, used only to size the
    conservative error margin of eq. (19)-(20); not used anywhere in the fit
    itself, so a numerical estimate is adequate."""
    k_grid = np.linspace(min(k_lo, k_hi), max(k_lo, k_hi), n_sample)
    eps = max(1e-6, (k_grid[1] - k_grid[0]) * 1e-3)
    sign_ks = np.sign(k_s) if k_s != 0 else 1.0
    # Clip the finite-difference offsets so every sampled point stays on the
    # wing's domain |k| >= |k_s| even when the band's near edge sits exactly
    # at the seam.
    k_plus = sign_ks * np.maximum(sign_ks * (k_grid + eps), abs(k_s))
    k_minus = sign_ks * np.maximum(sign_ks * (k_grid - eps), abs(k_s))
    _, _, d2h_plus = dictionary.atom_derivatives(k_plus, k_s)
    _, _, d2h_minus = dictionary.atom_derivatives(k_minus, k_s)
    d3h = (d2h_plus - d2h_minus) / (2 * eps)
    return np.max(np.abs(d3h), axis=1)


@dataclass
class BodyWingBand:
    """Grid-point rows for a band where slice n1's body is compared against
    slice n's wing (or vice versa), with per-atom error margins eps_i such
    that true_wing(x) is within eps_i of its grid value (paper eq. 19-20)."""

    x_grid: np.ndarray
    body_design: np.ndarray       # (n_grid, n_coef) design matrix of the body side
    wing_h: np.ndarray            # (n_atoms, n_grid) exact atom values of the wing side
    eps: np.ndarray                # (n_atoms,) conservative per-atom error bound


def body_wing_band(
    body_basis: BodyBasis,
    wing_dict: WingDictionary,
    k_s: float,
    band_lo: float,
    band_hi: float,
    error_fraction: float,
) -> BodyWingBand | None:
    """Band where ``body_basis`` (one slice) overlaps the wing ``wing_dict``
    of the other slice, spaced so the Hermite interpolation error bound
    eps_ij = dx^4/384 * max|h_i'''| stays within ``error_fraction`` of h_i
    at the band's outer edge (paper, Section 6, eq. 19-20)."""
    if band_hi <= band_lo:
        return None
    k_lo, k_hi = np.log(band_lo), np.log(band_hi)
    third_bound = _atom_third_derivative_bound(wing_dict, k_lo, k_hi, k_s)
    h_edge = wing_dict.atom_values(np.array([k_hi if abs(k_hi) > abs(k_lo) else k_lo]), k_s)[:, 0]
    target_eps = error_fraction * np.maximum(h_edge, 1e-12)
    with np.errstate(divide="ignore"):
        dx_needed = (384.0 * target_eps / np.maximum(third_bound, 1e-300)) ** 0.25
    dx = max(np.min(dx_needed), (abs(k_hi - k_lo)) / 500.0)
    n_grid = max(3, int(np.ceil(abs(k_hi - k_lo) / dx)) + 1)
    k_grid = np.linspace(k_lo, k_hi, n_grid)
    x_grid = np.exp(k_grid)
    eps = (np.diff(k_grid)[0] if n_grid > 1 else 0.0) ** 4 / 384.0 * third_bound
    h_vals = wing_dict.atom_values(k_grid, k_s)
    body_design = body_basis.design_matrix(x_grid, nu=0)
    return BodyWingBand(x_grid=x_grid, body_design=body_design, wing_h=h_vals, eps=eps)


def wing_summation_by_parts_beta(a, mu: np.ndarray, kappa: float):
    """beta_i = a_i * kappa^(1/2) * exp(mu_i * kappa) (paper, Appendix A).

    ``a`` may be a plain numpy array or a cvxpy vector expression. Plain
    ``*`` between two length-N arrays dispatches to matrix (dot) product in
    cvxpy, collapsing the vector to a scalar; ``cp.multiply`` is required
    for the elementwise product needed here.
    """
    factor = np.sqrt(kappa) * np.exp(mu * kappa)
    if isinstance(a, np.ndarray):
        return a * factor
    import cvxpy as cp

    return cp.multiply(a, factor)


def wing_calendar_cumulative_rows(mu: np.ndarray) -> np.ndarray:
    """Upper-triangular summation matrix U (N_a x N_a) with U[m, i] = 1 for
    i >= m, so that U @ beta gives S_m = sum_{i>=m} beta_i for m=1..N_a
    (paper, Appendix A summation by parts, atoms already ordered by
    decreasing mu as in src/wing_dictionary.py)."""
    n = len(mu)
    return np.triu(np.ones((n, n)))


def wing_calendar_normalisation(mu: np.ndarray, kappa_bar: float) -> np.ndarray:
    """psi_m(kappa_bar) = kappa_bar^(-1/2) * exp(-mu_m * kappa_bar), used to
    rescale the summation-by-parts margin into natural units (paper,
    Appendix A)."""
    return np.exp(-mu * kappa_bar) / np.sqrt(kappa_bar)
