"""B-spline body of the normalised call price c(x) = C(Fx)/F (paper, Section 3-4).

Uses scipy.interpolate.BSpline for basis evaluation and for the exact
derivative-coefficient recursion (``BSpline.derivative``), rather than
re-deriving de Boor's formulas from scratch: Context7 (scipy docs) confirms
``BSpline.derivative(nu)`` returns the spline's exact derivative on a reduced
knot vector, which is precisely the linear coefficient map D_1, D_2 the paper
specifies (eq. in Section 4: e_j = p(d_{j+1}-d_j)/(t_{j+p+1}-t_{j+1}), applied
again for f_j). Applying it to an identity coefficient matrix yields exactly
these D_1, D_2 maps as ordinary matrices, which we then use both for the
no-arbitrage coefficient constraints and for the C^2 seam join.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class BodyBasis:
    """A clamped B-spline body of degree ``degree`` on [x_l, x_h].

    t : full clamped knot vector (length n_coef + degree + 1)
    n_coef : number of spline coefficients (basis functions)
    D1, D2 : sparse-equivalent dense maps with e = D1 @ d, f = D2 @ D1 @ d
             giving the coefficients of c'(x) and c''(x) in their own
             (reduced-knot, lower-degree) B-spline bases.
    t1, t2 : knot vectors of the first- and second-derivative bases.
    """

    x_l: float
    x_h: float
    degree: int
    t: np.ndarray
    n_coef: int
    D1: np.ndarray
    t1: np.ndarray
    D2: np.ndarray
    t2: np.ndarray

    def design_matrix(self, x: np.ndarray, nu: int = 0) -> np.ndarray:
        """Dense matrix of (the nu-th x-derivative of) each basis function at points x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if nu == 0:
            t, p = self.t, self.degree
        elif nu == 1:
            t, p = self.t1, self.degree - 1
        elif nu == 2:
            t, p = self.t2, self.degree - 2
        else:
            raise ValueError("nu must be 0, 1 or 2")
        n = len(t) - p - 1
        mat = BSpline.design_matrix(x, t, p, extrapolate=True).toarray()
        assert mat.shape == (len(x), n)
        if nu == 0:
            return mat
        if nu == 1:
            return mat @ self.D1
        return mat @ self.D2

    def seam_quantities(self, x_s: float, k_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row vectors (c0, c1, c2) such that c0@d, c1@d, c2@d give the body's
        value, d/dk and d^2/dk^2 at the seam (paper, eq. 16-18, chain rule
        dc/dk = x*dc/dx, d^2c/dk^2 = x^2*c''(x) + x*c'(x))."""
        b0 = self.design_matrix(np.array([x_s]), nu=0)[0]
        b1 = self.design_matrix(np.array([x_s]), nu=1)[0]
        b2 = self.design_matrix(np.array([x_s]), nu=2)[0]
        c0 = b0
        c1 = x_s * b1
        c2 = x_s**2 * b2 + x_s * b1
        return c0, c1, c2


def _concentrated_knots(x_l: float, x_h: float, n_interior: int, q: float) -> np.ndarray:
    """Interior knots uniform in u = sgn(k)|k|^q, k = log(x), q<1 (paper, Section 3)."""
    k_l, k_h = np.log(x_l), np.log(x_h)

    def to_u(k: float) -> float:
        return np.sign(k) * np.abs(k) ** q

    def from_u(u: float) -> float:
        return np.sign(u) * np.abs(u) ** (1.0 / q)

    u_l, u_h = to_u(k_l), to_u(k_h)
    u_grid = np.linspace(u_l, u_h, n_interior + 2)[1:-1]
    k_grid = np.array([from_u(u) for u in u_grid])
    return np.exp(k_grid)


def build_body_basis(x_l: float, x_h: float, degree: int, n_interior: int, q: float) -> BodyBasis:
    """Clamped knot vector (multiplicity degree+1 at the ends) with n_interior
    interior knots concentrated near the money, and the derivative maps."""
    interior = _concentrated_knots(x_l, x_h, n_interior, q)
    t = np.concatenate([[x_l] * (degree + 1), interior, [x_h] * (degree + 1)])
    n_coef = len(t) - degree - 1

    eye = np.eye(n_coef)
    spline0 = BSpline(t, eye, degree)
    spline1_raw = spline0.derivative(nu=1)
    t1 = spline1_raw.t
    n1 = len(t1) - spline1_raw.k - 1
    D1 = spline1_raw.c[:n1]  # scipy pads c with extra rows beyond the valid basis count
    spline1 = BSpline(t1, D1, spline1_raw.k)  # re-wrap with the trimmed, valid coefficients

    # spline1's columns are already indexed by the *original* basis j (its
    # vector-valued output dimension), so differentiating it again yields the
    # full composed map D2 (f = D2 @ d) directly, with no further product by D1.
    spline2_raw = spline1.derivative(nu=1)
    t2 = spline2_raw.t
    n2 = len(t2) - spline2_raw.k - 1
    D2 = spline2_raw.c[:n2]

    return BodyBasis(x_l=x_l, x_h=x_h, degree=degree, t=t, n_coef=n_coef, D1=D1, t1=t1, D2=D2, t2=t2)


def smoothness_penalty_matrix(basis: BodyBasis, s_n: float, n_quad_per_span: int = 5) -> np.ndarray:
    """R with R_jj' = s_n^3 * integral of B_j'''(x) B_j'''(x) dx (paper, Section 7),
    via Gauss-Legendre quadrature on each distinct knot span of the third-derivative
    basis (exact for the polynomial integrand up to quadrature order)."""
    p = basis.degree
    if p < 3:
        return np.zeros((basis.n_coef, basis.n_coef))
    eye = np.eye(basis.n_coef)
    spline0 = BSpline(basis.t, eye, p)
    spline3 = spline0.derivative(nu=3)
    t3 = spline3.t
    breaks = np.unique(t3)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad_per_span)
    R = np.zeros((basis.n_coef, basis.n_coef))
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue
        x_quad = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        jac = 0.5 * (b - a)
        vals = spline3(x_quad)  # shape (n_quad, n_coef)
        R += (vals.T * weights * jac) @ vals
    return s_n**3 * R
