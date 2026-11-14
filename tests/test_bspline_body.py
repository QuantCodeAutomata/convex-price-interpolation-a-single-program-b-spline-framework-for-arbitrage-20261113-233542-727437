"""Unit tests for the B-spline body basis and its derivative maps (paper,
Sections 3-4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.bspline_body import build_body_basis, smoothness_penalty_matrix


@pytest.fixture
def cubic_basis():
    return build_body_basis(x_l=0.5, x_h=2.0, degree=3, n_interior=10, q=0.5)


@pytest.fixture
def quartic_basis():
    return build_body_basis(x_l=0.5, x_h=2.0, degree=4, n_interior=10, q=0.5)


def test_basis_partition_of_unity(cubic_basis):
    x = np.linspace(cubic_basis.x_l, cubic_basis.x_h, 50)
    design = cubic_basis.design_matrix(x, nu=0)
    np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-8)


def test_endpoint_interpolation(cubic_basis):
    """Clamped knots: the first/last basis function equals 1 at x_l/x_h."""
    design = cubic_basis.design_matrix(np.array([cubic_basis.x_l, cubic_basis.x_h]), nu=0)
    assert design[0, 0] == pytest.approx(1.0, abs=1e-8)
    assert design[1, -1] == pytest.approx(1.0, abs=1e-8)


def test_derivative_maps_match_finite_difference(cubic_basis):
    """D1, D2 applied to a random coefficient vector reproduce the finite-
    difference derivative of the resulting spline curve."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=cubic_basis.n_coef)
    x0 = 1.1
    eps = 1e-6
    c_plus = cubic_basis.design_matrix(np.array([x0 + eps]), nu=0) @ d
    c_minus = cubic_basis.design_matrix(np.array([x0 - eps]), nu=0) @ d
    fd_first = (c_plus - c_minus) / (2 * eps)
    analytic_first = cubic_basis.design_matrix(np.array([x0]), nu=1) @ d
    np.testing.assert_allclose(analytic_first, fd_first, rtol=1e-3, atol=1e-5)

    c0 = cubic_basis.design_matrix(np.array([x0]), nu=0) @ d
    fd_second = (c_plus - 2 * c0 + c_minus) / eps**2
    analytic_second = cubic_basis.design_matrix(np.array([x0]), nu=2) @ d
    np.testing.assert_allclose(analytic_second, fd_second, rtol=1e-2, atol=1e-4)


def test_monotone_decreasing_coefficients_give_monotone_spline(cubic_basis):
    """D_1 d <= 0 (paper's monotonicity constraint) should indeed force the
    evaluated curve to be non-increasing when the coefficients satisfy it."""
    d = np.linspace(1.5, 0.0, cubic_basis.n_coef)  # strictly decreasing control points
    e = cubic_basis.D1 @ d
    assert np.all(e <= 1e-10)
    x = np.linspace(cubic_basis.x_l, cubic_basis.x_h, 100)
    curve = cubic_basis.design_matrix(x, nu=0) @ d
    assert np.all(np.diff(curve) <= 1e-8)


def test_seam_quantities_chain_rule(cubic_basis):
    x_s = cubic_basis.x_h
    k_s = np.log(x_s)
    c0, c1, c2 = cubic_basis.seam_quantities(x_s, k_s)
    rng = np.random.default_rng(1)
    d = rng.normal(size=cubic_basis.n_coef)
    eps = 1e-6
    k_plus, k_minus = k_s + eps, k_s - eps
    val_plus = cubic_basis.design_matrix(np.array([np.exp(k_plus)]), nu=0) @ d
    val_minus = cubic_basis.design_matrix(np.array([np.exp(k_minus)]), nu=0) @ d
    val0 = cubic_basis.design_matrix(np.array([x_s]), nu=0) @ d
    fd_dk = (val_plus - val_minus) / (2 * eps)
    fd_d2k = (val_plus - 2 * val0 + val_minus) / eps**2
    np.testing.assert_allclose(c0 @ d, val0, rtol=1e-8)
    np.testing.assert_allclose(c1 @ d, fd_dk, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(c2 @ d, fd_d2k, rtol=1e-2, atol=1e-4)


def test_smoothness_penalty_psd(cubic_basis):
    R = smoothness_penalty_matrix(cubic_basis, s_n=0.1)
    assert R.shape == (cubic_basis.n_coef, cubic_basis.n_coef)
    np.testing.assert_allclose(R, R.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals >= -1e-8)


def test_smoothness_penalty_zero_for_quadratic_degree():
    basis = build_body_basis(x_l=0.5, x_h=2.0, degree=2, n_interior=5, q=0.5)
    R = smoothness_penalty_matrix(basis, s_n=0.1)
    np.testing.assert_allclose(R, 0.0)


def test_knot_concentration_near_the_money(cubic_basis):
    """Interior knots are concentrated near x=1 (the money) relative to a
    uniform-in-x grid, since the paper places them uniform in sgn(k)|k|^q
    with q<1 (Section 3)."""
    interior = cubic_basis.t[cubic_basis.degree + 1: -(cubic_basis.degree + 1)]
    gaps = np.diff(np.log(np.sort(interior)))
    # gaps near the centre (log(x) close to 0) should be smaller than gaps
    # near the edges for a concentration exponent q < 1.
    mid = len(gaps) // 2
    assert gaps[mid] < gaps[0] + 1e-9
