"""Unit tests for the closed-form variance-swap functional (paper, Appendix B)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from src.bspline_body import build_body_basis
from src.variance_swap import (
    body_variance_terms,
    build_variance_swap_terms,
    fair_variance_to_volatility,
    wing_variance_contribution,
)
from src.wing_dictionary import build_geometric_dictionary
from src import config


def test_body_variance_terms_match_numeric_integration():
    """A @ d (plus const) must match brute-force numerical integration of
    (c(x)-1+x)/x^2 on [x_l,1] and c(x)/x^2 on [1,x_h] for a random coefficient
    vector, confirming the analytic antiderivative used in the paper's
    replication formula (*)."""
    basis = build_body_basis(x_l=0.5, x_h=2.0, degree=3, n_interior=8, q=0.5)
    rng = np.random.default_rng(2)
    d = rng.normal(loc=0.0, scale=0.1, size=basis.n_coef)
    d[0] = 0.4  # keep curve roughly sane in scale, sign irrelevant for this linear check

    A, const = body_variance_terms(basis)
    analytic = A @ d + const

    def integrand(x):
        c = (basis.design_matrix(np.array([x]), nu=0) @ d)[0]
        return (c - 1.0 + x) / x**2 if x <= 1.0 else c / x**2

    numeric, _ = integrate.quad(integrand, basis.x_l, basis.x_h, points=[1.0], limit=200)
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_wing_variance_contribution_matches_numeric_integration():
    """g_i for a single put atom must match brute-force numerical integration
    of h_i(k) e^{-k} dk over the deep put wing (the substitution underlying
    the closed form, Appendix B derivation note in src/variance_swap.py)."""
    put_dict = build_geometric_dictionary(config.N_PUT_ATOMS, config.PUT_B_LO, config.PUT_B_HI, "put")
    k_s = -0.3
    kappa = abs(k_s)
    g = wing_variance_contribution(put_dict, kappa, "put")

    atom_index = 0  # thinnest-tail atom: fastest-converging integral

    def integrand(u):
        k = k_s - u
        h = put_dict.atom_values(np.array([k]), k_s)[atom_index, 0]
        return h * np.exp(-k)

    numeric, _ = integrate.quad(integrand, 0.0, 60.0, limit=400)
    assert g[atom_index] == pytest.approx(numeric, rel=1e-3)


def test_put_wing_variance_requires_mu_greater_than_one():
    put_dict = build_geometric_dictionary(3, 0.02, 1.95, "put")  # mu close to 1 at b_hi=1.95
    # Force one atom's mu below 1 to trigger the convergence guard.
    put_dict.mu[-1] = 0.99
    with pytest.raises(ValueError):
        wing_variance_contribution(put_dict, 0.3, "put")


def test_fair_variance_to_volatility_scaling():
    # A constant-volatility Black-76 smile has fair variance L = sigma^2 * T
    # exactly (log-contract replication is exact under GBM), so the round
    # trip should recover sigma.
    sigma = 0.2
    t = 0.5
    l_value = sigma**2 * t
    vol = fair_variance_to_volatility(l_value, t)
    assert vol == pytest.approx(sigma, rel=1e-10)


def test_fair_variance_to_volatility_floors_at_zero():
    assert fair_variance_to_volatility(-1e-9, 1.0) == 0.0


def test_build_variance_swap_terms_end_to_end_affine():
    """VarianceSwapTerms.value must be exactly affine (linear + constant) in
    (d, a_L, a_R): doubling the vectors away from a base point must change
    the value consistently with the stored linear maps."""
    basis = build_body_basis(x_l=0.6, x_h=1.8, degree=3, n_interior=6, q=0.5)
    put_dict = build_geometric_dictionary(config.N_PUT_ATOMS, config.PUT_B_LO, config.PUT_B_HI, "put")
    call_dict = build_geometric_dictionary(config.N_CALL_ATOMS, config.CALL_B_LO, config.CALL_B_HI, "call")
    k_s_put, k_s_call = np.log(0.6), np.log(1.8)
    terms = build_variance_swap_terms(basis, put_dict, call_dict, k_s_put, k_s_call)

    rng = np.random.default_rng(3)
    d0 = rng.normal(size=basis.n_coef)
    aL0 = rng.normal(size=put_dict.n_atoms)
    aR0 = rng.normal(size=call_dict.n_atoms)
    d1 = d0 + rng.normal(size=basis.n_coef)
    aL1 = aL0 + rng.normal(size=put_dict.n_atoms)
    aR1 = aR0 + rng.normal(size=call_dict.n_atoms)

    v0 = terms.value(d0, aL0, aR0)
    v1 = terms.value(d1, aL1, aR1)
    predicted_delta = 2.0 * (terms.A @ (d1 - d0) + terms.g_L @ (aL1 - aL0) + terms.g_R @ (aR1 - aR0))
    assert (v1 - v0) == pytest.approx(predicted_delta, rel=1e-8, abs=1e-10)
