"""Unit tests for the wing dictionary (paper, Section 5)."""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.wing_dictionary import (
    build_both_dictionaries,
    build_geometric_dictionary,
    check_admissibility,
    lee_mu,
    verify_quadratic_form,
)


def test_lee_mu_call_put_relationship():
    b = np.array([0.1, 0.5, 1.0, 1.5])
    mu_call = lee_mu(b, "call")
    mu_put = lee_mu(b, "put")
    # paper eq. 10: mu_put = 1 + mu_call for the same Lee slope.
    np.testing.assert_allclose(mu_put, 1.0 + mu_call)


def test_lee_mu_rejects_outside_lee_bound():
    with pytest.raises(ValueError):
        lee_mu(np.array([0.0, 1.0]), "call")
    with pytest.raises(ValueError):
        lee_mu(np.array([2.0]), "call")


def test_paper_dictionaries_fully_admissible():
    put_dict, call_dict = build_both_dictionaries(
        config.N_PUT_ATOMS, config.N_CALL_ATOMS,
        config.PUT_B_LO, config.PUT_B_HI, config.CALL_B_LO, config.CALL_B_HI,
    )
    assert put_dict.n_atoms == config.N_PUT_ATOMS
    assert call_dict.n_atoms == config.N_CALL_ATOMS
    assert np.all(check_admissibility(put_dict.mu, "put"))
    assert np.all(check_admissibility(call_dict.mu, "call"))
    # paper's expected ranges (Section 8): put mu in [1.01, 25.5], call mu in [0.35, 99.5].
    assert put_dict.mu.min() >= 1.0
    assert put_dict.mu.max() < 30.0
    assert call_dict.mu.min() > 0.0
    assert call_dict.mu.max() < 110.0
    # atoms ordered by decreasing mu (thinnest tail first).
    assert np.all(np.diff(put_dict.mu) <= 1e-12)
    assert np.all(np.diff(call_dict.mu) <= 1e-12)


def test_quadratic_form_cross_check_matches_shortcut():
    put_dict, call_dict = build_both_dictionaries(
        config.N_PUT_ATOMS, config.N_CALL_ATOMS,
        config.PUT_B_LO, config.PUT_B_HI, config.CALL_B_LO, config.CALL_B_HI,
    )
    k_s_put, k_s_call = -0.3, 0.25
    put_shortcut = check_admissibility(put_dict.mu, "put")
    put_quadratic = verify_quadratic_form(put_dict.mu, "put", k_s_put)
    np.testing.assert_array_equal(put_shortcut, put_quadratic)
    call_shortcut = check_admissibility(call_dict.mu, "call")
    call_quadratic = verify_quadratic_form(call_dict.mu, "call", k_s_call)
    np.testing.assert_array_equal(call_shortcut, call_quadratic)


def test_inadmissible_put_atom_detected():
    # A Lee slope near b=2 gives mu_call -> 0, mu_put -> 1 (boundary); a slope
    # that pushes mu_call negative is impossible within Lee's bound, so we
    # directly probe the admissibility function with a sub-1 mu to confirm it
    # flags it, rather than engineering an out-of-bound atom.
    assert check_admissibility(np.array([0.999]), "put")[0] == False
    assert check_admissibility(np.array([1.0]), "put")[0] == True
    assert check_admissibility(np.array([1e-6]), "call")[0] == True
    assert check_admissibility(np.array([0.0]), "call")[0] == False


def test_atom_values_seam_normalisation():
    d = build_geometric_dictionary(config.N_PUT_ATOMS, config.PUT_B_LO, config.PUT_B_HI, "put")
    k_s = -0.4
    h_at_seam = d.atom_values(np.array([k_s]), k_s)
    np.testing.assert_allclose(h_at_seam[:, 0], 1.0, atol=1e-10)
    # decay beyond the seam: h should strictly decrease as |k| grows past k_s.
    h_deep = d.atom_values(np.array([k_s - 1.0]), k_s)
    assert np.all(h_deep[:, 0] < h_at_seam[:, 0])


def test_atom_values_requires_beyond_seam():
    d = build_geometric_dictionary(config.N_CALL_ATOMS, config.CALL_B_LO, config.CALL_B_HI, "call")
    with pytest.raises(ValueError):
        d.atom_values(np.array([0.1]), k_s=0.5)


def test_seam_matrix_shape_and_conditioning():
    d = build_geometric_dictionary(config.N_CALL_ATOMS, config.CALL_B_LO, config.CALL_B_HI, "call")
    H = d.seam_matrix(0.3)
    assert H.shape == (3, config.N_CALL_ATOMS)
    cond = d.condition_number(0.3)
    assert np.isfinite(cond)
    assert cond > 1.0


def test_atom_derivatives_match_finite_differences():
    d = build_geometric_dictionary(config.N_PUT_ATOMS, config.PUT_B_LO, config.PUT_B_HI, "put")
    k_s = -0.5
    k0 = -1.2
    eps = 1e-6
    h, dh, d2h = d.atom_derivatives(np.array([k0]), k_s)
    h_plus = d.atom_values(np.array([k0 + eps]), k_s)[:, 0]
    h_minus = d.atom_values(np.array([k0 - eps]), k_s)[:, 0]
    fd_first = (h_plus - h_minus) / (2 * eps)
    np.testing.assert_allclose(dh[:, 0], fd_first, rtol=1e-4, atol=1e-6)
    fd_second = (h_plus - 2 * h[:, 0] + h_minus) / eps**2
    np.testing.assert_allclose(d2h[:, 0], fd_second, rtol=1e-3, atol=1e-5)


def test_call_cap_sweep_increases_deep_tail_mass():
    """Moving b_hi up on the call side admits a fatter (smaller-mu) atom, so
    the slowest-decaying atom's deep-tail value should increase monotonically
    with b_hi (paper, Section 8, cap-sensitivity experiment)."""
    k_s_call = 0.25
    k_deep = k_s_call + config.DEEP_TAIL_LOG_MONEYNESS_UNITS
    values = []
    for b_hi in config.CALL_B_HI_SWEEP:
        d = build_geometric_dictionary(config.N_CALL_ATOMS, config.CALL_B_LO, b_hi, "call")
        values.append(d.atom_values(np.array([k_deep]), k_s_call)[-1, 0])
    assert np.all(np.diff(values) > 0)
