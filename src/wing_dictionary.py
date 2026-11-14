"""The asymptotic wing dictionary (paper, Section 5).

Each wing atom is parametrised by a Lee slope b, mapped to a decay rate mu
through the Lee moment formula (paper, eq. 10). An atom's time-value shape is

    h_i(k) = sqrt(|k_s| / |k|) * exp(-mu_i * (|k| - |k_s|)),  |k| >= |k_s|   (paper, eq. 9)

anchored at the seam k_s. mu_i >= 1 on the put wing and mu_i > 0 on the call
wing are the admissibility (butterfly-freeness) conditions (paper, Section 5.1).

Custom implementation: the Lee moment / admissibility algebra and the wing
dictionary are the paper's own construction; no standard library implements
this closed-form butterfly-free tail family (checked: scipy.special offers the
building blocks -- erfcx, etc. -- used in src/variance_swap.py, but not the
wing construction itself).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lee_mu(b: np.ndarray, side: str) -> np.ndarray:
    """Lee moment formula mapping a Lee slope b to a decay rate mu (paper, eq. 10).

    side='call': mu = (2-b)^2 / (8b)
    side='put':  mu = 1 + (2-b)^2 / (8b)
    """
    b = np.asarray(b, dtype=float)
    if np.any(b <= 0) or np.any(b >= 2):
        raise ValueError("Lee slopes must lie strictly inside Lee's bound (0, 2)")
    lam = (2.0 - b) ** 2 / (8.0 * b)
    if side == "call":
        return lam
    if side == "put":
        return 1.0 + lam
    raise ValueError("side must be 'call' or 'put'")


def check_admissibility(mu: np.ndarray, side: str) -> np.ndarray:
    """Butterfly-freeness (admissibility) check (paper, Section 5.1).

    Put wing: admissible iff mu >= 1 (equivalent to Q_-(y) = mu(mu-1) +
    (mu-1/2)y + (3/4)y^2 >= 0 for y in (0, 1/|k_s|]).
    Call wing: admissible iff mu > 0 (equivalent to Q_+(y) = mu(mu+1) +
    (mu+1/2)y + (3/4)y^2 > 0 for all y > 0, which holds for every mu > 0).
    """
    mu = np.asarray(mu, dtype=float)
    if side == "put":
        return mu >= 1.0
    if side == "call":
        return mu > 0.0
    raise ValueError("side must be 'call' or 'put'")


def verify_quadratic_form(mu: np.ndarray, side: str, k_s: float, n_y: int = 200) -> bool:
    """Directly evaluate Q_-(y) or Q_+(y) on a grid y in (0, 1/|k_s|] as a cross-check
    of the mu >= 1 / mu > 0 admissibility shortcut (paper, Section 5.1, implementation_notes).
    """
    y_grid = np.linspace(1e-6, 1.0 / abs(k_s), n_y)
    ok = np.ones(len(np.atleast_1d(mu)), dtype=bool)
    for idx, m in enumerate(np.atleast_1d(mu)):
        if side == "put":
            q = m * (m - 1) + (m - 0.5) * y_grid + 0.75 * y_grid**2
            ok[idx] = np.all(q >= -1e-12)
        else:
            q = m * (m + 1) + (m + 0.5) * y_grid + 0.75 * y_grid**2
            ok[idx] = np.all(q > 0)
    return ok


@dataclass
class WingDictionary:
    """A dictionary of N_a atoms for one wing side, shared across maturities.

    Attributes
    ----------
    side : 'put' or 'call'
    b : Lee slopes, geometric on [b_lo, b_hi], ordered by decreasing mu (thinnest tail first)
    mu : decay rates from the Lee moment formula
    """

    side: str
    b: np.ndarray
    mu: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.b)

    def atom_values(self, k: np.ndarray, k_s: float) -> np.ndarray:
        """h_i(k) for every atom i, evaluated at log-moneyness points k (paper, eq. 9).

        k and k_s must share sign (both negative on the put wing, both
        positive on the call wing); |k| >= |k_s| is required (atoms are
        defined only beyond the seam).
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        if np.any(np.abs(k) < abs(k_s) - 1e-12):
            raise ValueError("atom_values requires |k| >= |k_s|")
        abs_k = np.abs(k)
        abs_ks = abs(k_s)
        ratio = np.sqrt(abs_ks / abs_k)
        decay = np.exp(-np.outer(self.mu, abs_k - abs_ks))
        return ratio[None, :] * decay

    def atom_derivatives(self, k: np.ndarray, k_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """h_i(k), dh_i/dk, d^2h_i/dk^2 for every atom at arbitrary points k inside
        the wing (not only the seam), by the same algebra as seam_matrix (paper,
        eq. 9 differentiated): dh/dk = -sign(k)*h*u(k), d^2h/dk^2 = h*(u^2+1/(2k^2))
        with u(k) = mu_i + 1/(2|k|). k and k_s must share the wing's sign.
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        h = self.atom_values(k, k_s)  # (n_atoms, len(k))
        abs_k = np.abs(k)
        u = self.mu[:, None] + 1.0 / (2.0 * abs_k)[None, :]
        sign = np.sign(k_s) if k_s != 0 else 1.0
        dh = -sign * h * u
        d2h = h * (u**2 + 1.0 / (2.0 * k**2)[None, :])
        return h, dh, d2h

    def seam_matrix(self, k_s: float) -> np.ndarray:
        """3 x N_a dictionary matrix H_s at seam k_s (paper, Section 5.3).

        Rows: atom seam value h_i=1, k-derivative d_k h_i = sign(k_s)*u_i with
        u_i = mu_i + 1/(2|k_s|), and second k-derivative u_i^2 + 1/(2 k_s^2).
        """
        h, dh, d2h = self.atom_derivatives(np.array([k_s]), k_s)
        return np.vstack([h[:, 0], dh[:, 0], d2h[:, 0]])

    def condition_number(self, k_s: float) -> float:
        return float(np.linalg.cond(self.seam_matrix(k_s)))


def build_geometric_dictionary(n_atoms: int, b_lo: float, b_hi: float, side: str) -> WingDictionary:
    """Lay n_atoms Lee slopes geometrically on [b_lo, b_hi] and map through eq. (10),
    ordered by decreasing mu (thinnest tail first, paper Section 5.2)."""
    b = np.geomspace(b_lo, b_hi, n_atoms)
    mu = lee_mu(b, side)
    order = np.argsort(-mu)  # decreasing mu: thinnest tail first
    return WingDictionary(side=side, b=b[order], mu=mu[order])


def build_both_dictionaries(
    n_put: int,
    n_call: int,
    put_b_lo: float,
    put_b_hi: float,
    call_b_lo: float,
    call_b_hi: float,
) -> tuple[WingDictionary, WingDictionary]:
    """Build the shared (put, call) wing dictionaries used by every maturity."""
    put_dict = build_geometric_dictionary(n_put, put_b_lo, put_b_hi, "put")
    call_dict = build_geometric_dictionary(n_call, call_b_lo, call_b_hi, "call")
    return put_dict, call_dict
