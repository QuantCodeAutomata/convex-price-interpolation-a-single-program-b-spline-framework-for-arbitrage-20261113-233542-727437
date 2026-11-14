"""Fair variance swap level as an affine functional of the QP variables
(paper, Appendix B / Section 8 "Variance swap quotes").

The replication formula for the fair variance of log(S_T/F) is

    L = 2 [ int_0^1 (c(x)-1+x)/x^2 dx + int_1^infty c(x)/x^2 dx ]        (*)

Both pieces are linear in the body coefficients d and the wing weights
a_L, a_R, so L_n = A_n @ d_n + g_L_n @ a_L_n + g_R_n @ a_R_n + const_n for
every maturity n. scipy.special.erfcx (Context7: scipy.special is the
standard library for the scaled complementary error function; no portfolio/
options library implements this paper-specific closed form) avoids the
cancellation of a huge exp(.) against a tiny erfc(.) in the wing integral.

Derivation of the wing closed form (checked against eq. derived from h_i,
paper eq. 9): on the put wing, x=e^k, k<0, kappa=|k_s|. Substituting
u=-k turns int_{-infty}^{k_s} h_i(k) e^{-k} dk into

    sqrt(kappa) exp(mu_i*kappa) int_kappa^infty u^{-1/2} exp(-nu_i u) du,
    nu_i = mu_i - 1,

and int_a^infty u^{-1/2} e^{-nu u} du = sqrt(pi/nu) erfc(sqrt(nu a)); writing
erfc(z) = exp(-z^2) erfcx(z) cancels mu_i*kappa - nu_i*kappa = kappa, giving
g_i = sqrt(pi/nu_i) * sqrt(kappa) * exp(+kappa) * erfcx(sqrt(nu_i*kappa)).
The call wing follows the same substitution with nu_i = mu_i + 1 and a sign
flip, giving exp(-kappa) instead of exp(+kappa) (the "upper/lower sign"
put/call distinction in the paper's Appendix A/B).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import comb, erfcx

from src.bspline_body import BodyBasis
from src.wing_dictionary import WingDictionary


def wing_variance_contribution(dictionary: WingDictionary, kappa: float, side: str) -> np.ndarray:
    """g_i for every atom of one wing's dictionary at seam magnitude kappa=|k_s|."""
    if side == "put":
        nu = dictionary.mu - 1.0
        sign = 1.0
    elif side == "call":
        nu = dictionary.mu + 1.0
        sign = -1.0
    else:
        raise ValueError("side must be 'put' or 'call'")
    if side == "put" and np.any(nu <= 0):
        raise ValueError("put-wing variance-swap integral requires mu_i > 1 (nu_i > 0) for every atom")
    z = np.sqrt(nu * kappa)
    return np.sqrt(np.pi / nu) * np.sqrt(kappa) * np.exp(sign * kappa) * erfcx(z)


def _local_power_coeffs(basis: BodyBasis) -> tuple[np.ndarray, np.ndarray]:
    """Breakpoints and, for each span i and local power m=0..p, the row vector
    gamma_m(i,:) (linear in d) such that c(x) = sum_m gamma_m(i,:)@d * (x-a_i)^m
    on span [a_i, a_{i+1}] (Taylor expansion of the clamped B-spline basis)."""
    p = basis.degree
    eye = np.eye(basis.n_coef)
    spline0 = BSpline(basis.t, eye, p)
    derivs = [spline0]
    cur = spline0
    for _ in range(p):
        cur = cur.derivative(nu=1)
        derivs.append(cur)
    breaks = np.unique(basis.t[p: len(basis.t) - p])
    n_spans = len(breaks) - 1
    coeffs = np.zeros((n_spans, p + 1, basis.n_coef))
    for i in range(n_spans):
        a, b = breaks[i], breaks[i + 1]
        x_eval = np.array([a + 1e-9 * max(b - a, 1e-12)])
        for m in range(p + 1):
            coeffs[i, m, :] = derivs[m](x_eval)[0] / factorial(m)
    return breaks, coeffs


def _shift_to_absolute_powers(local_coeffs: np.ndarray, a: float) -> np.ndarray:
    """Binomial expansion (x-a)^m = sum_j C(m,j) x^j (-a)^{m-j}, converting
    local Taylor coefficients gamma_m into absolute-power coefficients
    Gamma_j (paper's implementation_notes power basis Sigma gamma_m x^m)."""
    p1, n_coef = local_coeffs.shape
    p = p1 - 1
    absolute = np.zeros_like(local_coeffs)
    for m in range(p + 1):
        for j in range(m + 1):
            absolute[j] += comb(m, j) * (-a) ** (m - j) * local_coeffs[m]
    return absolute


def _definite_integral_x_power_over_x2(absolute_coeffs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """int_lo^hi (sum_j Gamma_j x^j) / x^2 dx, as a row vector linear in d
    (paper, implementation_notes: -gamma_0/x + gamma_1 log x + sum_{m>=2} gamma_m x^{m-1}/(m-1))."""
    p = absolute_coeffs.shape[0] - 1
    n_coef = absolute_coeffs.shape[1]
    total = np.zeros(n_coef)

    def antideriv(x: float) -> np.ndarray:
        val = np.zeros(n_coef)
        val += -absolute_coeffs[0] / x
        if p >= 1:
            val += absolute_coeffs[1] * np.log(x)
        for j in range(2, p + 1):
            val += absolute_coeffs[j] * x ** (j - 1) / (j - 1)
        return val

    total = antideriv(hi) - antideriv(lo)
    return total


def body_variance_terms(basis: BodyBasis) -> tuple[np.ndarray, float]:
    """Body's contribution to the variance-swap integral (*): a row vector A
    (A @ d is the d-linear part) and a constant (from the put-side intrinsic
    term int_{x_l}^{min(x_h,1)} (x-1)/x^2 dx = [log x + 1/x])."""
    breaks, local_coeffs = _local_power_coeffs(basis)
    n_coef = basis.n_coef
    A = np.zeros(n_coef)
    const = 0.0
    for i in range(len(breaks) - 1):
        a, b = breaks[i], breaks[i + 1]
        absolute = _shift_to_absolute_powers(local_coeffs[i], a)
        if b <= 1.0:
            A += _definite_integral_x_power_over_x2(absolute, a, b)
            const += (np.log(b) + 1.0 / b) - (np.log(a) + 1.0 / a)
        elif a >= 1.0:
            A += _definite_integral_x_power_over_x2(absolute, a, b)
        else:
            A += _definite_integral_x_power_over_x2(absolute, a, 1.0)
            const += (np.log(1.0) + 1.0) - (np.log(a) + 1.0 / a)
            A += _definite_integral_x_power_over_x2(absolute, 1.0, b)
    return A, const


@dataclass
class VarianceSwapTerms:
    """L_n = 2*(A @ d + g_L @ a_L + g_R @ a_R + const)."""

    A: np.ndarray
    g_L: np.ndarray
    g_R: np.ndarray
    const: float

    def value(self, d: np.ndarray, a_L: np.ndarray, a_R: np.ndarray) -> float:
        return 2.0 * (self.A @ d + self.g_L @ a_L + self.g_R @ a_R + self.const)

    def expr(self, d, a_L, a_R):
        """Build the same affine value as a cvxpy expression for use inside the QP."""
        return 2.0 * (self.A @ d + self.g_L @ a_L + self.g_R @ a_R + self.const)


def build_variance_swap_terms(
    basis: BodyBasis, put_dict: WingDictionary, call_dict: WingDictionary, k_s_put: float, k_s_call: float
) -> VarianceSwapTerms:
    A, const = body_variance_terms(basis)
    g_L = wing_variance_contribution(put_dict, abs(k_s_put), "put")
    g_R = wing_variance_contribution(call_dict, abs(k_s_call), "call")
    return VarianceSwapTerms(A=A, g_L=g_L, g_R=g_R, const=const)


def fair_variance_to_volatility(fair_variance: float, t_expiry: float) -> float:
    """Annualised variance-swap volatility sqrt(L/T) from the fair variance level."""
    return float(np.sqrt(max(fair_variance, 0.0) / t_expiry))
