"""Normalised Black-76 call price, vega and implied-volatility inversion.

Working in the paper's normalised coordinates x = K/F, c(x) = C(Fx)/F, the
forward is 1 and the formulas below are the standard Black-76 formulas with
F = 1 and strike x. scipy has no dedicated option-pricing routine, so the
closed-form Black-76 price (scipy.stats.norm for the CDF, scipy.optimize.brentq
for implied-volatility inversion) is implemented directly from its standard
formula.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def normalized_call_price(x: np.ndarray, vol: np.ndarray, t: float) -> np.ndarray:
    """c(x) = N(d1) - x*N(d2), the undiscounted Black-76 call price with F=1."""
    x = np.asarray(x, dtype=float)
    vol = np.asarray(vol, dtype=float)
    sqrt_t = np.sqrt(t)
    d1 = (-np.log(x) + 0.5 * vol**2 * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return norm.cdf(d1) - x * norm.cdf(d2)


def normalized_vega(x: np.ndarray, vol: np.ndarray, t: float) -> np.ndarray:
    """dc/dvol at F=1, i.e. the vega of the normalised call price."""
    x = np.asarray(x, dtype=float)
    vol = np.asarray(vol, dtype=float)
    sqrt_t = np.sqrt(t)
    d1 = (-np.log(x) + 0.5 * vol**2 * t) / (vol * sqrt_t)
    return sqrt_t * norm.pdf(d1)


def implied_vol(c: float, x: float, t: float, lo: float = 1e-4, hi: float = 5.0) -> float:
    """Invert normalized_call_price for the volatility, by bisection (Brent).

    Returns NaN if c lies outside the model's achievable price range
    [max(1-x, 0), 1] for any volatility (an arbitrage violation or a
    pathological fitted price), rather than returning a spurious root.
    """
    intrinsic = max(1.0 - x, 0.0)
    if not (intrinsic - 1e-10 <= c <= 1.0 + 1e-10):
        return float("nan")
    c = min(max(c, intrinsic + 1e-12), 1.0 - 1e-12)

    def obj(vol: float) -> float:
        return normalized_call_price(x, vol, t) - c

    f_lo, f_hi = obj(lo), obj(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    return brentq(obj, lo, hi, xtol=1e-10)


def implied_vol_array(c: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    return np.array([implied_vol(ci, xi, t) for ci, xi in zip(c, x)])
