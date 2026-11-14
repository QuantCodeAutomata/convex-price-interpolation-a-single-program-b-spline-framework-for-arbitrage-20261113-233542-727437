"""Single-maturity body+wing calibration as a convex QP (paper, Sections 3-5, 7).

Builds the normalised call price c(x) = C(Fx)/F for one maturity from a
cubic or quartic B-spline body on [x_l, x_h] joined in C^2 to put/call wing
atoms, under the linear no-arbitrage constraints of Section 4 and the C^2
join equalities of Section 5.3, and solves the weighted least-squares fit
with smoothness and ridge penalties (Section 7) as a quadratic program.

cvxpy + Clarabel (cvxpy is the standard Python convex-optimisation modelling
layer; Clarabel is registered as a cvxpy conic solver and is the solver the
paper names) are used to assemble and solve the program; no
library implements the paper's specific no-arbitrage QP, so the constraint
and objective assembly below is a custom implementation of Sections 4, 5, 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from data.data_loader import SliceData
from src import bspline_body, black76
from src.wing_dictionary import WingDictionary


@dataclass
class SliceFitInputs:
    """Market data for one maturity, normalised and filtered to OTM quotes."""

    expiry_date: object
    settlement: str
    t_expiry: float
    forward: float
    discount_factor: float
    x: np.ndarray            # normalised strikes K/F, sorted ascending
    c_mkt: np.ndarray        # normalised call-equivalent mid price via parity
    c_bid: np.ndarray
    c_ask: np.ndarray
    vol_bid: np.ndarray
    vol_ask: np.ndarray
    vol_mid: np.ndarray
    vega: np.ndarray         # normalised Black-76 vega at vol_mid
    weight: np.ndarray       # omega_i, clipped at a high quantile
    s_n: float                # ATM total volatility sqrt(w_n(0))
    x_l: float
    x_h: float


def prepare_fit_inputs(slice_data: SliceData, weight_clip_quantile: float) -> SliceFitInputs:
    """Normalise quotes, compute the market "call-equivalent" price via parity,
    Black-76 vega, and bid/ask-spread based weights (paper, Section 7)."""
    q = slice_data.quotes
    forward, t = slice_data.forward, slice_data.time_to_expiry
    x = q["strike"].to_numpy() / forward
    is_call = q["strike"].to_numpy() >= forward
    bid, ask = q["otm_bid"].to_numpy(), q["otm_ask"].to_numpy()
    vol_mid = q["otm_iv"].to_numpy()

    # Put-call parity: c(x) = C(Fx)/F directly for an OTM call quote, or
    # p(x) + 1 - x for an OTM put quote (normalised, undiscounted prices).
    parity_shift = np.where(is_call, 0.0, 1.0 - x)
    c_bid = bid / forward + parity_shift
    c_ask = ask / forward + parity_shift

    # delta_sigma_i: bid/ask implied-vol spread of the slice (paper, Section 7),
    # obtained by inverting the normalised bid/ask prices at each quote's vol.
    vol_bid_arr = np.array([black76.implied_vol(c_bid[i], x[i], t) for i in range(len(x))])
    vol_ask_arr = np.array([black76.implied_vol(c_ask[i], x[i], t) for i in range(len(x))])
    valid = np.isfinite(vol_bid_arr) & np.isfinite(vol_ask_arr) & (vol_mid > 0)
    c_mid = 0.5 * (c_bid + c_ask)

    x, c_mid, c_bid, c_ask = x[valid], c_mid[valid], c_bid[valid], c_ask[valid]
    vol_bid_arr, vol_ask_arr, vol_mid = vol_bid_arr[valid], vol_ask_arr[valid], vol_mid[valid]

    order = np.argsort(x)
    x, c_mid, c_bid, c_ask = x[order], c_mid[order], c_bid[order], c_ask[order]
    vol_bid_arr, vol_ask_arr, vol_mid = vol_bid_arr[order], vol_ask_arr[order], vol_mid[order]

    vega = black76.normalized_vega(x, vol_mid, t)
    delta_vol = np.clip(vol_ask_arr - vol_bid_arr, 1e-6, None)
    raw_weight = 1.0 / (vega * delta_vol) ** 2
    cutoff = np.quantile(raw_weight, weight_clip_quantile)
    weight = np.minimum(raw_weight, cutoff)

    # ATM total volatility s_n = sqrt(w_n(0)): interpolate the mid vol at x=1
    # from the two quotes nearest the forward on each side.
    atm_vol = np.interp(1.0, x, vol_mid)
    s_n = atm_vol * np.sqrt(t)

    return SliceFitInputs(
        expiry_date=slice_data.expiry_date,
        settlement=slice_data.settlement,
        t_expiry=t,
        forward=forward,
        discount_factor=slice_data.discount_factor,
        x=x, c_mkt=c_mid, c_bid=c_bid, c_ask=c_ask,
        vol_bid=vol_bid_arr, vol_ask=vol_ask_arr, vol_mid=vol_mid,
        vega=vega, weight=weight, s_n=float(s_n),
        x_l=float(x.min()), x_h=float(x.max()),
    )


@dataclass
class SliceQPVars:
    """Decision variables and static pieces for one maturity's QP block."""

    fit: SliceFitInputs
    basis: bspline_body.BodyBasis
    put_dict: WingDictionary
    call_dict: WingDictionary
    k_s_put: float
    k_s_call: float
    d: cp.Variable
    a_L: cp.Variable
    a_R: cp.Variable
    R: np.ndarray


def build_slice_vars(
    fit: SliceFitInputs,
    degree: int,
    n_interior: int,
    q: float,
    put_dict: WingDictionary,
    call_dict: WingDictionary,
) -> SliceQPVars:
    """Seams placed at the deepest and shallowest retained quote of the slice
    (paper, Section 5.3): the body domain [x_l, x_h] is exactly the quoted
    range and the two seams are its endpoints."""
    basis = bspline_body.build_body_basis(fit.x_l, fit.x_h, degree, n_interior, q)
    R = bspline_body.smoothness_penalty_matrix(basis, fit.s_n)
    d = cp.Variable(basis.n_coef, name=f"d_{fit.expiry_date}_{fit.settlement}")
    a_L = cp.Variable(put_dict.n_atoms, name=f"aL_{fit.expiry_date}_{fit.settlement}")
    a_R = cp.Variable(call_dict.n_atoms, name=f"aR_{fit.expiry_date}_{fit.settlement}")
    return SliceQPVars(
        fit=fit, basis=basis, put_dict=put_dict, call_dict=call_dict,
        k_s_put=np.log(fit.x_l), k_s_call=np.log(fit.x_h),
        d=d, a_L=a_L, a_R=a_R, R=R,
    )


def body_constraints(vars_: SliceQPVars, convexity_eps: float) -> list:
    """Monotonicity, slope bound, convexity floor and endpoint bounds on the
    body coefficients (paper, Section 4)."""
    basis, d = vars_.basis, vars_.d
    e = basis.D1 @ d
    f = basis.D2 @ d
    cons = [
        e <= 0,                                   # D_1 d <= 0: c is non-increasing
        e >= -1.0,                                # D_1 d >= -1: slope bounded below by -1
        f >= convexity_eps / vars_.fit.s_n,        # D_2 d >= eps/s_n: strict convexity floor
        d[0] >= 1.0 - vars_.fit.x_l,               # left endpoint bound
        d[basis.n_coef - 1] >= 0.0,                # right endpoint bound
    ]
    return cons


def wing_join_constraints(vars_: SliceQPVars) -> list:
    """C^2 join equalities at both seams (paper, eq. 16-18), with intrinsic
    (1-x, i.e. 1-e^k in log-moneyness) subtracted on the put side only, since
    the put-wing atoms represent time value, while the body c(x) includes the
    intrinsic value everywhere (implementation_notes, Section 5.3)."""
    basis = vars_.basis
    cons = []

    c0_put, c1_put, c2_put = basis.seam_quantities(vars_.fit.x_l, vars_.k_s_put)
    x_s_put = vars_.fit.x_l
    intrinsic0, intrinsic1, intrinsic2 = 1.0 - x_s_put, -x_s_put, -x_s_put
    target_put = cp.hstack([
        c0_put @ vars_.d - intrinsic0,
        c1_put @ vars_.d - intrinsic1,
        c2_put @ vars_.d - intrinsic2,
    ])
    H_put = vars_.put_dict.seam_matrix(vars_.k_s_put)
    cons.append(H_put @ vars_.a_L == target_put)

    c0_call, c1_call, c2_call = basis.seam_quantities(vars_.fit.x_h, vars_.k_s_call)
    target_call = cp.hstack([c0_call @ vars_.d, c1_call @ vars_.d, c2_call @ vars_.d])
    H_call = vars_.call_dict.seam_matrix(vars_.k_s_call)
    cons.append(H_call @ vars_.a_R == target_call)

    cons += [vars_.a_L >= 0, vars_.a_R >= 0]
    return cons


def data_fit_term(vars_: SliceQPVars) -> cp.Expression:
    """Weighted least-squares residual of the body against market mid prices
    (paper, Section 7); only quotes inside [x_l, x_h] enter the body fit by
    construction since the seams are the quoted range's endpoints."""
    basis = vars_.basis
    design = basis.design_matrix(vars_.fit.x, nu=0)
    fitted = design @ vars_.d
    resid = fitted - vars_.fit.c_mkt
    n = len(vars_.fit.x)
    return cp.sum(cp.multiply(vars_.fit.weight, cp.square(resid))) / n


def penalty_terms(vars_: SliceQPVars, rho: float, eta: float) -> cp.Expression:
    smooth = rho * cp.quad_form(vars_.d, cp.psd_wrap(vars_.R))
    ridge = eta * (cp.sum_squares(vars_.a_L) + cp.sum_squares(vars_.a_R))
    return smooth + ridge


@dataclass
class FitResult:
    vars: SliceQPVars
    d_value: np.ndarray
    a_L_value: np.ndarray
    a_R_value: np.ndarray
    solve_time: float
    status: str


def solve_single_slice(
    fit: SliceFitInputs,
    degree: int,
    n_interior: int,
    q: float,
    put_dict: WingDictionary,
    call_dict: WingDictionary,
    rho: float,
    eta: float,
    convexity_eps: float,
    solver: str = cp.CLARABEL,
) -> FitResult:
    """Build and solve the single-maturity QP (used directly by exp_2; the
    per-slice pieces here are reused, with calendar constraints added across
    slices, by src/calendar_constraints.py for exp_3)."""
    vars_ = build_slice_vars(fit, degree, n_interior, q, put_dict, call_dict)
    constraints = body_constraints(vars_, convexity_eps) + wing_join_constraints(vars_)
    objective = cp.Minimize(data_fit_term(vars_) + penalty_terms(vars_, rho, eta))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP did not solve to optimality: status={problem.status}")
    return FitResult(
        vars=vars_,
        d_value=vars_.d.value,
        a_L_value=vars_.a_L.value,
        a_R_value=vars_.a_R.value,
        solve_time=problem.solver_stats.solve_time,
        status=problem.status,
    )


def evaluate_fit(result: FitResult, x: np.ndarray) -> np.ndarray:
    """Fitted normalised call price c(x) over body and wing regions."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    vars_ = result.vars
    out = np.empty_like(x)
    in_body = (x >= vars_.fit.x_l) & (x <= vars_.fit.x_h)
    in_put = x < vars_.fit.x_l
    in_call = x > vars_.fit.x_h

    if np.any(in_body):
        design = vars_.basis.design_matrix(x[in_body], nu=0)
        out[in_body] = design @ result.d_value

    if np.any(in_put):
        k = np.log(x[in_put])
        h, _, _ = vars_.put_dict.atom_derivatives(k, vars_.k_s_put)
        v = result.a_L_value @ h
        out[in_put] = (1.0 - x[in_put]) + v

    if np.any(in_call):
        k = np.log(x[in_call])
        h, _, _ = vars_.call_dict.atom_derivatives(k, vars_.k_s_call)
        out[in_call] = result.a_R_value @ h

    return out


def implied_density(result: FitResult, x: np.ndarray) -> np.ndarray:
    """Second x-derivative c''(x), proportional to the Breeden-Litzenberger
    risk-neutral density, over body and wing regions."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    vars_ = result.vars
    out = np.empty_like(x)
    in_body = (x >= vars_.fit.x_l) & (x <= vars_.fit.x_h)
    in_put = x < vars_.fit.x_l
    in_call = x > vars_.fit.x_h

    if np.any(in_body):
        design2 = vars_.basis.design_matrix(x[in_body], nu=2)
        out[in_body] = design2 @ result.d_value

    if np.any(in_put):
        k = np.log(x[in_put])
        _, dh, d2h = vars_.put_dict.atom_derivatives(k, vars_.k_s_put)
        v_k = result.a_L_value @ dh
        v_kk = result.a_L_value @ d2h
        out[in_put] = (v_kk - v_k) / x[in_put] ** 2

    if np.any(in_call):
        k = np.log(x[in_call])
        _, dh, d2h = vars_.call_dict.atom_derivatives(k, vars_.k_s_call)
        v_k = result.a_R_value @ dh
        v_kk = result.a_R_value @ d2h
        out[in_call] = (v_kk - v_k) / x[in_call] ** 2

    return out


def fit_statistics(result: FitResult) -> dict:
    """Percentage of body quotes fitted inside the bid/ask spread and RMS
    error in implied-vol basis points against the mid (paper, Section 8)."""
    fit = result.vars.fit
    fitted_price = evaluate_fit(result, fit.x)
    fitted_vol = np.array(
        [black76.implied_vol(fitted_price[i], fit.x[i], fit.t_expiry) for i in range(len(fit.x))]
    )
    valid = np.isfinite(fitted_vol)
    inside = (fitted_vol[valid] >= fit.vol_bid[valid] - 1e-10) & (fitted_vol[valid] <= fit.vol_ask[valid] + 1e-10)
    rms_vol_bps = float(np.sqrt(np.mean((fitted_vol[valid] - fit.vol_mid[valid]) ** 2)) * 1e4)
    return {
        "n_quotes": int(len(fit.x)),
        "n_inside": int(np.sum(inside)),
        "pct_inside": float(np.mean(inside) * 100.0),
        "rms_vol_bps": rms_vol_bps,
        "fitted_vol": fitted_vol,
    }
