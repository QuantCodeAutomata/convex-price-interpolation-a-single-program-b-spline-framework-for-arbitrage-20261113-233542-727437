"""Full surface calibration as a single convex QP (paper, Section 6; exp_3, exp_4).

Assembles the per-maturity body/wing blocks of src/calibration.py for every
slice, orders maturities by time to expiry, and adds the three families of
calendar (across-maturity) inequalities of src/calendar_constraints.py between
every adjacent pair, so the whole ladder is calibrated by one QP solve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from src import calendar_constraints as cc
from src import calibration
from src.variance_swap import VarianceSwapTerms, build_variance_swap_terms
from src.wing_dictionary import WingDictionary


@dataclass
class CalendarPairDiagnostics:
    n_index: int
    n1_index: int
    n_body_wing_rows: int
    n_wing_wing_rows: int
    used_grid_fallback_put: bool
    used_grid_fallback_call: bool


@dataclass
class SurfaceQP:
    slice_vars: list  # list[calibration.SliceQPVars], ordered by time to expiry
    constraints: list
    objective: cp.Expression
    n_calendar_inequalities: int
    pair_diagnostics: list[CalendarPairDiagnostics]
    varswap_terms: list[VarianceSwapTerms] | None = None


def _one_sided_band_constraint(
    dominant_design_or_h: np.ndarray,
    dominant_var: cp.Variable,
    subordinate_h: np.ndarray,
    subordinate_var: cp.Variable,
    eps: np.ndarray,
) -> cp.Constraint:
    """dominant - subordinate - eps@subordinate_weights >= 0, row-wise (paper,
    Section 6, eq. 19-20: the wing side is only known up to the Hermite
    interpolation error margin eps, so it is conservatively discounted by
    eps @ weights before comparing to the exactly-represented body side)."""
    margin = eps @ subordinate_var if eps.size else 0.0
    return dominant_design_or_h @ dominant_var - subordinate_h.T @ subordinate_var - margin >= 0


def _wing_wing_rows(
    mu: np.ndarray,
    a_n: cp.Variable,
    kappa_n: float,
    a_n1: cp.Variable,
    kappa_n1: float,
    delta_mu: float,
) -> cp.Constraint:
    """Summation-by-parts deep wing-wing calendar condition (paper, Appendix A).

    beta_i^(n) = a_i^(n) * sqrt(kappa_n) * exp(mu_i*kappa_n); with the atoms
    already ordered by decreasing mu (src/wing_dictionary.py), S_m = sum_{i>=m}
    (beta_i^(n+1) - (1+delta_mu)*beta_i^(n)) >= 0 for m=1..N_a is a sufficient
    condition for c_{n+1}(k) - c_n(k) >= 0 for every k beyond both seams,
    normalised by psi_m(kappa_bar) so every pair's rows have comparable scale.
    """
    beta_n = cc.wing_summation_by_parts_beta(a_n, mu, kappa_n)
    beta_n1 = cc.wing_summation_by_parts_beta(a_n1, mu, kappa_n1)
    U = cc.wing_calendar_cumulative_rows(mu)
    kappa_bar = max(kappa_n, kappa_n1)
    psi = cc.wing_calendar_normalisation(mu, kappa_bar)
    S = U @ (beta_n1 - (1.0 + delta_mu) * beta_n)
    return cp.multiply(S, 1.0 / psi) >= 0


def build_surface_qp(
    fits: list,
    degree: int,
    n_interior: int,
    q: float,
    put_dict: WingDictionary,
    call_dict: WingDictionary,
    rho: float,
    eta: float,
    convexity_eps: float,
    subdivisions: int,
    body_wing_error_fraction: float,
    wing_margin_delta_mu: float,
    with_variance_swap: bool = False,
) -> SurfaceQP:
    """Assemble the joint QP: per-slice body/wing blocks plus calendar
    inequalities for every adjacent pair of maturities (paper, Section 6)."""
    order = np.argsort([f.t_expiry for f in fits])
    fits_sorted = [fits[i] for i in order]

    slice_vars = [
        calibration.build_slice_vars(f, degree, n_interior, q, put_dict, call_dict)
        for f in fits_sorted
    ]
    constraints: list = []
    objective_terms = []
    varswap_terms: list[VarianceSwapTerms] = []
    for v in slice_vars:
        constraints += calibration.body_constraints(v, convexity_eps)
        constraints += calibration.wing_join_constraints(v)
        objective_terms.append(calibration.data_fit_term(v) + calibration.penalty_terms(v, rho, eta))
        if with_variance_swap:
            varswap_terms.append(
                build_variance_swap_terms(v.basis, v.put_dict, v.call_dict, v.k_s_put, v.k_s_call)
            )

    n_calendar = 0
    pair_diag = []
    mu_put, mu_call = put_dict.mu, call_dict.mu
    for n in range(len(slice_vars) - 1):
        v_n, v_n1 = slice_vars[n], slice_vars[n + 1]

        overlap = cc.body_body_overlap(v_n.basis, v_n1.basis, subdivisions)
        if overlap is not None:
            constraints.append(overlap.M_n1 @ v_n1.d - overlap.M_n @ v_n.d >= 0)
            n_calendar += overlap.M_n.shape[0]

        n_band_rows = 0
        used_grid_put = used_grid_call = False
        # Put-side body-wing band: whichever slice's body domain reaches
        # further left is "dominant" (body side); the other's put wing fills
        # the gap and is discounted by its Hermite error margin.
        if v_n1.fit.x_l < v_n.fit.x_l - 1e-12:
            band = cc.body_wing_band(
                v_n1.basis, v_n.put_dict, v_n.k_s_put,
                band_lo=v_n1.fit.x_l, band_hi=v_n.fit.x_l,
                error_fraction=body_wing_error_fraction,
            )
            if band is not None:
                constraints.append(
                    _one_sided_band_constraint(band.body_design, v_n1.d, band.wing_h, v_n.a_L, band.eps)
                )
                n_band_rows += band.x_grid.size
        elif v_n.fit.x_l < v_n1.fit.x_l - 1e-12:
            used_grid_put = True
            band = cc.body_wing_band(
                v_n.basis, v_n1.put_dict, v_n1.k_s_put,
                band_lo=v_n.fit.x_l, band_hi=v_n1.fit.x_l,
                error_fraction=body_wing_error_fraction,
            )
            if band is not None:
                constraints.append(
                    _one_sided_band_constraint(band.wing_h, v_n1.a_L, band.body_design, v_n.d, np.zeros(1))
                )
                n_band_rows += band.x_grid.size

        # Call-side body-wing band, symmetric construction.
        if v_n1.fit.x_h > v_n.fit.x_h + 1e-12:
            band = cc.body_wing_band(
                v_n1.basis, v_n.call_dict, v_n.k_s_call,
                band_lo=v_n.fit.x_h, band_hi=v_n1.fit.x_h,
                error_fraction=body_wing_error_fraction,
            )
            if band is not None:
                constraints.append(
                    _one_sided_band_constraint(band.body_design, v_n1.d, band.wing_h, v_n.a_R, band.eps)
                )
                n_band_rows += band.x_grid.size
        elif v_n.fit.x_h > v_n1.fit.x_h + 1e-12:
            used_grid_call = True
            band = cc.body_wing_band(
                v_n.basis, v_n1.call_dict, v_n1.k_s_call,
                band_lo=v_n1.fit.x_h, band_hi=v_n.fit.x_h,
                error_fraction=body_wing_error_fraction,
            )
            if band is not None:
                constraints.append(
                    _one_sided_band_constraint(band.wing_h, v_n1.a_R, band.body_design, v_n.d, np.zeros(1))
                )
                n_band_rows += band.x_grid.size

        # Deep put-wing / put-wing and call-wing / call-wing regions beyond
        # both seams (paper, Appendix A).
        constraints.append(
            _wing_wing_rows(mu_put, v_n.a_L, abs(v_n.k_s_put), v_n1.a_L, abs(v_n1.k_s_put), wing_margin_delta_mu)
        )
        constraints.append(
            _wing_wing_rows(mu_call, v_n.a_R, abs(v_n.k_s_call), v_n1.a_R, abs(v_n1.k_s_call), wing_margin_delta_mu)
        )
        n_calendar += n_band_rows + 2 * len(mu_put)

        pair_diag.append(
            CalendarPairDiagnostics(
                n_index=n, n1_index=n + 1, n_body_wing_rows=n_band_rows,
                n_wing_wing_rows=2 * len(mu_put), used_grid_fallback_put=used_grid_put,
                used_grid_fallback_call=used_grid_call,
            )
        )

    objective = cp.sum(objective_terms)
    return SurfaceQP(
        slice_vars=slice_vars, constraints=constraints, objective=objective,
        n_calendar_inequalities=n_calendar, pair_diagnostics=pair_diag,
        varswap_terms=varswap_terms if with_variance_swap else None,
    )


@dataclass
class SurfaceFitResult:
    slice_vars: list
    d_values: list[np.ndarray]
    a_L_values: list[np.ndarray]
    a_R_values: list[np.ndarray]
    solve_time: float
    status: str
    varswap_values: list[float] | None = None


def solve_surface(qp: SurfaceQP, solver: str = cp.CLARABEL) -> SurfaceFitResult:
    problem = cp.Problem(cp.Minimize(qp.objective), qp.constraints)
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Surface QP did not solve to optimality: status={problem.status}")
    d_values = [v.d.value for v in qp.slice_vars]
    a_L_values = [v.a_L.value for v in qp.slice_vars]
    a_R_values = [v.a_R.value for v in qp.slice_vars]
    varswap_values = None
    if qp.varswap_terms is not None:
        varswap_values = [
            term.value(d, aL, aR)
            for term, d, aL, aR in zip(qp.varswap_terms, d_values, a_L_values, a_R_values)
        ]
    return SurfaceFitResult(
        slice_vars=qp.slice_vars, d_values=d_values, a_L_values=a_L_values, a_R_values=a_R_values,
        solve_time=problem.solver_stats.solve_time, status=problem.status, varswap_values=varswap_values,
    )


def as_fit_result(surface: SurfaceFitResult, index: int) -> calibration.FitResult:
    """Wrap one slice's joint-solve output as a calibration.FitResult, so
    src/calibration.py's evaluate_fit/implied_density/fit_statistics work
    unchanged on surface fits (paper, Section 8 diagnostics)."""
    return calibration.FitResult(
        vars=surface.slice_vars[index],
        d_value=surface.d_values[index],
        a_L_value=surface.a_L_values[index],
        a_R_value=surface.a_R_values[index],
        solve_time=surface.solve_time,
        status=surface.status,
    )


def verify_no_calendar_arbitrage(surface: SurfaceFitResult, n_grid_points: int) -> dict:
    """Dense verification grid spanning one log-moneyness unit past every
    seam (paper, Section 8): checks c_{n+1}(x) - c_n(x) >= 0 directly, with no
    reliance on the sufficient-condition machinery used to impose it."""
    order = np.argsort([v.fit.t_expiry for v in surface.slice_vars])
    worst = 0.0
    n_checked = 0
    violations = []
    for idx in range(len(order) - 1):
        n, n1 = order[idx], order[idx + 1]
        v_n, v_n1 = surface.slice_vars[n], surface.slice_vars[n1]
        lo = min(v_n.k_s_put, v_n1.k_s_put) - 1.0
        hi = max(v_n.k_s_call, v_n1.k_s_call) + 1.0
        k_grid = np.linspace(lo, hi, max(2, n_grid_points // max(1, len(order) - 1)))
        x_grid = np.exp(k_grid)
        res_n = as_fit_result(surface, n)
        res_n1 = as_fit_result(surface, n1)
        c_n = calibration.evaluate_fit(res_n, x_grid)
        c_n1 = calibration.evaluate_fit(res_n1, x_grid)
        diff = c_n1 - c_n
        n_checked += len(diff)
        if diff.min() < worst:
            worst = float(diff.min())
        bad = np.where(diff < -1e-8)[0]
        if len(bad):
            violations.append({"pair": (n, n1), "n_bad": int(len(bad)), "worst": float(diff[bad].min())})
    return {"n_points_checked": n_checked, "worst_violation": worst, "violations": violations}
