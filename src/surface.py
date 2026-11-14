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
    calendar_slacks: list = field(default_factory=list)


def _band_calendar_constraint(
    n1_term: np.ndarray,
    n1_var: cp.Variable,
    n_term: np.ndarray,
    n_var: cp.Variable,
    eps: np.ndarray,
    margin_on: str,
    slack: cp.Variable | None = None,
) -> cp.Constraint:
    """c_{n+1} - c_n - margin >= 0, row-wise, on a band where one of the two
    maturities is represented by its exact body spline and the other by its
    wing dictionary (paper, Section 6, eq. 19-20). Whichever side is the wing
    is only known up to the Hermite interpolation error bound eps per atom,
    so its contribution is conservatively discounted by eps @ wing_weights
    before comparing the two sides (``margin_on`` names which operand, "n" or
    "n1", carries the wing and hence the margin)."""
    # The wing side's design array is stored as (n_atoms, n_grid) (one row per
    # atom); the body side's design matrix is already (n_grid, n_coef). Only
    # the operand named by ``margin_on`` is the wing, so only that one needs
    # transposing before the grid-row matrix product.
    if margin_on == "n1":
        margin = eps @ n1_var if eps.size else 0.0
        n1_term = n1_term.T
    elif margin_on == "n":
        margin = eps @ n_var if eps.size else 0.0
        n_term = n_term.T
    else:
        raise ValueError(f"margin_on must be 'n' or 'n1', got {margin_on!r}")
    lhs = n1_term @ n1_var - n_term @ n_var - margin
    if slack is not None:
        lhs = lhs + slack
    return lhs >= 0


def _wing_wing_rows(
    mu: np.ndarray,
    a_n: cp.Variable,
    kappa_n: float,
    a_n1: cp.Variable,
    kappa_n1: float,
    delta_mu: float,
    slack: cp.Variable | None = None,
) -> cp.Constraint:
    """Summation-by-parts deep wing-wing calendar condition (paper, Appendix A).

    beta_i^(n) = a_i^(n) * sqrt(kappa_n) * exp(mu_i*kappa_n); with the atoms
    already ordered by decreasing mu (src/wing_dictionary.py), S_m = sum_{i>=m}
    (beta_i^(n+1) - (1+delta_mu)*beta_i^(n)) >= 0 for m=1..N_a is a sufficient
    condition for c_{n+1}(k) - c_n(k) >= 0 for every k beyond both seams,
    normalised by psi_m(kappa_bar) = kappa_bar^(-1/2)*exp(-mu_m*kappa_bar) so
    every pair's rows have comparable scale.

    Each row's coefficient on a_i is folded directly into
    exp(mu_i*kappa_x - mu_m*kappa_bar) instead of forming the raw beta_i and
    psi_m factors separately and multiplying them afterward: atoms are
    ordered by decreasing mu, the summation only touches i >= m (so
    mu_i <= mu_m), and kappa_bar = max(kappa_n, kappa_n1), so every exponent
    here is <= 0 and the row stays well-conditioned. Forming exp(mu_i*kappa_x)
    on its own first overflows for the deepest atoms (mu up to ~100) on the
    widest seams (kappa up to ~3 for the longest maturity), even though the
    product with psi_m is mathematically O(1).
    """
    n = len(mu)
    kappa_bar = max(kappa_n, kappa_n1)
    i_ge_m = (np.arange(n)[None, :] >= np.arange(n)[:, None]).astype(float)
    mu_col, mu_row = mu[None, :], mu[:, None]

    def _weights(kappa: float) -> np.ndarray:
        return i_ge_m * np.sqrt(kappa / kappa_bar) * np.exp(mu_col * kappa - mu_row * kappa_bar)

    lhs = _weights(kappa_n1) @ a_n1 - (1.0 + delta_mu) * (_weights(kappa_n) @ a_n)
    if slack is not None:
        lhs = lhs + slack
    return lhs >= 0


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
    body_body_calendar_mode: str = "merged_knot",
    slack_penalty: float = 0.0,
) -> SurfaceQP:
    """Assemble the joint QP: per-slice body/wing blocks plus calendar
    inequalities for every adjacent pair of maturities (paper, Section 6).

    ``slack_penalty`` > 0 adds a non-negative, quadratically penalised slack
    to every calendar inequality (see src/config.py, CALENDAR_SLACK_PENALTY):
    the paper's own curated snapshot needs none, but a live chain can carry a
    handful of microscopic raw calendar crossings from asynchronous bid/ask
    quotes on near-worthless deep-OTM contracts, which would otherwise make
    the single joint QP infeasible.
    """
    order = np.argsort([f.t_expiry for f in fits])
    fits_sorted = [fits[i] for i in order]

    slice_vars = [
        calibration.build_slice_vars(f, degree, n_interior, q, put_dict, call_dict)
        for f in fits_sorted
    ]
    constraints: list = []
    objective_terms = []
    varswap_terms: list[VarianceSwapTerms] = []
    calendar_slacks: list = []

    def _slack(n_rows: int) -> cp.Variable | None:
        if slack_penalty <= 0:
            return None
        s = cp.Variable(n_rows, nonneg=True)
        calendar_slacks.append(s)
        objective_terms.append(slack_penalty * cp.sum_squares(s))
        return s
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
            if body_body_calendar_mode == "merged_knot":
                s = _slack(overlap.M_n.shape[0])
                lhs = overlap.M_n1 @ v_n1.d - overlap.M_n @ v_n.d
                constraints.append((lhs + s if s is not None else lhs) >= 0)
                n_calendar += overlap.M_n.shape[0]
            elif body_body_calendar_mode == "grid":
                # Same inequality count as the merged-knot condition above, but
                # plain pointwise evaluation with no coefficient merge and no
                # safety margin (paper, Section 8, "grid-based calendar
                # imposition" baseline used only for comparison in exp_3).
                grid_rows = cc.body_body_calendar_grid_rows(v_n.basis, v_n1.basis, overlap.M_n.shape[0])
                if grid_rows is not None:
                    design_n, design_n1 = grid_rows
                    s = _slack(design_n.shape[0])
                    lhs = design_n1 @ v_n1.d - design_n @ v_n.d
                    constraints.append((lhs + s if s is not None else lhs) >= 0)
                    n_calendar += design_n.shape[0]
            else:
                raise ValueError(f"unknown body_body_calendar_mode={body_body_calendar_mode!r}")

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
                    _band_calendar_constraint(
                        band.body_design, v_n1.d, band.wing_h, v_n.a_L, band.eps,
                        margin_on="n", slack=_slack(band.x_grid.size),
                    )
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
                    _band_calendar_constraint(
                        band.wing_h, v_n1.a_L, band.body_design, v_n.d, band.eps,
                        margin_on="n1", slack=_slack(band.x_grid.size),
                    )
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
                    _band_calendar_constraint(
                        band.body_design, v_n1.d, band.wing_h, v_n.a_R, band.eps,
                        margin_on="n", slack=_slack(band.x_grid.size),
                    )
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
                    _band_calendar_constraint(
                        band.wing_h, v_n1.a_R, band.body_design, v_n.d, band.eps,
                        margin_on="n1", slack=_slack(band.x_grid.size),
                    )
                )
                n_band_rows += band.x_grid.size

        # Deep put-wing / put-wing and call-wing / call-wing regions beyond
        # both seams (paper, Appendix A).
        constraints.append(
            _wing_wing_rows(
                mu_put, v_n.a_L, abs(v_n.k_s_put), v_n1.a_L, abs(v_n1.k_s_put),
                wing_margin_delta_mu, slack=_slack(len(mu_put)),
            )
        )
        constraints.append(
            _wing_wing_rows(
                mu_call, v_n.a_R, abs(v_n.k_s_call), v_n1.a_R, abs(v_n1.k_s_call),
                wing_margin_delta_mu, slack=_slack(len(mu_call)),
            )
        )
        n_wing_wing = len(mu_put) + len(mu_call)
        n_calendar += n_band_rows + n_wing_wing

        pair_diag.append(
            CalendarPairDiagnostics(
                n_index=n, n1_index=n + 1, n_body_wing_rows=n_band_rows,
                n_wing_wing_rows=n_wing_wing, used_grid_fallback_put=used_grid_put,
                used_grid_fallback_call=used_grid_call,
            )
        )

    objective = cp.sum(objective_terms)
    return SurfaceQP(
        slice_vars=slice_vars, constraints=constraints, objective=objective,
        n_calendar_inequalities=n_calendar, pair_diagnostics=pair_diag,
        varswap_terms=varswap_terms if with_variance_swap else None,
        calendar_slacks=calendar_slacks,
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
    # The roughness-penalty block (src/bspline_body.smoothness_penalty_matrix) has a
    # wide dynamic range because knot spacing varies by orders of magnitude between
    # the near-the-money and wing regions of a slice; Clarabel's default Ruiz
    # equilibration needs more iterations and a looser gap tolerance to certify
    # optimality on this scale than on a well-scaled textbook QP.
    try:
        problem.solve(solver=solver, max_iter=2000, tol_gap_abs=1e-5, tol_gap_rel=1e-6,
                      tol_feas=1e-6, time_limit=45.0)
    except cp.error.SolverError:
        pass
    if problem.status not in ("optimal", "optimal_inaccurate"):
        # Clarabel can stall at "InsufficientProgress" on this wide-dynamic-range
        # roughness-penalty block (see src/bspline_body.smoothness_penalty_matrix)
        # while the duality gap is already small, or fail outright on a large
        # joint program; SCS's diagonal scaling handles the conditioning at the
        # cost of more iterations.
        try:
            problem.solve(solver=cp.SCS, max_iters=10000, eps=1e-4)
        except cp.error.SolverError:
            pass
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
