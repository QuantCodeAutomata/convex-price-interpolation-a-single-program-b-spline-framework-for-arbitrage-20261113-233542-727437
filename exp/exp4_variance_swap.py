"""Experiment 4: variance swap quotes integrated into the convex program.

Reproduces the paper's demonstration (Section 8, Appendix B) that the fair
variance swap level is an affine functional of the single-maturity QP's
decision variables, that the unquoted put wing supplies a material share of
that level, and that adding variance swap quotes as a least-squares term pins
the put wing without moving the option fit.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import DataLoaderError, SliceData, build_slices
from src import calibration, config, variance_swap
from src.wing_dictionary import build_both_dictionaries

RESULTS_DIR = "results"


def _load_slices() -> list[SliceData]:
    try:
        slices, meta = build_slices()
    except DataLoaderError as exc:
        print(f"ERROR: failed to load the live option chain: {exc}")
        raise SystemExit(1)
    if not slices:
        print("ERROR: live feed returned no usable slices.")
        raise SystemExit(1)
    print(f"Loaded {len(slices)} slices from the live feed (meta: {meta}).")
    return sorted(slices, key=lambda s: s.time_to_expiry)


def _fit_with_varswap(
    fit: calibration.SliceFitInputs,
    put_b_hi: float,
    varswap_quote: float | None,
    varswap_weight: float = 1.0,
):
    """Build and solve one slice's QP (paper, Sections 4,5,7), optionally with
    a least-squares variance-swap term (L_n/L_n^mkt - 1)^2 added to the
    objective, as described for the variance-swap integration in Section 8."""
    put_dict, call_dict = build_both_dictionaries(
        config.N_PUT_ATOMS, config.N_CALL_ATOMS,
        config.PUT_B_LO, put_b_hi,
        config.CALL_B_LO, config.CALL_B_HI,
    )
    vars_ = calibration.build_slice_vars(
        fit, config.BODY_DEGREE_CUBIC, config.N_INTERIOR_KNOTS, config.KNOT_CONCENTRATION_Q,
        put_dict, call_dict,
    )
    constraints = calibration.body_constraints(vars_, config.CONVEXITY_FLOOR_EPSILON)
    constraints += calibration.wing_join_constraints(vars_)
    objective = calibration.data_fit_term(vars_) + calibration.penalty_terms(
        vars_, config.SMOOTHNESS_PENALTY_RHO, config.WING_RIDGE_PENALTY_ETA
    )
    vs_terms = variance_swap.build_variance_swap_terms(
        vars_.basis, put_dict, call_dict, vars_.k_s_put, vars_.k_s_call
    )
    if varswap_quote is not None:
        l_expr = vs_terms.expr(vars_.d, vars_.a_L, vars_.a_R)
        objective = objective + varswap_weight * cp.square(l_expr / varswap_quote - 1.0)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP did not solve to optimality: status={problem.status}")
    result = calibration.FitResult(
        vars=vars_, d_value=vars_.d.value, a_L_value=vars_.a_L.value, a_R_value=vars_.a_R.value,
        solve_time=problem.solver_stats.solve_time, status=problem.status,
    )
    fair_variance = vs_terms.value(result.d_value, result.a_L_value, result.a_R_value)
    put_wing_variance = 2.0 * (vs_terms.g_L @ result.a_L_value)
    return result, vs_terms, fair_variance, put_wing_variance


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    slices = _load_slices()
    weight_q = config.WEIGHT_CLIP_QUANTILE

    # Step 1: reference fit with the thin put dictionary (b_hi=1.0), recording
    # L_n^quote for every maturity -- these synthetic variance swap "quotes"
    # stand in for the market's variance swap quotes (paper, Section 8).
    quote_rows = []
    l_quote = {}
    put_wing_share = {}
    for s in slices:
        fit = calibration.prepare_fit_inputs(s, weight_q)
        if len(fit.x) < 4:
            continue
        result, vs_terms, fair_var, put_wing_var = _fit_with_varswap(
            fit, config.THIN_PUT_DICTIONARY_B_HI, None
        )
        vol = variance_swap.fair_variance_to_volatility(fair_var, fit.t_expiry)
        l_quote[(s.expiry_date, s.settlement)] = fair_var
        share = put_wing_var / fair_var if fair_var > 0 else float("nan")
        put_wing_share[(s.expiry_date, s.settlement)] = share
        quote_rows.append({
            "expiry_date": str(s.expiry_date), "settlement": s.settlement,
            "t_expiry": fit.t_expiry, "fair_variance": fair_var,
            "varswap_vol": vol, "put_wing_share": share,
        })
    quote_df = pd.DataFrame(quote_rows).sort_values("t_expiry")
    quote_df.to_csv(f"{RESULTS_DIR}/exp4_reference_varswap_quotes.csv", index=False)

    shortest = quote_df.iloc[0]
    short_mask = quote_df["t_expiry"] < 21.0 / 365.0
    print(f"Shortest expiry ({shortest['t_expiry']:.4f}y) put-wing share: {shortest['put_wing_share']*100:.1f}%")
    if short_mask.any():
        print(f"Median put-wing share, expiries < 3 weeks: {quote_df.loc[short_mask, 'put_wing_share'].median()*100:.1f}%")
    print(f"Median put-wing share across the ladder: {quote_df['put_wing_share'].median()*100:.1f}%")

    # Step 2: sweep the put-side cap b_hi and track variance-swap volatility
    # and option-fit quality at the shortest expiry and across the ladder.
    sweep_rows = []
    sample_slices = [quote_df.iloc[0]]
    mid = quote_df.iloc[len(quote_df) // 2]
    sample_slices.append(mid)
    sample_keys = {(r["expiry_date"], r["settlement"]) for r in sample_slices}
    for s in slices:
        key = (str(s.expiry_date), s.settlement)
        if key not in sample_keys:
            continue
        fit = calibration.prepare_fit_inputs(s, weight_q)
        if len(fit.x) < 4:
            continue
        for b_hi in config.PUT_B_HI_SWEEP:
            result, vs_terms, fair_var, _ = _fit_with_varswap(fit, b_hi, None)
            stats = calibration.fit_statistics(result)
            vol = variance_swap.fair_variance_to_volatility(fair_var, fit.t_expiry)
            sweep_rows.append({
                "expiry_date": key[0], "settlement": key[1], "t_expiry": fit.t_expiry,
                "put_b_hi": b_hi, "varswap_vol": vol, "pct_inside": stats["pct_inside"],
            })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f"{RESULTS_DIR}/exp4_cap_sensitivity.csv", index=False)
    for key, grp in sweep_df.groupby(["expiry_date", "settlement"]):
        rng = grp["varswap_vol"].max() - grp["varswap_vol"].min()
        print(f"{key}: varswap vol range over b_hi sweep = {rng*100:.2f} vol points, "
              f"pct_inside range = [{grp['pct_inside'].min():.2f}, {grp['pct_inside'].max():.2f}]")

    # Step 3: fit the standard dictionary (b_hi=1.6) with and without the
    # recorded variance swap quotes added as a least-squares term, and compare.
    compare_rows = []
    fattest_put_weight = {"without": [], "with": []}
    for s in slices:
        key = (s.expiry_date, s.settlement)
        if key not in l_quote:
            continue
        fit = calibration.prepare_fit_inputs(s, weight_q)
        if len(fit.x) < 4:
            continue
        res_wo, vs_wo, fv_wo, _ = _fit_with_varswap(fit, config.PUT_B_HI, None)
        res_w, vs_w, fv_w, _ = _fit_with_varswap(fit, config.PUT_B_HI, l_quote[key])
        mkt = l_quote[key]
        vol_wo = variance_swap.fair_variance_to_volatility(fv_wo, fit.t_expiry)
        vol_w = variance_swap.fair_variance_to_volatility(fv_w, fit.t_expiry)
        vol_mkt = variance_swap.fair_variance_to_volatility(mkt, fit.t_expiry)
        stats_wo = calibration.fit_statistics(res_wo)
        stats_w = calibration.fit_statistics(res_w)
        compare_rows.append({
            "expiry_date": str(s.expiry_date), "settlement": s.settlement, "t_expiry": fit.t_expiry,
            "diff_bps_without": (vol_wo - vol_mkt) * 1e4, "diff_bps_with": (vol_w - vol_mkt) * 1e4,
            "pct_inside_without": stats_wo["pct_inside"], "pct_inside_with": stats_w["pct_inside"],
            "rms_vol_bps_without": stats_wo["rms_vol_bps"], "rms_vol_bps_with": stats_w["rms_vol_bps"],
            "fattest_put_weight_without": float(res_wo.a_L_value[-1]),
            "fattest_put_weight_with": float(res_w.a_L_value[-1]),
        })
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(f"{RESULTS_DIR}/exp4_varswap_quote_fit.csv", index=False)

    print("\nWithout variance swap quotes: median |diff| = "
          f"{compare_df['diff_bps_without'].abs().median():.1f} bps, "
          f"max = {compare_df['diff_bps_without'].abs().max():.1f} bps")
    print("With variance swap quotes: median |diff| = "
          f"{compare_df['diff_bps_with'].abs().median():.1f} bps, "
          f"max = {compare_df['diff_bps_with'].abs().max():.1f} bps")
    print(f"Option fit without: median pct_inside={compare_df['pct_inside_without'].median():.2f}, "
          f"median rms={compare_df['rms_vol_bps_without'].median():.2f} bps")
    print(f"Option fit with: median pct_inside={compare_df['pct_inside_with'].median():.2f}, "
          f"median rms={compare_df['rms_vol_bps_with'].median():.2f} bps")
    print(f"Median fattest-put-atom weight without: {compare_df['fattest_put_weight_without'].median():.3e}")
    print(f"Median fattest-put-atom weight with: {compare_df['fattest_put_weight_with'].median():.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(quote_df["t_expiry"], quote_df["put_wing_share"] * 100, "o-")
    axes[0].set_xlabel("Time to expiry (years)")
    axes[0].set_ylabel("Put-wing share of fair variance (%)")
    axes[0].set_title("Unquoted put-wing contribution to variance swap")
    axes[1].plot(compare_df["t_expiry"], compare_df["diff_bps_without"], "o-", label="without varswap quotes")
    axes[1].plot(compare_df["t_expiry"], compare_df["diff_bps_with"], "s-", label="with varswap quotes")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Time to expiry (years)")
    axes[1].set_ylabel("Fitted - quoted varswap vol (bps)")
    axes[1].set_title("Variance swap fit, with vs without quotes")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_variance_swap.png", dpi=150)
    plt.close(fig)
    print("Saved results/exp4_*.csv and results/exp4_variance_swap.png")


if __name__ == "__main__":
    main()
