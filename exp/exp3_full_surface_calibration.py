"""Experiment 3: full surface calibration as a single convex QP.

Calibrates every retained maturity slice jointly, imposing body no-arbitrage,
wing C^2 joins, and the three calendar (across-maturity) conditions of
paper Section 6 simultaneously, then verifies the result on a dense grid and
runs the robustness checks requested for this experiment (paper, Section 8):
merged-knot subdivision count, and a grid-based calendar baseline that omits
the coefficient-level sufficient condition.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import DataLoaderError, build_slices
from src import calibration, config, surface
from src.wing_dictionary import build_both_dictionaries

RESULTS_DIR = "results"


def _load_fits(full: bool = False):
    try:
        slices, meta = build_slices()
    except DataLoaderError as exc:
        print(f"ERROR: failed to load the live option chain: {exc}")
        raise SystemExit(1)
    if not slices:
        print("ERROR: live feed returned no usable slices.")
        raise SystemExit(1)
    print(f"Loaded {len(slices)} slices on {meta['n_expiry_dates']} expiry dates, "
          f"{meta['n_quotes']} quotes (maturities "
          f"{min(s.time_to_expiry for s in slices):.4f}-{max(s.time_to_expiry for s in slices):.4f}y)")
    fits = [calibration.prepare_fit_inputs(s, config.WEIGHT_CLIP_QUANTILE) for s in slices]
    fits = [f for f in fits if len(f.x) >= config.MIN_QUOTES_STRADDLING_FORWARD]
    fits = sorted(fits, key=lambda f: f.t_expiry)
    n_target = config.N_MATURITIES_PAPER if full else config.N_MATURITIES_RUN
    if len(fits) > n_target:
        # Evenly spaced maturity subset spanning the full short-to-long range,
        # always keeping any AM/PM settlement pair found on the same expiry
        # date so that check still exercises the real data (see
        # results/RESULTS.md, "Deviations from the paper").
        idx = np.unique(np.round(np.linspace(0, len(fits) - 1, n_target)).astype(int))
        kept = set(idx.tolist())
        by_date: dict = {}
        for i, f in enumerate(fits):
            by_date.setdefault(f.expiry_date, []).append(i)
        for date, members in by_date.items():
            if len(members) > 1 and any(m in kept for m in members):
                kept.update(members)
        fits = [fits[i] for i in sorted(kept)]
        print(f"  using a {len(fits)}-maturity subset for this run "
              f"(paper configuration: {config.N_MATURITIES_PAPER} maturities, see results/RESULTS.md)")
    return fits, meta


def _aggregate_fit_stats(result: surface.SurfaceFitResult) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(result.slice_vars):
        fr = surface.as_fit_result(result, i)
        st = calibration.fit_statistics(fr)
        rows.append({
            "expiry_date": v.fit.expiry_date, "settlement": v.fit.settlement,
            "t_expiry": v.fit.t_expiry, "n_quotes": st["n_quotes"], "n_inside": st["n_inside"],
            "pct_inside": st["pct_inside"], "rms_vol_bps": st["rms_vol_bps"],
            "miss_bps": float(np.max(np.abs(st["fitted_vol"] - np.clip(
                st["fitted_vol"], v.fit.vol_bid, v.fit.vol_ask)) * 1e4)) if st["n_inside"] < st["n_quotes"] else 0.0,
        })
    return pd.DataFrame(rows)


def _am_pm_check(fits) -> pd.DataFrame:
    """Same-date AM/PM pairs must have the PM slice's total variance at or
    above the AM slice's at every common strike (paper, Section 8)."""
    by_date: dict = {}
    for f in fits:
        by_date.setdefault(f.expiry_date, {})[f.settlement] = f
    rows = []
    for date, pair in by_date.items():
        if "AM" in pair and "PM" in pair:
            am, pm = pair["AM"], pair["PM"]
            x_grid = np.linspace(max(am.x_l, pm.x_l), min(am.x_h, pm.x_h), 200)
            w_am = am.vol_mid.mean() ** 2 * am.t_expiry  # coarse proxy ahead of the joint solve
            w_pm = pm.vol_mid.mean() ** 2 * pm.t_expiry
            rows.append({"expiry_date": str(date), "t_am": am.t_expiry, "t_pm": pm.t_expiry,
                         "mean_total_var_am": w_am, "mean_total_var_pm": w_pm, "pm_ge_am": bool(w_pm >= w_am)})
    return pd.DataFrame(rows)


def _am_pm_check_fitted(result: surface.SurfaceFitResult) -> pd.DataFrame:
    by_date: dict = {}
    for i, v in enumerate(result.slice_vars):
        by_date.setdefault(v.fit.expiry_date, {})[v.fit.settlement] = i
    rows = []
    for date, pair in by_date.items():
        if "AM" in pair and "PM" in pair:
            i_am, i_pm = pair["AM"], pair["PM"]
            v_am, v_pm = result.slice_vars[i_am], result.slice_vars[i_pm]
            lo = max(v_am.fit.x_l, v_pm.fit.x_l)
            hi = min(v_am.fit.x_h, v_pm.fit.x_h)
            if hi <= lo:
                continue
            x_grid = np.linspace(lo, hi, 200)
            fr_am, fr_pm = surface.as_fit_result(result, i_am), surface.as_fit_result(result, i_pm)
            c_am = calibration.evaluate_fit(fr_am, x_grid)
            c_pm = calibration.evaluate_fit(fr_pm, x_grid)
            import src.black76 as black76
            vol_am = np.array([black76.implied_vol(c_am[j], x_grid[j], v_am.fit.t_expiry) for j in range(len(x_grid))])
            vol_pm = np.array([black76.implied_vol(c_pm[j], x_grid[j], v_pm.fit.t_expiry) for j in range(len(x_grid))])
            w_am = vol_am ** 2 * v_am.fit.t_expiry
            w_pm = vol_pm ** 2 * v_pm.fit.t_expiry
            ok = bool(np.all(w_pm >= w_am - 1e-10))
            rows.append({"expiry_date": str(date), "t_am": v_am.fit.t_expiry, "t_pm": v_pm.fit.t_expiry,
                         "n_points": len(x_grid), "worst_margin": float(np.min(w_pm - w_am)),
                         "pm_ge_am_everywhere": ok})
    return pd.DataFrame(rows)


def _wing_variance_linearity(result: surface.SurfaceFitResult) -> dict:
    """Departure of the wing's total variance from the best straight line in
    k over 1-3 ATM-vol standard deviations past each seam (paper, Section 8,
    "wing variance approximately linear")."""
    import src.black76 as black76
    departures = {1: [], 2: [], 3: []}
    for i, v in enumerate(result.slice_vars):
        fr = surface.as_fit_result(result, i)
        s_n = v.fit.s_n
        for side, k_s, sign in (("put", v.k_s_put, -1.0), ("call", v.k_s_call, 1.0)):
            offsets = np.array([1.0, 2.0, 3.0]) * max(s_n, 1e-6)
            k_vals = k_s + sign * offsets
            x_vals = np.exp(k_vals)
            x_vals = x_vals[(x_vals > 0) & np.isfinite(x_vals)]
            if len(x_vals) < 3:
                continue
            prices = calibration.evaluate_fit(fr, x_vals)
            vols = np.array([black76.implied_vol(prices[j], x_vals[j], v.fit.t_expiry) for j in range(len(x_vals))])
            if not np.all(np.isfinite(vols)):
                continue
            w = vols ** 2 * v.fit.t_expiry
            k_used = np.log(x_vals)
            slope, intercept = np.polyfit(k_used, w, 1)
            line = slope * k_used + intercept
            rel = np.abs(w - line) / np.abs(line)
            for n_std, r in zip((1, 2, 3), rel):
                departures[n_std].append(r)
    out = {}
    for n_std, vals in departures.items():
        vals = np.array(vals)
        out[f"median_pct_{n_std}std"] = float(np.median(vals) * 100) if len(vals) else float("nan")
        out[f"p90_pct_{n_std}std"] = float(np.quantile(vals, 0.9) * 100) if len(vals) else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full maturity ladder instead of the reduced run subset")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fits, meta = _load_fits(full=args.full)
    put_dict, call_dict = build_both_dictionaries(
        config.N_PUT_ATOMS, config.N_CALL_ATOMS, config.PUT_B_LO, config.PUT_B_HI, config.CALL_B_LO, config.CALL_B_HI
    )

    print(f"Building joint QP for {len(fits)} slices ...")
    t0 = time.time()
    qp = surface.build_surface_qp(
        fits, config.BODY_DEGREE_CUBIC, config.N_INTERIOR_KNOTS, config.KNOT_CONCENTRATION_Q,
        put_dict, call_dict, config.SMOOTHNESS_PENALTY_RHO, config.WING_RIDGE_PENALTY_ETA,
        config.CONVEXITY_FLOOR_EPSILON, config.MERGED_KNOT_SUBDIVISIONS,
        config.BODY_WING_BAND_ERROR_FRACTION, config.WING_CALENDAR_MARGIN_DELTA_MU,
        slack_penalty=config.CALENDAR_SLACK_PENALTY,
    )
    build_time = time.time() - t0
    print(f"  build time {build_time:.2f}s, {qp.n_calendar_inequalities} calendar inequalities")

    t0 = time.time()
    result = surface.solve_surface(qp)
    solve_time = time.time() - t0
    print(f"  solve time {solve_time:.2f}s, status={result.status}")

    stats_df = _aggregate_fit_stats(result)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "exp3_fit_stats.csv"), index=False)
    total_quotes = int(stats_df["n_quotes"].sum())
    total_inside = int(stats_df["n_inside"].sum())
    pct_inside_overall = 100.0 * total_inside / total_quotes
    n_misses = total_quotes - total_inside
    miss_bps_values = stats_df.loc[stats_df["n_inside"] < stats_df["n_quotes"], "miss_bps"]
    median_miss_bps = float(miss_bps_values.median()) if len(miss_bps_values) else 0.0
    print(f"  {pct_inside_overall:.2f}% of {total_quotes} fitted prices inside spread, "
          f"{n_misses} misses, median miss {median_miss_bps:.2f} bps")

    check = surface.verify_no_calendar_arbitrage(result, config.CALENDAR_VERIFICATION_GRID_POINTS)
    print(f"  calendar check on {check['n_points_checked']} grid points: "
          f"worst violation {check['worst_violation']:.2e}, {len(check['violations'])} pairs with violations")

    shortest_idx = int(stats_df["t_expiry"].idxmin())
    shortest_row = stats_df.loc[shortest_idx]
    print(f"  shortest slice T={shortest_row['t_expiry']:.5f}y: "
          f"{shortest_row['n_inside']}/{shortest_row['n_quotes']} quotes inside spread")

    am_pm_df = _am_pm_check_fitted(result)
    am_pm_df.to_csv(os.path.join(RESULTS_DIR, "exp3_am_pm_check.csv"), index=False)
    n_am_pm_ok = int(am_pm_df["pm_ge_am_everywhere"].sum()) if len(am_pm_df) else 0
    print(f"  AM/PM ordering: {n_am_pm_ok}/{len(am_pm_df)} pairs with PM total variance >= AM everywhere")

    linearity = _wing_variance_linearity(result)
    with open(os.path.join(RESULTS_DIR, "exp3_wing_linearity.json"), "w") as fh:
        json.dump(linearity, fh, indent=2)
    print(f"  wing variance linearity: {linearity}")

    # --- Sensitivity: 3 vs 4 merged-knot subdivisions -----------------------
    print("Re-solving with 3 merged-knot subdivisions for sensitivity ...")
    qp3 = surface.build_surface_qp(
        fits, config.BODY_DEGREE_CUBIC, config.N_INTERIOR_KNOTS, config.KNOT_CONCENTRATION_Q,
        put_dict, call_dict, config.SMOOTHNESS_PENALTY_RHO, config.WING_RIDGE_PENALTY_ETA,
        config.CONVEXITY_FLOOR_EPSILON, config.MERGED_KNOT_SUBDIVISIONS_SENSITIVITY,
        config.BODY_WING_BAND_ERROR_FRACTION, config.WING_CALENDAR_MARGIN_DELTA_MU,
        slack_penalty=config.CALENDAR_SLACK_PENALTY,
    )
    result3 = surface.solve_surface(qp3)
    moves_bps, moves_frac_spread = [], []
    import src.black76 as black76
    for i, (v4, v3) in enumerate(zip(result.slice_vars, result3.slice_vars)):
        fr4, fr3 = surface.as_fit_result(result, i), surface.as_fit_result(result3, i)
        p4 = calibration.evaluate_fit(fr4, v4.fit.x)
        p3 = calibration.evaluate_fit(fr3, v4.fit.x)
        vol4 = np.array([black76.implied_vol(p4[j], v4.fit.x[j], v4.fit.t_expiry) for j in range(len(v4.fit.x))])
        vol3 = np.array([black76.implied_vol(p3[j], v4.fit.x[j], v4.fit.t_expiry) for j in range(len(v4.fit.x))])
        ok = np.isfinite(vol4) & np.isfinite(vol3)
        diff_bps = np.abs(vol4[ok] - vol3[ok]) * 1e4
        spread_bps = np.clip((v4.fit.vol_ask[ok] - v4.fit.vol_bid[ok]) * 1e4, 1e-6, None)
        moves_bps.extend(diff_bps.tolist())
        moves_frac_spread.extend((diff_bps / spread_bps).tolist())
    moves_bps, moves_frac_spread = np.array(moves_bps), np.array(moves_frac_spread)
    sensitivity = {
        "max_move_bps": float(np.max(moves_bps)) if len(moves_bps) else float("nan"),
        "pct_moving_more_than_5pct_spread": float(np.mean(moves_frac_spread > 0.05) * 100) if len(moves_frac_spread) else float("nan"),
        "pct_moving_less_than_0p7bps": float(np.mean(moves_bps < 0.7) * 100) if len(moves_bps) else float("nan"),
    }
    with open(os.path.join(RESULTS_DIR, "exp3_subdivision_sensitivity.json"), "w") as fh:
        json.dump(sensitivity, fh, indent=2)
    print(f"  subdivision sensitivity (4 vs 3): {sensitivity}")

    # --- Grid-based calendar baseline (comparison only) ---------------------
    print("Re-solving with the grid-based calendar baseline for comparison ...")
    qp_grid = surface.build_surface_qp(
        fits, config.BODY_DEGREE_CUBIC, config.N_INTERIOR_KNOTS, config.KNOT_CONCENTRATION_Q,
        put_dict, call_dict, config.SMOOTHNESS_PENALTY_RHO, config.WING_RIDGE_PENALTY_ETA,
        config.CONVEXITY_FLOOR_EPSILON, config.MERGED_KNOT_SUBDIVISIONS,
        config.BODY_WING_BAND_ERROR_FRACTION, config.WING_CALENDAR_MARGIN_DELTA_MU,
        body_body_calendar_mode="grid", slack_penalty=config.CALENDAR_SLACK_PENALTY,
    )
    result_grid = surface.solve_surface(qp_grid)
    check_grid = surface.verify_no_calendar_arbitrage(result_grid, config.CALENDAR_VERIFICATION_GRID_POINTS)
    print(f"  grid-based baseline: worst violation {check_grid['worst_violation']:.2e}, "
          f"{len(check_grid['violations'])} pairs with violations")

    summary = {
        "n_slices": len(fits), "n_expiry_dates": meta["n_expiry_dates"], "n_quotes_total": total_quotes,
        "build_time_s": build_time, "solve_time_s": solve_time, "status": result.status,
        "n_calendar_inequalities": qp.n_calendar_inequalities,
        "pct_inside_overall": pct_inside_overall, "n_misses": n_misses, "median_miss_bps": median_miss_bps,
        "calendar_worst_violation_merged_knot": check["worst_violation"],
        "calendar_n_violating_pairs_merged_knot": len(check["violations"]),
        "calendar_worst_violation_grid_baseline": check_grid["worst_violation"],
        "calendar_n_violating_pairs_grid_baseline": len(check_grid["violations"]),
        "shortest_slice_t_expiry": float(shortest_row["t_expiry"]),
        "shortest_slice_pct_inside": float(shortest_row["pct_inside"]),
        "am_pm_pairs_total": len(am_pm_df), "am_pm_pairs_ordered_correctly": n_am_pm_ok,
        **{f"wing_linearity_{k}": v for k, v in linearity.items()},
        **{f"subdivision_{k}": v for k, v in sensitivity.items()},
    }
    with open(os.path.join(RESULTS_DIR, "exp3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, v in enumerate(result.slice_vars):
        fr = surface.as_fit_result(result, i)
        k_grid = np.linspace(v.k_s_put - 0.3, v.k_s_call + 0.3, 200)
        x_grid = np.exp(k_grid)
        prices = calibration.evaluate_fit(fr, x_grid)
        vols = np.array([black76.implied_vol(prices[j], x_grid[j], v.fit.t_expiry) for j in range(len(x_grid))])
        ax.plot(k_grid, vols, color=plt.cm.viridis(i / max(1, len(result.slice_vars) - 1)), lw=0.8)
    ax.set_xlabel("log-moneyness k = log(K/F)")
    ax.set_ylabel("fitted implied volatility")
    ax.set_title(f"Full surface calibration: {len(fits)} maturities (paper, Section 8)")
    plot_path = os.path.join(RESULTS_DIR, "exp3_surface.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Saved:")
    for name in ("exp3_fit_stats.csv", "exp3_am_pm_check.csv", "exp3_wing_linearity.json",
                 "exp3_subdivision_sensitivity.json", "exp3_summary.json", "exp3_surface.png"):
        print(f"  {os.path.join(RESULTS_DIR, name)}")
    print("Done.")


if __name__ == "__main__":
    main()
