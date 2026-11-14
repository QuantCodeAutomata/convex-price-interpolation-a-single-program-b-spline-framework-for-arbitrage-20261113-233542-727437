"""Experiment 2: single-maturity B-spline fit, cubic vs quartic body.

Replicates the paper's comparison of a cubic (degree 3) and a quartic
(degree 4) B-spline body fitted to the same single-maturity option slice,
sharing identical wing dictionaries, knots, penalties and convexity floor.
Only the body degree differs between the two fits (paper, Section 8).
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
from src import black76, calibration, config
from src.wing_dictionary import build_both_dictionaries

RESULTS_DIR = "results"
TARGET_T = 0.49  # paper's example maturity, in years


def _load_slices() -> list[SliceData]:
    """Fetch the live Cboe chain exactly once and build the filtered slices."""
    try:
        slices, meta = build_slices()
    except DataLoaderError as exc:
        print(f"ERROR: failed to load the live option chain: {exc}")
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 - any other failure is fatal here
        print(f"ERROR: unexpected failure while loading the live option chain: {exc}")
        raise SystemExit(1)
    if not slices:
        print("ERROR: live feed returned no usable slices.")
        raise SystemExit(1)
    print(f"Loaded {len(slices)} slices from the live feed (meta: {meta}).")
    return slices


def _pick_slice(slices: list[SliceData]) -> SliceData:
    """Pick the slice closest to TARGET_T years, preferring PM settlement."""
    pm_slices = [s for s in slices if s.settlement == "PM"]
    pool = pm_slices if pm_slices else slices
    chosen = min(pool, key=lambda s: abs(s.time_to_expiry - TARGET_T))
    print(
        f"Picked slice: expiry={chosen.expiry_date}, settlement={chosen.settlement}, "
        f"T={chosen.time_to_expiry:.4f}y, forward={chosen.forward:.2f}, "
        f"n_quotes={len(chosen.quotes)}"
    )
    return chosen


def _iv_curve(result: calibration.FitResult, x_grid: np.ndarray, t: float) -> np.ndarray:
    """Evaluate the fitted price on x_grid and convert pointwise to implied vol."""
    prices = calibration.evaluate_fit(result, x_grid)
    return np.array([black76.implied_vol(prices[i], x_grid[i], t) for i in range(len(x_grid))])


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    slices = _load_slices()
    chosen = _pick_slice(slices)
    fit = calibration.prepare_fit_inputs(chosen, weight_clip_quantile=config.WEIGHT_CLIP_QUANTILE)

    put_dict, call_dict = build_both_dictionaries(
        config.N_PUT_ATOMS, config.N_CALL_ATOMS,
        config.PUT_B_LO, config.PUT_B_HI,
        config.CALL_B_LO, config.CALL_B_HI,
    )

    common_kwargs = dict(
        n_interior=config.N_INTERIOR_KNOTS,
        q=config.KNOT_CONCENTRATION_Q,
        rho=config.SMOOTHNESS_PENALTY_RHO,
        eta=config.WING_RIDGE_PENALTY_ETA,
        convexity_eps=config.CONVEXITY_FLOOR_EPSILON,
    )

    result_cubic = calibration.solve_single_slice(
        fit, config.BODY_DEGREE_CUBIC, put_dict=put_dict, call_dict=call_dict,
        solver=cp.CLARABEL, **common_kwargs,
    )
    result_quartic = calibration.solve_single_slice(
        fit, config.BODY_DEGREE_QUARTIC, put_dict=put_dict, call_dict=call_dict,
        solver=cp.CLARABEL, **common_kwargs,
    )

    stats_cubic = calibration.fit_statistics(result_cubic)
    stats_quartic = calibration.fit_statistics(result_quartic)

    for degree, stats in ((3, stats_cubic), (4, stats_quartic)):
        print(
            f"degree={degree}: n_quotes={stats['n_quotes']}, n_inside={stats['n_inside']}, "
            f"pct_inside={stats['pct_inside']:.2f}%, rms_vol_bps={stats['rms_vol_bps']:.3f}"
        )

    # --- Common fine grid over the body [x_l, x_h] for cubic vs quartic comparison ---
    x_grid = np.linspace(fit.x_l, fit.x_h, 400)
    iv_cubic = _iv_curve(result_cubic, x_grid, fit.t_expiry)
    iv_quartic = _iv_curve(result_quartic, x_grid, fit.t_expiry)
    both_valid = np.isfinite(iv_cubic) & np.isfinite(iv_quartic)
    iv_diff_bps = (iv_cubic[both_valid] - iv_quartic[both_valid]) * 1e4
    rms_diff_bps = float(np.sqrt(np.mean(iv_diff_bps ** 2))) if np.any(both_valid) else float("nan")
    max_diff_bps = float(np.max(np.abs(iv_diff_bps))) if np.any(both_valid) else float("nan")
    print(f"Cubic vs quartic IV curve: rms_diff={rms_diff_bps:.3f} bps, max_diff={max_diff_bps:.3f} bps")

    # --- Density comparison on the same grid ---
    density_cubic = calibration.implied_density(result_cubic, x_grid)
    density_quartic = calibration.implied_density(result_quartic, x_grid)

    try:
        assert np.all(density_cubic >= -1e-8), "cubic fitted density is negative somewhere"
        assert np.all(density_quartic >= -1e-8), "quartic fitted density is negative somewhere"
    except AssertionError as exc:
        print(f"WARNING: non-negativity check failed: {exc}")

    peak_cubic = float(np.max(density_cubic))
    peak_quartic = float(np.max(density_quartic))
    max_density_abs_diff = float(np.max(np.abs(density_cubic - density_quartic)))
    density_agreement_pct = 100.0 * max_density_abs_diff / max(peak_cubic, peak_quartic)
    print(f"Density agreement: max abs diff = {density_agreement_pct:.3f}% of peak density")

    # --- Plot ---
    fig, (ax_iv, ax_dens) = plt.subplots(1, 2, figsize=(13, 5))

    yerr_lo = np.clip(fit.vol_mid - fit.vol_bid, 0.0, None)
    yerr_hi = np.clip(fit.vol_ask - fit.vol_mid, 0.0, None)
    ax_iv.errorbar(
        fit.x, fit.vol_mid, yerr=[yerr_lo, yerr_hi],
        fmt="o", ms=3, color="gray", ecolor="lightgray", elinewidth=1, capsize=2,
        label="market bid/ask (mid \u00b1 spread)",
    )
    ax_iv.plot(x_grid, iv_cubic, color="tab:blue", label="cubic (degree 3) fit")
    ax_iv.plot(x_grid, iv_quartic, color="tab:red", linestyle="--", label="quartic (degree 4) fit")
    ax_iv.set_xlabel("normalised strike x = K/F")
    ax_iv.set_ylabel("implied volatility")
    ax_iv.set_title(
        f"Single-maturity IV fit (T={fit.t_expiry:.3f}y, {chosen.settlement}-settled)"
    )
    ax_iv.legend()

    ax_dens.plot(x_grid, density_cubic, color="tab:blue", label="cubic (degree 3) density")
    ax_dens.plot(x_grid, density_quartic, color="tab:red", linestyle="--", label="quartic (degree 4) density")
    ax_dens.set_xlabel("normalised strike x = K/F")
    ax_dens.set_ylabel("risk-neutral density c''(x)")
    ax_dens.set_title("Fitted risk-neutral densities")
    ax_dens.legend()

    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "exp2_iv_fit.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # --- Save CSVs ---
    stats_df = pd.DataFrame(
        [
            {
                "degree": 3,
                "n_quotes": stats_cubic["n_quotes"],
                "n_inside": stats_cubic["n_inside"],
                "pct_inside": stats_cubic["pct_inside"],
                "rms_vol_bps": stats_cubic["rms_vol_bps"],
            },
            {
                "degree": 4,
                "n_quotes": stats_quartic["n_quotes"],
                "n_inside": stats_quartic["n_inside"],
                "pct_inside": stats_quartic["pct_inside"],
                "rms_vol_bps": stats_quartic["rms_vol_bps"],
            },
        ]
    )
    stats_path = os.path.join(RESULTS_DIR, "exp2_fit_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    comparison_rows = [
        {"metric": "rms_diff_bps", "value": rms_diff_bps},
        {"metric": "max_diff_bps", "value": max_diff_bps},
        {"metric": "density_agreement_pct", "value": density_agreement_pct},
        {"metric": "expiry_date", "value": str(chosen.expiry_date)},
        {"metric": "settlement", "value": chosen.settlement},
        {"metric": "time_to_expiry", "value": fit.t_expiry},
        {"metric": "forward", "value": fit.forward},
        {"metric": "n_quotes_used", "value": len(fit.x)},
    ]
    comparison_path = os.path.join(RESULTS_DIR, "exp2_cubic_vs_quartic.csv")
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    print("Saved:")
    print(f"  {plot_path}")
    print(f"  {stats_path}")
    print(f"  {comparison_path}")
    print("Done.")


if __name__ == "__main__":
    main()
