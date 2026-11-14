"""Experiment 1: wing dictionary construction and admissibility (paper, Section 5).

Builds the shared put/call wing dictionaries, checks every atom's
admissibility (butterfly-freeness) two independent ways, reports the seam
condition number across every maturity slice of the live Cboe chain (falling
back to an illustrative seam grid if the live feed is unavailable), runs the
call-side cap sensitivity sweep, and plots the atom shapes.

Run with: python -m exp.exp1_wing_dictionary
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import DataLoaderError, SliceData, build_slices
from src import config
from src.wing_dictionary import (
    WingDictionary,
    build_both_dictionaries,
    build_geometric_dictionary,
    check_admissibility,
    verify_quadratic_form,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
logger = logging.getLogger("exp1_wing_dictionary")

# Seam magnitude used to cross-check admissibility with verify_quadratic_form;
# small in magnitude so the y-grid (0, 1/|k_s|] covers a wide range, making
# the check close to the full y > 0 claim behind mu >= 1 / mu > 0.
_ADMISSIBILITY_PROBE_K_S_MAGNITUDE = 0.01
_MU_BOUNDARY_TOLERANCE = 1e-6


def verify_side_admissibility(side: str, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-check check_admissibility against verify_quadratic_form for one side."""
    check = check_admissibility(mu, side)
    k_s_probe = -_ADMISSIBILITY_PROBE_K_S_MAGNITUDE if side == "put" else _ADMISSIBILITY_PROBE_K_S_MAGNITUDE
    quad_check = verify_quadratic_form(mu, side, k_s_probe)
    mismatch = check != quad_check
    if np.any(mismatch):
        near_boundary = np.abs(mu - 1.0) < _MU_BOUNDARY_TOLERANCE
        bad = mismatch & ~near_boundary
        if np.any(bad):
            raise AssertionError(
                f"Admissibility mismatch on side={side!r} at mu={mu[bad]}: "
                f"check_admissibility={check[bad]} vs verify_quadratic_form={quad_check[bad]}"
            )
    return check, quad_check


def build_atoms_dataframe(put_dict: WingDictionary, call_dict: WingDictionary) -> pd.DataFrame:
    rows = []
    for side, d in (("put", put_dict), ("call", call_dict)):
        check, quad_check = verify_side_admissibility(side, d.mu)
        for i in range(d.n_atoms):
            rows.append(
                {
                    "side": side,
                    "atom_index": i,
                    "b": float(d.b[i]),
                    "mu": float(d.mu[i]),
                    "admissible": bool(check[i]),
                    "admissible_quadratic_check": bool(quad_check[i]),
                }
            )
    return pd.DataFrame(rows)


def compute_condition_numbers_real(
    slices: list[SliceData], put_dict: WingDictionary, call_dict: WingDictionary
) -> tuple[pd.DataFrame, dict, list[float], list[float]]:
    rows = []
    put_ks: list[float] = []
    call_ks: list[float] = []
    for i, s in enumerate(slices):
        strikes = s.quotes["strike"]
        k_s_put = float(np.log(strikes.min() / s.forward))
        k_s_call = float(np.log(strikes.max() / s.forward))
        cond_put = put_dict.condition_number(k_s_put)
        cond_call = call_dict.condition_number(k_s_call)
        put_ks.append(k_s_put)
        call_ks.append(k_s_call)
        common = {"slice_index": i, "expiry_date": s.expiry_date, "settlement": s.settlement, "t_expiry": s.time_to_expiry}
        rows.append({**common, "side": "put", "k_s": k_s_put, "condition_number": cond_put})
        rows.append({**common, "side": "call", "k_s": k_s_call, "condition_number": cond_call})
    df = pd.DataFrame(rows)[["side", "slice_index", "expiry_date", "settlement", "t_expiry", "k_s", "condition_number"]]
    stats = {
        "put_median": float(df[df.side == "put"]["condition_number"].median()),
        "put_max": float(df[df.side == "put"]["condition_number"].max()),
        "call_median": float(df[df.side == "call"]["condition_number"].median()),
        "call_max": float(df[df.side == "call"]["condition_number"].max()),
        "n_slices": len(slices),
    }
    return df, stats, put_ks, call_ks


def illustrative_condition_numbers(put_dict: WingDictionary, call_dict: WingDictionary) -> tuple[pd.DataFrame, dict]:
    """Fallback seam grid (0.02 to 2.5 in |k_s|) used only when the live feed fails."""
    magnitudes = np.geomspace(0.02, 2.5, 15)
    rows = []
    for side, d, sign in (("put", put_dict, -1.0), ("call", call_dict, 1.0)):
        for m in magnitudes:
            cond = d.condition_number(sign * float(m))
            rows.append({"side": side, "k_s_magnitude": float(m), "condition_number": cond, "note": "illustrative, no live data"})
    df = pd.DataFrame(rows)
    stats = {
        "put_median": float(df[df.side == "put"]["condition_number"].median()),
        "put_max": float(df[df.side == "put"]["condition_number"].max()),
        "call_median": float(df[df.side == "call"]["condition_number"].median()),
        "call_max": float(df[df.side == "call"]["condition_number"].max()),
        "n_slices": 0,
    }
    return df, stats


def cap_sensitivity_table(k_s_call: float) -> pd.DataFrame:
    """Sweep the call-side cap b_hi and report the fattest-tail atom's deep-tail value.

    Paper's expected effect: the slowest-decaying (last, smallest-mu) atom's
    value at k_s_call + DEEP_TAIL_LOG_MONEYNESS_UNITS should grow by roughly an
    order of magnitude or more as b_hi rises from 0.55 to 1.00, since a larger
    cap admits a smaller-mu atom whose tail decays much more slowly.
    """
    rows = []
    for b_hi in config.CALL_B_HI_SWEEP:
        d = build_geometric_dictionary(config.N_CALL_ATOMS, config.CALL_B_LO, b_hi, "call")
        k_eval = k_s_call + config.DEEP_TAIL_LOG_MONEYNESS_UNITS
        values = d.atom_values(np.array([k_eval]), k_s_call)
        deep_tail_price = float(values[-1, 0])
        rows.append({"b_hi": b_hi, "deep_tail_price": deep_tail_price, "k_s_used": k_s_call})
    return pd.DataFrame(rows)


def plot_atom_shapes(
    put_dict: WingDictionary, call_dict: WingDictionary, k_s_put: float, k_s_call: float, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    k_put = -np.linspace(abs(k_s_put), abs(k_s_put) + 3.0, 200)
    h_put = put_dict.atom_values(k_put, k_s_put)
    for i in range(put_dict.n_atoms):
        axes[0].plot(k_put, h_put[i], label=f"atom {i} (b={put_dict.b[i]:.3f}, mu={put_dict.mu[i]:.3f})")
    axes[0].set_title(f"Put wing atoms h_i(k), seam k_s={k_s_put:.4f}")
    axes[0].set_xlabel("log-moneyness k")
    axes[0].set_ylabel("h_i(k)")
    axes[0].legend(fontsize=7)

    k_call = np.linspace(abs(k_s_call), abs(k_s_call) + 3.0, 200)
    h_call = call_dict.atom_values(k_call, k_s_call)
    for i in range(call_dict.n_atoms):
        axes[1].plot(k_call, h_call[i], label=f"atom {i} (b={call_dict.b[i]:.3f}, mu={call_dict.mu[i]:.3f})")
    axes[1].set_title(f"Call wing atoms h_i(k), seam k_s={k_s_call:.4f}")
    axes[1].set_xlabel("log-moneyness k")
    axes[1].set_ylabel("h_i(k)")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    put_dict, call_dict = build_both_dictionaries(
        config.N_PUT_ATOMS,
        config.N_CALL_ATOMS,
        config.PUT_B_LO,
        config.PUT_B_HI,
        config.CALL_B_LO,
        config.CALL_B_HI,
    )

    atoms_df = build_atoms_dataframe(put_dict, call_dict)
    atoms_df.to_csv(RESULTS_DIR / "exp1_atoms.csv", index=False)

    notes_lines: list[str] = []
    slices: list[SliceData] | None = None
    try:
        slices, meta = build_slices()
        logger.info("build_slices succeeded: %d slices, %d expiry dates", meta["n_slices"], meta["n_expiry_dates"])
    except DataLoaderError as exc:
        logger.warning("build_slices failed (DataLoaderError): %s -- falling back to illustrative seam grid", exc)
    except Exception as exc:  # noqa: BLE001 - network failures can surface as various exceptions
        logger.warning("build_slices failed (unexpected error): %s -- falling back to illustrative seam grid", exc)

    if slices:
        cond_df, stats, put_ks, call_ks = compute_condition_numbers_real(slices, put_dict, call_dict)
        k_s_call_rep = float(np.median(np.abs(call_ks)))
        k_s_put_rep = -float(np.median(np.abs(put_ks)))
        data_source_label = f"live Cboe slices (n_slices={stats['n_slices']})"
    else:
        cond_df, stats = illustrative_condition_numbers(put_dict, call_dict)
        k_s_call_rep = 0.3  # documented illustrative placeholder
        k_s_put_rep = -0.3
        data_source_label = "illustrative grid (no live data)"
        notes_lines.append("no live slice data; seam grid is illustrative, not from real quotes")
    cond_df.to_csv(RESULTS_DIR / "exp1_condition_numbers.csv", index=False)

    cap_df = cap_sensitivity_table(k_s_call_rep)
    cap_df.to_csv(RESULTS_DIR / "exp1_cap_sensitivity.csv", index=False)

    plot_atom_shapes(put_dict, call_dict, k_s_put_rep, k_s_call_rep, RESULTS_DIR / "exp1_wing_atoms.png")

    if notes_lines:
        with open(RESULTS_DIR / "exp1_notes.txt", "a") as f:
            for line in notes_lines:
                f.write(line + "\n")

    n_put_admissible = int(atoms_df[atoms_df.side == "put"]["admissible"].sum())
    n_call_admissible = int(atoms_df[atoms_df.side == "call"]["admissible"].sum())

    print("=" * 72)
    print("Experiment 1: wing dictionary construction and admissibility")
    print("=" * 72)
    print(f"Put wing:  {n_put_admissible}/{put_dict.n_atoms} atoms admissible (mu >= 1)")
    print(f"Call wing: {n_call_admissible}/{call_dict.n_atoms} atoms admissible (mu > 0)")
    print("check_admissibility agrees with verify_quadratic_form (1e-6 tolerance near mu=1).")
    print()
    print(f"Condition numbers computed from: {data_source_label}")
    print(f"  put  side: median={stats['put_median']:.3e}  max={stats['put_max']:.3e}")
    print(f"  call side: median={stats['call_median']:.3e}  max={stats['call_max']:.3e}")
    if slices:
        print(f"  {stats['n_slices']} slices (maturities) all served by the same shared put_dict/call_dict objects")
    print()
    print(f"Cap sensitivity sweep (deep-tail atom value, k_s_call used = {k_s_call_rep:.4f}):")
    print(cap_df.to_string(index=False))
    print()
    print(f"Results written under: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
