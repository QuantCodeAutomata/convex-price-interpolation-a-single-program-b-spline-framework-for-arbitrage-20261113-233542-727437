# Convex Price Interpolation: A single-program B-spline framework for arbitrage-free option prices

Replication of the paper *"Convex Price Interpolation: A single-program B-spline
framework for arbitrage-free option prices"*.

This repository is a work in progress; see `results/RESULTS.md` for the current
status of each experiment, data source and any deviation from the paper.

## Layout

- `src/` — shared library: Black-76 pricing (`black76.py`), the asymptotic wing
  dictionary (`wing_dictionary.py`), the B-spline body basis (`bspline_body.py`),
  the single-maturity QP (`calibration.py`), calendar no-arbitrage conditions
  (`calendar_constraints.py`), the joint multi-maturity QP (`surface.py`), the
  variance-swap closed form (`variance_swap.py`) and all parameters
  (`config.py`).
- `data/data_loader.py` — loads the live SPX/SPXW option chain from Cboe's
  public delayed-quote feed (no API key required) and builds forward prices,
  a shared zero curve and filtered per-maturity quote slices.
- `exp/` — one script per experiment, runnable as `python -m exp.<name>`.
- `tests/` — unit tests (`python -m pytest -m "not market_data"`) and
  market-data integration tests (`python -m pytest -m market_data`), the
  latter requiring outbound network access to Cboe's feed.
- `results/` — tables, figures and `RESULTS.md` written by the experiment
  scripts.

## Setup

```bash
pip install -r requirements.txt
```

No API key is required: the paper's data source, Cboe's delayed-quote feed,
is a public JSON endpoint.

## Running

```bash
python -m pytest -m "not market_data"      # unit tests on synthetic fixtures
python -m pytest -m market_data            # integration tests on a live quote slice
python -m exp.exp1_wing_dictionary
python -m exp.exp2_single_maturity_fit
python -m exp.exp3_full_surface_calibration
python -m exp.exp4_variance_swap
```

`reproduce.sh` runs the full sequence above.

See `results/RESULTS.md` for the comparison against the paper's reported
numbers, data-source and configuration notes, and the "Deviations from the
paper" / "Not reproduced" sections.
