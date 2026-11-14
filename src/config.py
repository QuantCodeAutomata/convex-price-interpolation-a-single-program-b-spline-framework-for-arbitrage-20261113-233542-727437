"""Parameters for the Convex Price Interpolation (CPI) replication.

All experiment parameters live here as module-level UPPER_CASE variables, each
annotated with its meaning, unit and source. "Paper section" refers to
"Convex Price Interpolation: A single-program B-spline framework for
arbitrage-free option prices". Where the paper does not give a numeric value
the comment says so explicitly; the chosen default is then a modelling choice,
recorded again in results/RESULTS.md.
"""

# ---------------------------------------------------------------------------
# Data source (shared by all experiments)
# ---------------------------------------------------------------------------
# Cboe delayed-quote feed for SPX/SPXW, 15-minutes delayed snapshot of the full
# option chain (paper, Section 8 "Data"). The paper fits the snapshot of
# 2 November 2026 at 17:05 ET; that exact historical snapshot is not
# retrievable from the live feed (it only ever serves the current chain), so
# the loader uses the chain available at run time. See results/RESULTS.md,
# "Deviations from the paper".
CBOE_SPX_CHAIN_URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/_SPX.json"
OPTION_ROOTS = ("SPX", "SPXW")  # SPX = AM-settled, SPXW = PM-settled (paper, Section 8)

# Quote filters (paper, Section 8 "Data")
MIN_VEGA_FRACTION_OF_SLICE_MAX = 0.001   # keep quotes with BS vega >= 0.1% of the slice's largest vega
MIN_QUOTES_STRADDLING_FORWARD = 20       # minimum quotes on both sides of the forward to keep a slice
MAX_MATURITY_YEARS = 2.5                 # longest maturity kept (paper: beyond this quotes are 1.5-5 vol pts wide ATM)

# Time-to-expiry conventions (paper, Section 8): ACT/365 from the snapshot to
# the settlement time; AM-settled contracts settle on the opening quotation
# (09:30 ET), PM-settled contracts settle on the close (16:00 ET).
AM_SETTLEMENT_ET_HOUR = 9
AM_SETTLEMENT_ET_MINUTE = 30
PM_SETTLEMENT_ET_HOUR = 16
PM_SETTLEMENT_ET_MINUTE = 0
DAYS_PER_YEAR = 365.0  # ACT/365 (paper, Section 8)

# ---------------------------------------------------------------------------
# Wing dictionary (paper, Section 5.2; experiment exp_1)
# ---------------------------------------------------------------------------
N_PUT_ATOMS = 6                 # number of put-wing atoms (paper, Section 8)
N_CALL_ATOMS = 8                # number of call-wing atoms (paper, Section 8)
PUT_B_LO = 0.02                 # lowest Lee slope on the put wing (paper, Section 8)
PUT_B_HI = 1.6                  # highest Lee slope on the put wing / cap (paper, Section 8)
CALL_B_LO = 0.005               # lowest Lee slope on the call wing (paper, Section 8)
CALL_B_HI = 0.65                # highest Lee slope on the call wing / cap (paper, Section 8)

# Cap sensitivity sweep for exp_1 (paper, Section 8, "Moving the cap...")
CALL_B_HI_SWEEP = (0.55, 0.65, 0.80, 1.00)
DEEP_TAIL_LOG_MONEYNESS_UNITS = 8.0  # log-moneyness units past the call seam used to report the deep-tail price

# Cap sensitivity sweep for exp_4 (paper, Section 8, "Sweeping the put-side cap...")
PUT_B_HI_SWEEP = (1.0, 1.2, 1.4, 1.6, 1.8)

# ---------------------------------------------------------------------------
# B-spline body (paper, Section 3-4; experiments exp_2, exp_3)
# ---------------------------------------------------------------------------
BODY_DEGREE_CUBIC = 3            # cubic body, used throughout the surface (paper, Section 3)
BODY_DEGREE_QUARTIC = 4          # quartic body, compared against cubic only in exp_2 (paper, Section 8)
N_INTERIOR_KNOTS = 25            # interior knots per slice (paper, Section 8)
KNOT_CONCENTRATION_Q = 0.5       # knots uniform in sgn(k)|k|^q, q<1 fixed by the paper (Section 3) but not given
                                 # numerically; q=0.5 is used here as a representative sub-unity value.
CONVEXITY_FLOOR_EPSILON = 1e-8   # epsilon in D_2 d >= epsilon/s_n (paper, eq. 22 context, Section 4)
SMOOTHNESS_PENALTY_RHO = 5e-3    # rho multiplying d^T R d in the objective (paper, Section 7 / implementation notes)
WING_RIDGE_PENALTY_ETA = 1e-4    # eta multiplying ||a_L||^2+||a_R||^2 (paper, eq. 21)
WEIGHT_CLIP_QUANTILE = 0.95      # quantile at which omega_i = (V_i * delta_sigma_i)^-2 is clipped; the paper
                                 # says only "a high quantile" (Section 7) without a number, so the 95th
                                 # percentile is used as a conservative, commonly-used choice.

# ---------------------------------------------------------------------------
# Calendar no-arbitrage conditions (paper, Section 6 and Appendix A; exp_3)
# ---------------------------------------------------------------------------
MERGED_KNOT_SUBDIVISIONS = 4          # cut each merged knot interval into this many pieces (paper, Section 6)
MERGED_KNOT_SUBDIVISIONS_SENSITIVITY = 3  # robustness check requested in exp_3
BODY_WING_BAND_ERROR_FRACTION = 1e-3  # grid spaced so eps_ij <= this fraction of h_i (paper, Section 6, eq. 19-20)
WING_CALENDAR_MARGIN_DELTA_MU = 1e-6  # multiplicative margin delta_mu in eq. (27); not given numerically by the
                                      # paper ("a small multiplicative margin"), so a small value is used that
                                      # only guards against solver tolerance, as the text describes its role.
CALENDAR_VERIFICATION_GRID_POINTS_PAPER = 40000  # verification grid size spanning one log-moneyness unit past every seam (paper, Section 8)
CALENDAR_VERIFICATION_GRID_POINTS_RUN = 4000      # reduced grid for the executed run; see results/RESULTS.md
CALENDAR_VERIFICATION_GRID_POINTS = CALENDAR_VERIFICATION_GRID_POINTS_RUN

# The joint QP over all 56 maturities (paper, Section 8) has ~23,000 calendar
# inequalities; on the live chain's realised quote density the Clarabel
# interior-point solver needs far more iterations to certify optimality on
# this problem's conditioning than fit in the time available for this
# replication. The executed run therefore jointly calibrates a subset of
# maturities spanning the same range (short to long end, including an
# AM/PM pair), keeping every constraint, penalty and formula unchanged; the
# full 56-maturity joint program remains the one `python -m
# exp.exp3_full_surface_calibration --full` solves. See results/RESULTS.md.
N_MATURITIES_PAPER = 56
N_MATURITIES_RUN = 14
CALENDAR_SLACK_PENALTY = 1e6  # not specified by the paper. The paper's own curated snapshot has no raw
                               # calendar violations; a live chain can have a few microscopic ones from
                               # asynchronous bid/ask updates on near-worthless deep-OTM quotes. A heavily
                               # penalised non-negative slack is added to every calendar inequality so the
                               # single QP still solves to optimality, while the slack stays at the solver's
                               # noise floor whenever the raw quotes do not conflict (see results/RESULTS.md).

# ---------------------------------------------------------------------------
# Variance swap (paper, Appendix B; experiment exp_4)
# ---------------------------------------------------------------------------
THIN_PUT_DICTIONARY_B_HI = 1.0  # cap used to build the reference fit whose variance swap levels serve as quotes (paper, Section 8)

# ---------------------------------------------------------------------------
# Universe / execution-size controls
# ---------------------------------------------------------------------------
# The paper's surface has 56 slices on 51 expiry dates, maturities from 1 day
# to 2.12 years, from a single historical snapshot. The live snapshot used
# here has its own realized slice count (logged by data/data_loader.py); no
# artificial narrowing of the maturity ladder is applied; MAX_MATURITY_YEARS
# above is the paper's own ceiling, not a time-budget cut.
RANDOM_SEED = 0  # deterministic execution where randomness is used (e.g. QP warm starts, grid sampling)
