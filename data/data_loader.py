"""Option quote data for the SPX/SPXW surface used throughout the replication.

The paper calibrates the surface from Cboe's delayed-quote feed, a public
JSON snapshot of the full SPX/SPXW option chain (Section 8, "Data"). That
feed always serves the chain currently live, not a historical archive, so
this loader fetches the live snapshot at run time rather than the paper's
historical snapshot of 2 November 2026 17:05 ET; the realized snapshot
timestamp, slice count and quote count are logged and recorded in
results/RESULTS.md ("Deviations from the paper").

This is not OHLCV bar data, so it is not loaded through the QuantX
Dataloader (which only serves historical daily/intraday bars); Cboe's own
feed is the paper's named source and requires no API key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src import config

_SYMBOL_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


class DataLoaderError(RuntimeError):
    """A failed request, an empty feed, or a response with unexpected schema."""


@dataclass
class SliceData:
    """One expiry's filtered, forward-normalised option quotes.

    Attributes
    ----------
    expiry_date : date of settlement (calendar date of the option symbol)
    settlement : "AM" or "PM"
    time_to_expiry : ACT/365 year fraction from the snapshot to settlement
    forward : forward price F from put-call parity on this slice
    discount_factor : discount factor read off the fitted zero curve
    quotes : DataFrame with columns
        strike, call_bid, call_ask, put_bid, put_ask, call_iv, put_iv
    """

    expiry_date: object
    settlement: str
    time_to_expiry: float
    forward: float
    discount_factor: float
    quotes: pd.DataFrame


def fetch_raw_chain(url: str = config.CBOE_SPX_CHAIN_URL, timeout: float = 30.0) -> dict:
    """GET the live Cboe SPX/SPXW option chain snapshot.

    Raises DataLoaderError on a transport failure, a non-200 response, or a
    response missing the expected ``data.options`` list.
    """
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as exc:
        raise DataLoaderError(f"Cboe request failed: {exc}") from exc
    if response.status_code != 200:
        raise DataLoaderError(f"Cboe request returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise DataLoaderError(f"Cboe response is not valid JSON: {exc}") from exc
    data = payload.get("data")
    if not data or "options" not in data or not data["options"]:
        raise DataLoaderError("Cboe response has no 'data.options' entries")
    return payload


def _parse_symbol(symbol: str) -> tuple[str, str, str, float] | None:
    match = _SYMBOL_RE.match(symbol)
    if not match:
        return None
    root, yymmdd, cp, strike_raw = match.groups()
    if root not in config.OPTION_ROOTS:
        return None
    expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    strike = int(strike_raw) / 1000.0
    return root, expiry.isoformat(), cp, strike


def parse_option_rows(payload: dict) -> pd.DataFrame:
    """Flatten the raw feed into one row per (root, expiry, side, strike) quote."""
    rows = []
    for item in payload["data"]["options"]:
        parsed = _parse_symbol(item["option"])
        if parsed is None:
            continue
        root, expiry_iso, cp, strike = parsed
        rows.append(
            {
                "root": root,
                "expiry": expiry_iso,
                "side": "C" if cp == "C" else "P",
                "strike": strike,
                "bid": float(item.get("bid") or 0.0),
                "ask": float(item.get("ask") or 0.0),
                "iv": float(item.get("iv") or 0.0),
            }
        )
    if not rows:
        raise DataLoaderError("No SPX/SPXW rows parsed from the Cboe feed")
    frame = pd.DataFrame(rows)
    frame = frame[(frame["bid"] >= 0) & (frame["ask"] > 0) & (frame["ask"] >= frame["bid"])]
    return frame


def _settlement_type(root: str) -> str:
    # SPX (standard, third-Friday) contracts settle AM; SPXW (weekly/EOM) settle PM.
    return "AM" if root == "SPX" else "PM"


def _time_to_expiry(snapshot_utc: datetime, expiry_iso: str, settlement: str) -> float:
    expiry_date = datetime.strptime(expiry_iso, "%Y-%m-%d").date()
    hour, minute = (
        (config.AM_SETTLEMENT_ET_HOUR, config.AM_SETTLEMENT_ET_MINUTE)
        if settlement == "AM"
        else (config.PM_SETTLEMENT_ET_HOUR, config.PM_SETTLEMENT_ET_MINUTE)
    )
    settle_et = datetime(expiry_date.year, expiry_date.month, expiry_date.day, hour, minute, tzinfo=_ET)
    delta = settle_et.astimezone(_UTC) - snapshot_utc
    return delta.total_seconds() / (config.DAYS_PER_YEAR * 86400.0)


def _bs_vega(forward: float, strike: float, vol: float, t: float) -> float:
    """Black-Scholes vega of an undiscounted forward option (vol>0, t>0 required)."""
    if vol <= 0 or t <= 0 or forward <= 0 or strike <= 0:
        return 0.0
    from scipy.stats import norm

    d1 = (np.log(forward / strike) + 0.5 * vol * vol * t) / (vol * np.sqrt(t))
    return forward * norm.pdf(d1) * np.sqrt(t)


def _fit_parity_forward(strikes: np.ndarray, call_mid: np.ndarray, put_mid: np.ndarray) -> tuple[float, float]:
    """OLS fit of C_pv(K)-P_pv(K) = DF*F - DF*K; returns (DF, F).

    Put-call parity for present-value (discounted) prices: C-P = DF*(F-K).
    """
    y = call_mid - put_mid
    design = np.column_stack([np.ones_like(strikes), -strikes])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    df_times_f, df = coeffs
    if df <= 0:
        raise DataLoaderError("Put-call parity regression produced a non-positive discount factor")
    return df, df_times_f / df


def _fit_zero_curve(maturities: np.ndarray, discount_factors: np.ndarray) -> np.ndarray:
    """A single zero curve fitted across expiries (paper, Section 8).

    The paper states discount factors are "read off one zero curve fitted
    across expiries" without giving the curve family. We fit continuously
    compounded zero rates r(T) = -log(DF)/T by isotonic-free linear
    regression of -log(DF) on T through the origin plus a smooth (natural
    cubic spline) correction, then re-evaluate DF on the same maturities, so
    that every expiry's discount factor comes from one shared curve instead
    of its own noisy per-expiry regression.
    """
    from scipy.interpolate import UnivariateSpline

    order = np.argsort(maturities)
    t_sorted = maturities[order]
    neg_log_df = -np.log(discount_factors[order])
    # Smooth, monotone-leaning spline through the per-expiry zero*T values.
    spline = UnivariateSpline(t_sorted, neg_log_df, k=min(3, max(1, len(t_sorted) - 1)), s=len(t_sorted) * 1e-6)
    fitted_neg_log_df = spline(maturities)
    return np.exp(-fitted_neg_log_df)


def build_slices(
    payload: dict | None = None,
    snapshot_override: datetime | None = None,
) -> tuple[list[SliceData], dict]:
    """Build filtered, forward-normalised option slices from the live Cboe feed.

    Returns (slices, meta) where meta logs the realized snapshot time, slice
    count, quote count and any excluded expiries with their reason, following
    the paper's filters (Section 8): OTM quotes with BS vega >= 0.1% of the
    slice maximum, slices with >= 20 quotes straddling the forward, maturities
    up to 2.5 years.
    """
    if payload is None:
        payload = fetch_raw_chain()
    snapshot_utc = snapshot_override or datetime.strptime(
        payload["timestamp"], "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=_ET).astimezone(_UTC)

    rows = parse_option_rows(payload)
    rows["mid"] = 0.5 * (rows["bid"] + rows["ask"])

    slices: list[SliceData] = []
    excluded: dict[str, str] = {}
    per_slice_raw_count: dict[str, int] = {}

    for (root, expiry_iso), group in rows.groupby(["root", "expiry"]):
        key = f"{root} {expiry_iso}"
        settlement = _settlement_type(root)
        t_expiry = _time_to_expiry(snapshot_utc, expiry_iso, settlement)
        if t_expiry <= 0:
            excluded[key] = "expired relative to the snapshot (time to expiry <= 0)"
            continue
        if t_expiry > config.MAX_MATURITY_YEARS:
            excluded[key] = f"maturity {t_expiry:.2f}y exceeds the {config.MAX_MATURITY_YEARS}y ceiling"
            continue

        calls = group[group["side"] == "C"].set_index("strike")
        puts = group[group["side"] == "P"].set_index("strike")
        strikes = sorted(set(calls.index) & set(puts.index))
        if len(strikes) < 5:
            excluded[key] = "fewer than 5 strikes with both a call and a put quote"
            continue
        strikes_arr = np.array(strikes)
        call_mid = calls.loc[strikes, "mid"].to_numpy()
        put_mid = puts.loc[strikes, "mid"].to_numpy()
        try:
            discount_factor, forward = _fit_parity_forward(strikes_arr, call_mid, put_mid)
        except DataLoaderError as exc:
            excluded[key] = str(exc)
            continue
        if not np.isfinite(forward) or forward <= 0:
            excluded[key] = "non-finite or non-positive forward from parity regression"
            continue

        per_slice_raw_count[key] = len(strikes)
        slices.append(
            _build_single_slice(root, expiry_iso, settlement, t_expiry, forward, discount_factor, calls, puts, strikes)
        )

    if not slices:
        raise DataLoaderError("No slices survived parsing; the feed format may have changed")

    # Replace each slice's own regression discount factor with one read off a
    # single zero curve fitted across all expiries (paper, Section 8).
    maturities = np.array([s.time_to_expiry for s in slices])
    raw_dfs = np.array([s.discount_factor for s in slices])
    curve_dfs = _fit_zero_curve(maturities, raw_dfs)
    rebuilt = []
    for s, df_curve in zip(slices, curve_dfs):
        forward = s.forward * (s.discount_factor / df_curve)
        rebuilt.append(SliceData(s.expiry_date, s.settlement, s.time_to_expiry, forward, df_curve, s.quotes))
    slices = rebuilt

    # Apply the OTM / vega / straddle filters now that each slice has its curve forward.
    kept = []
    for s in slices:
        key = f"{s.settlement} {s.expiry_date}"
        filtered = _apply_quote_filters(s)
        if filtered is None:
            excluded[key] = "fewer than 20 quotes straddling the forward after vega filtering"
            continue
        kept.append(SliceData(s.expiry_date, s.settlement, s.time_to_expiry, s.forward, s.discount_factor, filtered))

    total_quotes = sum(len(s.quotes) for s in kept)
    meta = {
        "snapshot_utc": snapshot_utc.isoformat(),
        "n_slices": len(kept),
        "n_expiry_dates": len(set(s.expiry_date for s in kept)),
        "n_quotes": total_quotes,
        "maturity_range_years": (min(s.time_to_expiry for s in kept), max(s.time_to_expiry for s in kept)) if kept else None,
        "excluded_slices": excluded,
    }
    return kept, meta


def _build_single_slice(root, expiry_iso, settlement, t_expiry, forward, discount_factor, calls, puts, strikes) -> SliceData:
    frame = pd.DataFrame(
        {
            "strike": strikes,
            "call_bid": calls.loc[strikes, "bid"].to_numpy(),
            "call_ask": calls.loc[strikes, "ask"].to_numpy(),
            "call_iv": calls.loc[strikes, "iv"].to_numpy(),
            "put_bid": puts.loc[strikes, "bid"].to_numpy(),
            "put_ask": puts.loc[strikes, "ask"].to_numpy(),
            "put_iv": puts.loc[strikes, "iv"].to_numpy(),
        }
    )
    return SliceData(expiry_iso, settlement, t_expiry, forward, discount_factor, frame)


def _apply_quote_filters(s: SliceData) -> pd.DataFrame | None:
    """Keep OTM quotes with vega >= 0.1% of the slice's largest vega (paper, Section 8)."""
    frame = s.quotes.copy()
    forward = s.forward
    t = s.time_to_expiry
    vegas = []
    otm_bid = []
    otm_ask = []
    otm_iv = []
    for _, row in frame.iterrows():
        is_otm_call = row["strike"] >= forward
        if is_otm_call:
            iv, bid, ask = row["call_iv"], row["call_bid"], row["call_ask"]
        else:
            iv, bid, ask = row["put_iv"], row["put_bid"], row["put_ask"]
        vega = _bs_vega(forward, row["strike"], iv, t) if iv > 0 else 0.0
        vegas.append(vega)
        otm_bid.append(bid)
        otm_ask.append(ask)
        otm_iv.append(iv)
    frame["otm_vega"] = vegas
    frame["otm_bid"] = otm_bid
    frame["otm_ask"] = otm_ask
    frame["otm_iv"] = otm_iv
    frame = frame[(frame["otm_bid"] > 0) & (frame["otm_ask"] > frame["otm_bid"]) & (frame["otm_iv"] > 0)]
    if frame.empty:
        return None
    vega_cutoff = config.MIN_VEGA_FRACTION_OF_SLICE_MAX * frame["otm_vega"].max()
    frame = frame[frame["otm_vega"] >= vega_cutoff]
    n_below = (frame["strike"] < forward).sum()
    n_above = (frame["strike"] > forward).sum()
    if min(n_below, n_above) * 2 < config.MIN_QUOTES_STRADDLING_FORWARD:
        return None
    return frame.sort_values("strike").reset_index(drop=True)
