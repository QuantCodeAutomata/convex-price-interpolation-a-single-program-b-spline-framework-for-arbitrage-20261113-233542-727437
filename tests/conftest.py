"""Shared pytest fixtures for the Convex Price Interpolation replication."""
from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test loads real option quotes from the live Cboe feed"
    )


@pytest.fixture(scope="session")
def live_chain_available() -> bool:
    """Whether the live Cboe feed is reachable (no API key required, but the
    test environment may have no outbound network access)."""
    import requests

    try:
        r = requests.get(
            "https://cdn-api.cboe.com/api/global/delayed_quotes/options/_SPX.json",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def cached_slices(request: pytest.FixtureRequest, live_chain_available: bool):
    """A session-cached, reduced set of real SPX slices loaded once through
    data/data_loader.py, skipped when the live feed is not reachable."""
    if not live_chain_available:
        pytest.skip("Live Cboe option-chain feed is not reachable from this environment")
    cache_dir = request.config.cache.mkdir("cpi_slices")
    cache_file = cache_dir / "slices.pkl"
    import pickle

    from data.data_loader import build_slices

    if cache_file.exists():
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)
    slices, meta = build_slices()
    # Keep a small, fast slice for unit-style market_data tests: the shortest
    # maturity and one roughly mid-dated slice.
    slices_sorted = sorted(slices, key=lambda s: s.time_to_expiry)
    reduced = slices_sorted[:1] + slices_sorted[len(slices_sorted) // 2: len(slices_sorted) // 2 + 1]
    with open(cache_file, "wb") as fh:
        pickle.dump((reduced, meta), fh)
    return reduced, meta
