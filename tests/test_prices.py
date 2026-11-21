"""Tests for the prices and benchmark tables (product spec, "Prices — Bereket").

Skipped until data/interim/prices.csv exists on the machine running the tests;
the data is shared via the team folder, not committed.

Checks (from the spec):
    - exact columns in order, snake_case
    - types: date is parseable, ticker is text, OHLC are floats, volume is int
    - no nulls in key columns (date, ticker)
    - unique (date, ticker)
    - low <= open, close <= high, volume >= 0
    - every ticker belongs to the universe
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from universe import tickers as universe_tickers

ROOT = Path(__file__).resolve().parent.parent
PRICES = ROOT / "data" / "interim" / "prices.csv"
BENCH = ROOT / "data" / "interim" / "benchmark.csv"

PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
BENCH_COLS = ["date", "spy_open", "spy_close"]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    if not PRICES.exists():
        pytest.skip(f"{PRICES.relative_to(ROOT)} not present yet")
    return pd.read_csv(PRICES)


@pytest.fixture(scope="module")
def bench() -> pd.DataFrame:
    if not BENCH.exists():
        pytest.skip(f"{BENCH.relative_to(ROOT)} not present yet")
    return pd.read_csv(BENCH)


# --- prices.csv -------------------------------------------------------------


def test_prices_columns_exact_order(prices):
    assert list(prices.columns) == PRICE_COLS


def test_prices_types(prices):
    # date parses as YYYY-MM-DD, round-trips.
    parsed = pd.to_datetime(prices["date"], format="%Y-%m-%d", errors="raise")
    assert (parsed.dt.strftime("%Y-%m-%d") == prices["date"]).all()
    assert prices["ticker"].map(type).eq(str).all()
    for c in ["open", "high", "low", "close"]:
        assert pd.api.types.is_float_dtype(prices[c]), c
    assert pd.api.types.is_integer_dtype(prices["volume"])


def test_prices_no_nulls_in_keys(prices):
    assert not prices["date"].isna().any()
    assert not prices["ticker"].isna().any()


def test_prices_no_duplicate_keys(prices):
    assert not prices.duplicated(["date", "ticker"]).any()


def test_prices_range_invariants(prices):
    assert (prices["low"] <= prices["open"]).all(), "low must be <= open"
    assert (prices["close"] <= prices["high"]).all(), "close must be <= high"
    assert (prices["open"] <= prices["high"]).all(), "open must be <= high"
    assert (prices["low"] <= prices["close"]).all(), "low must be <= close"
    assert (prices["volume"] >= 0).all(), "volume must be >= 0"


def test_prices_tickers_are_universe(prices):
    uni = set(universe_tickers())
    extra = set(prices["ticker"]) - uni
    assert not extra, f"tickers not in universe: {sorted(extra)[:10]}"


def test_prices_start_not_before_2017(prices):
    assert prices["date"].min() >= "2017-01-01"


# --- benchmark.csv ----------------------------------------------------------


def test_benchmark_columns_exact_order(bench):
    assert list(bench.columns) == BENCH_COLS


def test_benchmark_types(bench):
    parsed = pd.to_datetime(bench["date"], format="%Y-%m-%d", errors="raise")
    assert (parsed.dt.strftime("%Y-%m-%d") == bench["date"]).all()
    assert pd.api.types.is_float_dtype(bench["spy_open"])
    assert pd.api.types.is_float_dtype(bench["spy_close"])


def test_benchmark_no_nulls(bench):
    assert not bench.isna().any().any()


def test_benchmark_unique_date(bench):
    assert bench["date"].is_unique


def test_benchmark_covers_prices_dates(prices, bench):
    pdates = set(prices["date"])
    bdates = set(bench["date"])
    missing = pdates - bdates
    assert not missing, f"benchmark missing dates: {sorted(missing)[:5]}"
