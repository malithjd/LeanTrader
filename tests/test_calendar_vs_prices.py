"""The trading calendar in dating.py must match the dates in prices.csv.

Skipped until Bereket's data/interim/prices.csv exists on your machine.
"""

from pathlib import Path

import pandas as pd
import pytest

from dating import trading_days

PRICES = Path(__file__).resolve().parent.parent / "data" / "interim" / "prices.csv"


@pytest.mark.skipif(not PRICES.exists(), reason="data/interim/prices.csv not present yet")
def test_price_dates_equal_nyse_sessions():
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(PRICES, usecols=["date"])["date"]).unique()).sort_values()
    expected = trading_days(dates.min(), dates.max())
    missing = expected.difference(dates)
    extra = dates.difference(expected)
    assert missing.empty, f"NYSE trading days with no prices: {list(missing[:10].date)}"
    assert extra.empty, f"Price dates that are not NYSE trading days: {list(extra[:10].date)}"
