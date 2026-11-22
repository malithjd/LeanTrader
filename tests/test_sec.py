"""Tests for the SEC tables (product spec, "SEC filings — Malith").

Part 1 runs everywhere (no network): sec_edgar's logic on a small fake SEC response.
Part 2 checks the real CSVs and is skipped until data/interim/sec_*.csv exist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dating import trading_days
from sec_edgar import DAILY_COLS, DAILY_FROM, EVENT_COLS, FORMS, ITEMS, build_daily, build_events
from universe import tickers as universe_tickers

ROOT = Path(__file__).resolve().parent.parent
EVENTS_CSV = ROOT / "data" / "interim" / "sec_events.csv"
DAILY_CSV = ROOT / "data" / "interim" / "sec_daily.csv"

APPLE_CIK = "0000320193"

# A fake submissions page in SEC's columnar shape. Row 0 is a real Apple filing.
FAKE = pd.DataFrame({
    "accessionNumber": ["0000320193-26-000024", "0000320193-26-000023", "0000320193-26-000020",
                        "0000320193-26-000019", "0000320193-16-000001"],
    "filingDate": ["2026-11-02", "2026-10-27", "2026-07-31", "2026-07-31", "2016-05-02"],
    "reportDate": ["2026-11-02", "2026-10-26", "2026-06-27", "", "2016-05-01"],
    "acceptanceDateTime": [
        "2026-11-02T21:30:23.000Z",  # 4:30 pm ET -> after the cutoff -> Tue Nov 3
        "2026-10-27T14:00:00.000Z",  # 10:00 am ET -> same day
        "2026-07-31T21:00:00.000Z",  # Fri 5:00 pm ET -> Mon Aug 3
        "2026-07-31T15:00:00.000Z",
        "2016-05-02T20:00:00.000Z",
    ],
    "form": ["8-K", "8-K", "10-Q", "4", "8-K"],
    "items": ["2.02,9.01", "5.02", "", "", "2.02"],
})


def fake_fetch(cik: str) -> pd.DataFrame:
    return FAKE.copy()


@pytest.fixture
def events() -> pd.DataFrame:
    uni = pd.DataFrame({"ticker": ["AAPL"], "cik": [APPLE_CIK]})
    return build_events(uni, fetch=fake_fetch)


# ---------- Part 1: logic ----------

def test_acceptance_time_is_real_utc():
    """Checked once against EDGAR: the index page of 0000320193-26-000024 says
    'Accepted 2026-11-02 16:30:23' (New York) and the JSON says 21:30:23Z."""
    ny = pd.Timestamp("2026-11-02T21:30:23.000Z").tz_convert("America/New_York")
    assert ny.strftime("%Y-%m-%d %H:%M:%S") == "2026-11-02 16:30:23"


def test_events_columns_and_forms(events):
    assert list(events.columns) == EVENT_COLS
    assert set(events["form"]) <= set(FORMS)
    assert "4" not in set(events["form"])
    assert (events["filing_date"] >= "2017-01-01").all(), "filings before 2017 are dropped"
    assert len(events) == 3


def test_events_dating_and_items(events):
    row = events.set_index("accession").loc["0000320193-26-000024"]
    assert row["accepted_utc"] == "2026-11-02T21:30:23Z"
    assert row["decision_date"] == "2026-11-03"
    assert row["items"] == "2.02;9.01"
    assert row["cik"] == APPLE_CIK

    morning = events.set_index("accession").loc["0000320193-26-000023"]
    assert morning["decision_date"] == "2026-10-27"

    tenq = events.set_index("accession").loc["0000320193-26-000020"]
    assert tenq["decision_date"] == "2026-08-03"
    assert tenq["items"] == ""


def test_shared_cik_copies_to_each_share_class():
    uni = pd.DataFrame({"ticker": ["GOOG", "GOOGL"], "cik": ["0001652044", "0001652044"]})
    ev = build_events(uni, fetch=fake_fetch)
    assert sorted(ev["ticker"].unique()) == ["GOOG", "GOOGL"]
    assert ev.groupby("accession")["ticker"].nunique().eq(2).all()


@pytest.fixture
def daily(events) -> pd.DataFrame:
    return build_daily(events, ["AAPL"], end="2026-11-20").set_index("date")


def test_daily_shape(daily):
    assert list(daily.reset_index().columns) == DAILY_COLS
    assert len(daily) == len(trading_days(DAILY_FROM, "2026-11-20"))
    assert daily.index[0] == "2018-01-02"


def test_daily_flags_on_event_days(daily):
    nov3 = daily.loc["2026-11-03"]
    assert (nov3["has_8k"], nov3["i_202"], nov3["days_since_202"], nov3["post_earn_5d"]) == (1, 1, 0, 1)
    assert nov3["n_8k_20d"] == 2  # Oct 27 and Nov 3
    oct27 = daily.loc["2026-10-27"]
    assert (oct27["has_8k"], oct27["i_502"], oct27["i_202"]) == (1, 1, 0)
    assert daily.loc["2026-08-03", "has_periodic"] == 1
    assert daily.loc["2026-08-03", "has_8k"] == 0


def test_daily_counters(daily):
    assert daily.loc["2026-11-02", "days_since_8k"] == 4  # Oct 28, 29, 30, Nov 2
    assert daily.loc["2026-11-09", "post_earn_5d"] == 1   # 4 trading days after Nov 3
    assert daily.loc["2026-11-10", "post_earn_5d"] == 0   # 5 trading days after
    assert daily.loc["2018-01-02", "days_since_202"] == 90  # no earnings yet -> cap
    quiet = daily.loc["2026-09-15"]
    assert quiet[["has_8k", "has_periodic", *ITEMS]].sum() == 0


def test_filing_after_end_waits(events):
    d = build_daily(events, ["AAPL"], end="2026-11-02").set_index("date")
    assert d.index[-1] == "2026-11-02"
    assert d["i_202"].sum() == 0  # the Nov 2 after-hours 8-K counts for Nov 3, not yet here


# ---------- Part 2: the real files ----------

@pytest.fixture(scope="module")
def real_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        pytest.skip("data/interim/sec_events.csv not present yet")
    return pd.read_csv(EVENTS_CSV, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def real_daily() -> pd.DataFrame:
    if not DAILY_CSV.exists():
        pytest.skip("data/interim/sec_daily.csv not present yet")
    return pd.read_csv(DAILY_CSV, dtype={"date": str, "ticker": str})


def test_real_events_contract(real_events):
    assert list(real_events.columns) == EVENT_COLS
    for col in ["ticker", "cik", "accession", "form", "accepted_utc", "decision_date"]:
        assert (real_events[col] != "").all(), f"empty {col}"
    assert not real_events.duplicated(["accession", "ticker"]).any()
    assert set(real_events["form"]) <= set(FORMS)
    assert set(real_events["ticker"]) <= set(universe_tickers())
    accepted_ny_day = (pd.to_datetime(real_events["accepted_utc"], utc=True)
                       .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    assert (real_events["decision_date"] >= accepted_ny_day).all(), "a filing is dated before it was accepted"


def test_real_daily_contract(real_daily):
    assert list(real_daily.columns) == DAILY_COLS
    assert not real_daily.duplicated(["date", "ticker"]).any()
    assert real_daily[["date", "ticker"]].notna().all().all()
    days = trading_days(DAILY_FROM, real_daily["date"].max())
    assert len(real_daily) == len(days) * 100, "row count must equal trading days x 100"
    assert set(real_daily["ticker"]) == set(universe_tickers())
    flags = ["has_8k", *ITEMS, "has_periodic", "post_earn_5d"]
    assert real_daily[flags].isin([0, 1]).all().all()
    assert real_daily[["days_since_202", "days_since_8k"]].le(90).all().all()