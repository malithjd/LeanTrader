"""Tests for the shared dating rule (dating.py).

Each case is a timestamp in UTC and the trading day it must map to. New York
is UTC-4 in summer (EDT) and UTC-5 in winter (EST), so 4 pm ET is 20:00 UTC in
summer and 21:00 UTC in winter.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dating import decision_date, is_trading_day, trading_days

CASES = [
    # (UTC timestamp, expected decision date, why)
    ("2024-07-01T12:00:00Z", date(2024, 7, 1), "Monday 8 am ET, before the open"),
    ("2024-07-01T19:59:59Z", date(2024, 7, 1), "Monday 3:59:59 pm EDT"),
    ("2024-07-01T20:00:00Z", date(2024, 7, 2), "Monday 4:00 pm EDT exactly -> next day"),
    ("2024-01-02T20:59:00Z", date(2024, 1, 2), "Tuesday 3:59 pm EST (winter)"),
    ("2024-01-02T21:00:00Z", date(2024, 1, 3), "Tuesday 4:00 pm EST exactly -> next day"),
    ("2024-01-02T20:30:00Z", date(2024, 1, 2), "3:30 pm EST is before the cutoff in winter"),
    ("2024-07-02T20:30:00Z", date(2024, 7, 3), "4:30 pm EDT is after the cutoff in summer"),
    ("2024-07-03T03:00:00Z", date(2024, 7, 3), "UTC says Jul 3, ET says Jul 2 11 pm -> after cutoff -> Jul 3"),
    ("2024-07-05T21:00:00Z", date(2024, 7, 8), "Friday 5 pm -> Monday"),
    ("2024-07-06T15:00:00Z", date(2024, 7, 8), "Saturday -> Monday"),
    ("2024-07-07T23:00:00Z", date(2024, 7, 8), "Sunday evening -> Monday"),
    ("2024-07-04T14:00:00Z", date(2024, 7, 5), "Independence Day (closed) -> Jul 5"),
    ("2024-11-27T22:00:00Z", date(2024, 11, 29), "Wed before Thanksgiving, 5 pm -> Fri (Thu closed)"),
    ("2024-11-29T19:30:00Z", date(2024, 11, 29), "Black Friday 2:30 pm: early close at 1 pm, but the rule is 4 pm"),
    ("2018-12-04T22:00:00Z", date(2018, 12, 6), "Dec 5, 2018 closed (President Bush day of mourning)"),
    ("2025-01-08T22:00:00Z", date(2025, 1, 10), "Jan 9, 2025 closed (President Carter day of mourning)"),
    ("2020-03-09T13:00:00Z", date(2020, 3, 9), "COVID crash day still dated normally"),
]


@pytest.mark.parametrize("ts, expected, why", CASES, ids=[c[2] for c in CASES])
def test_known_cases(ts, expected, why):
    assert decision_date(ts) == expected, why


@pytest.mark.parametrize("ts, expected, why", CASES[:3], ids=[c[2] for c in CASES[:3]])
def test_other_timezones_give_same_answer(ts, expected, why):
    eastern = pd.Timestamp(ts).tz_convert("America/New_York")
    tokyo = pd.Timestamp(ts).tz_convert("Asia/Tokyo")
    assert decision_date(eastern) == expected
    assert decision_date(tokyo) == expected


def test_column_input_keeps_index_and_dtype():
    s = pd.Series([c[0] for c in CASES], index=[f"row{i}" for i in range(len(CASES))])
    out = decision_date(s)
    assert isinstance(out, pd.Series)
    assert out.name == "decision_date"
    assert list(out.index) == list(s.index)
    assert out.dtype.kind == "M"
    assert [d.date() for d in out] == [c[1] for c in CASES]


def test_list_input_returns_series():
    out = decision_date([CASES[0][0], CASES[2][0]])
    assert [d.date() for d in out] == [CASES[0][1], CASES[2][1]]


def test_csv_prints_plain_dates(tmp_path):
    df = pd.DataFrame({"published_utc": ["2024-07-01T12:00:00Z", "2024-07-05T21:00:00Z"]})
    df["decision_date"] = decision_date(df["published_utc"])
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    assert path.read_text().splitlines()[1].endswith(",2024-07-01")


def test_naive_timestamp_is_refused():
    with pytest.raises(ValueError, match="no timezone"):
        decision_date("2024-07-01 12:00:00")
    with pytest.raises(ValueError, match="no timezone"):
        decision_date(pd.Series(["2024-07-01 12:00:00", "2024-07-02 12:00:00"]))


def test_missing_timestamp_is_refused():
    with pytest.raises(ValueError, match="Missing"):
        decision_date(pd.Series(["2024-07-01T12:00:00Z", None]))


def test_out_of_window_is_refused():
    with pytest.raises(ValueError, match="calendar window"):
        decision_date("2010-01-04T15:00:00Z")


def test_never_dated_before_it_was_available():
    """Leakage property on 20,000 random timestamps, 2017-2026.

    The decision date is a trading day, never earlier than the item's New York
    date, and equals that date only if the item arrived before 4 pm on a trading day.
    """
    rng = np.random.default_rng(0)
    lo = pd.Timestamp("2017-01-01", tz="UTC").value
    hi = pd.Timestamp("2026-11-01", tz="UTC").value
    ts = pd.Series(pd.to_datetime(rng.integers(lo, hi, 20_000), utc=True))
    out = decision_date(ts)

    local = ts.dt.tz_convert("America/New_York")
    local_day = local.dt.tz_localize(None).dt.normalize()
    sessions = set(trading_days("2017-01-01", "2027-12-31"))

    assert out.isin(sessions).all()
    assert (out >= local_day).all()
    same_day = out == local_day
    assert (local[same_day].dt.hour < 16).all()
    assert local_day[same_day].isin(sessions).all()
    # Never more than a long weekend plus a holiday away.
    assert ((out - local_day).dt.days <= 5).all()


def test_calendar_basics():
    days = trading_days("2024-01-01", "2024-12-31")
    assert len(days) == 252  # NYSE had 252 sessions in 2024
    assert is_trading_day("2024-07-05")
    assert not is_trading_day("2024-07-04")
    assert not is_trading_day("2024-07-06")
