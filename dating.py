"""The shared dating rule. Every LeanTrader table gets `decision_date` from here.

Rule (product spec, "The dating rule"):
    An item counts for trading day t only if it was available before 4:00 pm
    New York time on day t. Otherwise it counts for the next trading day.
    Weekends, market holidays and special closures roll forward.

Calendar:
    NYSE sessions from `exchange_calendars` (XNYS). A test
    (tests/test_calendar_vs_prices.py) checks that these sessions match the
    dates in data/interim/prices.csv once Bereket's prices exist.

Usage:
    from dating import decision_date
    news["decision_date"] = decision_date(news["published_utc"])   # a column
    decision_date("2024-05-01T19:59:00Z")                           # one value

Do not write your own version of this rule. If something looks wrong, tell MJ.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"
CUTOFF_HOUR = 16  # 4:00 pm New York time; 16:00:00 exactly counts as "after"
CALENDAR_START = "2016-01-01"  # prices start 2017-01-01; one spare year
CALENDAR_END = "2027-12-31"  # covers the forward test


@lru_cache(maxsize=1)
def _sessions() -> pd.DatetimeIndex:
    """All NYSE trading days in the calendar window, as tz-naive midnight dates."""
    # exchange_calendars 4.13 trips a pandas 3.1 deprecation warning internally.
    # It doesn't affect the sessions (the tests check known holidays), so hide it here only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        cal = xcals.get_calendar("XNYS", start=CALENDAR_START, end=CALENDAR_END)
    sessions = pd.DatetimeIndex(cal.sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE trading days between start and end, both included."""
    s = _sessions()
    return s[(s >= pd.Timestamp(start)) & (s <= pd.Timestamp(end))]


def is_trading_day(day) -> bool:
    """True if `day` (a date or date string) is an NYSE trading day."""
    return pd.Timestamp(day).normalize() in _sessions()


def _parse_aware(values: pd.Series) -> pd.Series:
    """Parse to timezone-aware UTC. Refuse naive timestamps and missing values."""
    if values.isna().any():
        bad = values[values.isna()].index[:5].tolist()
        raise ValueError(f"Missing timestamps at index {bad}. Drop or fix them before dating.")

    try:
        parsed = pd.to_datetime(values)
    except (ValueError, TypeError):
        parsed = None  # mixed UTC offsets or mixed naive/aware; checked one by one below

    if parsed is not None and parsed.dtype.kind == "M":
        if parsed.dt.tz is None:
            raise ValueError(
                "Timestamps have no timezone. Store them in UTC with a 'Z' "
                "(e.g. 2024-05-01T13:30:00Z) or pass tz-aware values."
            )
        return parsed.dt.tz_convert("UTC")

    for v in values:
        if pd.Timestamp(v).tzinfo is None:
            raise ValueError(f"Timestamp {v!r} has no timezone. Every timestamp must be tz-aware.")
    return pd.to_datetime(values, utc=True)


def decision_date(ts):
    """First trading day an item can be used, from the time it became available.

    Args:
        ts: one timestamp (str, datetime or pd.Timestamp) or a column of them
            (pd.Series, pd.Index, list, array). Must be timezone-aware; UTC
            strings ending in 'Z' are fine. Naive or missing values raise.

    Returns:
        One timestamp -> datetime.date.
        A column -> pd.Series of datetime64 dates (same index as a Series input),
        named "decision_date". Written with to_csv it prints as YYYY-MM-DD.
    """
    is_column = isinstance(ts, (pd.Series, pd.Index, np.ndarray, list, tuple))
    if isinstance(ts, pd.Series):
        values = ts
    elif is_column:
        values = pd.Series(list(ts))
    else:
        values = pd.Series([ts])

    local = _parse_aware(values).dt.tz_convert(MARKET_TZ)

    # Local calendar date, pushed one day forward if it's 4 pm or later.
    after_cutoff = (local.dt.hour >= CUTOFF_HOUR).astype("int64")
    base = local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_cutoff, unit="D")

    # First trading day on or after that date (rolls weekends and holidays forward).
    sessions = _sessions()
    if (base < sessions[0]).any() or (base > sessions[-1]).any():
        raise ValueError(
            f"Timestamp outside the calendar window {CALENDAR_START} to {CALENDAR_END}. "
            "Widen CALENDAR_START/CALENDAR_END in dating.py."
        )
    pos = np.searchsorted(sessions.values, base.values, side="left")
    out = pd.Series(sessions.values[pos], index=values.index, name="decision_date")

    if is_column:
        return out
    return out.iloc[0].date()


__all__ = ["decision_date", "trading_days", "is_trading_day", "MARKET_TZ", "CUTOFF_HOUR"]
