"""The leakage gate (product spec, "Evals -> Leakage gate"). Nothing trains until this passes.

Checks:
    1. decision_date is never before an item's timestamp (SEC now, news when it lands)
    2. no feature uses data after 4 pm ET on its date
    3. label columns are never features
    4. a planted canary (tomorrow's return) gets caught, and no real feature does
plus the model_table.csv schema.

Part 1 runs everywhere on synthetic data. Part 2 needs the real files in data/interim/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dating import MARKET_TZ, decision_date
from factors import FACTOR_COLS, compute_factors
from labels import LABEL_COLS, build_labels
from leakage import MAX_ABS_IC, canary_feature, daily_rank_ic, ic_outliers, lookahead_columns
from merge import (FEATURE_COLS, KEY_COLS, MODEL_COLS, NEWS_COLS, NEWS_COUNT_COLS, NEWS_SENT_COLS,
                   SEC_COLS, build_model_table)
from tests.test_factors_labels import synthetic_prices

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"


def factors_with_canary(prices: pd.DataFrame) -> pd.DataFrame:
    f = compute_factors(prices, start=None)
    return f.merge(canary_feature(prices), on=KEY_COLS, how="left")


# ---------- Part 1: logic, synthetic ----------

def test_labels_are_never_features():
    assert not set(LABEL_COLS) & set(FEATURE_COLS)
    bad_words = ("fwd", "label", "target", "future", "canary")
    assert not [c for c in FEATURE_COLS if any(w in c for w in bad_words)]
    assert MODEL_COLS == [*KEY_COLS, "sector", *FEATURE_COLS, *LABEL_COLS]
    assert len(set(MODEL_COLS)) == len(MODEL_COLS)


def test_lookahead_detector_catches_canary_synthetic():
    p = synthetic_prices()
    cutoff = sorted(p["date"].unique())[300]
    leaks = lookahead_columns(factors_with_canary, p, cutoff, [*FACTOR_COLS, "canary"])
    assert leaks == ["canary"]


def test_ic_detector_catches_canary_synthetic():
    p = synthetic_prices(n_days=300, tickers=[f"T{i:02d}" for i in range(40)], seed=1)
    t = compute_factors(p, start=None).merge(build_labels(p, start=None), on=KEY_COLS)
    t = t.merge(canary_feature(p), on=KEY_COLS)
    assert ic_outliers(t, [*FACTOR_COLS, "canary"]) == ["canary"]


def test_merge_without_news_synthetic():
    p = synthetic_prices()
    tickers = sorted(p["ticker"].unique())
    sec = p[["date", "ticker"]].assign(**{c: 0 for c in SEC_COLS})
    uni = pd.DataFrame({"ticker": tickers, "sector": "Industrials"})
    t = build_model_table(p, compute_factors(p), sec, build_labels(p), uni, news_daily=None)
    assert list(t.columns) == MODEL_COLS
    assert (t[NEWS_COUNT_COLS] == 0).all().all()
    assert t[NEWS_SENT_COLS].isna().all().all()


def test_merge_with_news_synthetic():
    p = synthetic_prices()
    tickers = sorted(p["ticker"].unique())
    sec = p[["date", "ticker"]].assign(**{c: 0 for c in SEC_COLS})
    uni = pd.DataFrame({"ticker": tickers, "sector": "Industrials"})
    day = sorted(p["date"].unique())[-10]
    row = {c: 0 for c in NEWS_COLS} | {"has_news": 1, "n_news": 2, "sent_mean": 0.4, "sent_conf_wtd": 0.5,
                                       "sent_extreme": 0.7, "n_pos": 2, "sent_mean_5d": 0.3, "evt_earnings": 1}
    news = pd.DataFrame([{"date": day, "ticker": "AAA", "model": "llama32_3b_ft", **row},
                         {"date": day, "ticker": "AAA", "model": "finbert", **row, "sent_mean": -0.9}])
    t = build_model_table(p, compute_factors(p), sec, build_labels(p), uni, news_daily=news)
    hit = t[(t["date"] == day) & (t["ticker"] == "AAA")].iloc[0]
    assert hit["has_news"] == 1 and hit["sent_mean"] == pytest.approx(0.4)
    assert t["has_news"].sum() == 1


# ---------- Part 2: the real files ----------

def _need(*names: str) -> list[pd.DataFrame]:
    out = []
    for n in names:
        path = INTERIM / n
        if not path.exists():
            pytest.skip(f"data/interim/{n} not present yet")
        out.append(pd.read_csv(path))
    return out


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return _need("model_table.csv")[0]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _need("prices.csv")[0]


def test_schema(table, prices):
    assert list(table.columns) == MODEL_COLS
    assert not table[KEY_COLS].isna().any().any()
    assert not table.duplicated(KEY_COLS).any()
    grid = prices[prices["date"] >= "2018-01-01"][KEY_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(table[KEY_COLS], grid)
    assert table["sector"].notna().all()
    assert not table[SEC_COLS + NEWS_COUNT_COLS].isna().any().any()


def _check_decision_dates(ts: pd.Series, dd: pd.Series, what: str) -> None:
    ts_utc = pd.to_datetime(ts, utc=True)
    dd = pd.to_datetime(dd)
    local_day = ts_utc.dt.tz_convert(MARKET_TZ).dt.tz_localize(None).dt.normalize()
    assert (dd >= local_day).all(), f"{what}: decision_date before the item's own date"
    after_close = ts_utc.dt.tz_convert(MARKET_TZ).dt.hour >= 16
    assert (dd[after_close] > local_day[after_close]).all(), f"{what}: after-4pm item used the same day"
    pd.testing.assert_series_equal(decision_date(ts).reset_index(drop=True), dd.reset_index(drop=True),
                                   check_names=False, check_dtype=False)


def test_sec_decision_date_never_before_timestamp():
    ev = _need("sec_events.csv")[0]
    _check_decision_dates(ev["accepted_utc"], ev["decision_date"], "sec_events")


def test_news_decision_date_never_before_timestamp():
    news = _need("news_clean.csv")[0]
    _check_decision_dates(news["published_utc"], news["decision_date"], "news_clean")


def test_sec_flags_land_on_decision_date(table):
    """has_8k on (d, ticker) is 1 exactly when an 8-K's decision_date is d: never the filing date."""
    ev = _need("sec_events.csv")[0]
    eight_k = ev[ev["form"].isin(["8-K", "8-K/A"])][["decision_date", "ticker"]].drop_duplicates()
    eight_k = eight_k.rename(columns={"decision_date": "date"}).assign(expected=1)
    t = table[KEY_COLS + ["has_8k"]].merge(eight_k, on=KEY_COLS, how="left").fillna({"expected": 0})
    assert (t["has_8k"] == t["expected"]).all()


def test_factors_have_no_lookahead_real(prices):
    for cutoff in ["2020-03-16", "2024-06-14"]:
        assert lookahead_columns(lambda p: compute_factors(p, start=None), prices, cutoff, FACTOR_COLS) == []


def test_canary_is_caught_real(prices):
    leaks = lookahead_columns(factors_with_canary, prices, "2024-06-14", [*FACTOR_COLS, "canary"])
    assert leaks == ["canary"]


def test_ic_gate_real(table, prices):
    """Real features stay far under MAX_ABS_IC; the planted canary is far over it."""
    t = table.merge(canary_feature(prices), on=KEY_COLS, how="left")
    cols = [c for c in FEATURE_COLS if t[c].nunique() > 1] + ["canary"]
    ic = daily_rank_ic(t, cols)
    print("\nmean daily rank IC:\n" + ic.round(4).to_string())
    flagged = ic[ic.abs() > MAX_ABS_IC].index.tolist()
    assert flagged == ["canary"], f"flagged: {flagged}"
    assert ic["canary"] > 2 * MAX_ABS_IC


def test_news_defaults_when_missing(table):
    if (INTERIM / "news_daily.csv").exists():
        pytest.skip("news_daily.csv present; defaults not in use")
    assert (table[NEWS_COUNT_COLS] == 0).all().all()
    assert table[NEWS_SENT_COLS].isna().all().all()


def test_label_matches_prices_by_hand(table, prices):
    aapl = prices[prices["ticker"] == "AAPL"].reset_index(drop=True)
    i = aapl.index[aapl["date"] == "2023-03-01"][0]
    expect = aapl.loc[i + 6, "open"] / aapl.loc[i + 1, "open"] - 1
    got = table[(table["ticker"] == "AAPL") & (table["date"] == "2023-03-01")]["fwd_ret_5d"].item()
    assert got == pytest.approx(expect)
    day = table[table["date"] == "2023-03-01"]
    assert np.isclose(day["fwd_ret_5d_rel"].mean(), 0.0, atol=1e-12)
