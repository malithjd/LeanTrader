"""Tests for factors.csv and labels.csv (product spec, "Derived tables").

Part 1 runs everywhere on a synthetic price panel (no data files needed).
Part 2 checks the real CSVs and is skipped until data/interim/{prices,factors,labels}.csv exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factors import COLUMNS as FACTOR_FILE_COLS
from factors import FACTOR_COLS, compute_factors, raw_factors
from labels import COLUMNS as LABEL_FILE_COLS
from labels import build_labels

ROOT = Path(__file__).resolve().parent.parent
PRICES_CSV = ROOT / "data" / "interim" / "prices.csv"
FACTORS_CSV = ROOT / "data" / "interim" / "factors.csv"
LABELS_CSV = ROOT / "data" / "interim" / "labels.csv"


def synthetic_prices(n_days: int = 400, tickers=("AAA", "BBB", "CCC", "DDD", "EEE"), seed: int = 0) -> pd.DataFrame:
    """Random-walk OHLCV in the prices.csv shape. EEE lists late and has one missing day."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    rows = []
    for i, t in enumerate(tickers):
        # Like real markets: open = yesterday's close x overnight move, close = open x intraday move.
        overnight = rng.normal(0, 0.006, n_days)
        intraday = rng.normal(0, 0.012, n_days)
        open_ = 50 * (1 + i) * np.exp(np.cumsum(overnight + intraday) - intraday)
        close = open_ * np.exp(intraday)
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        vol = rng.integers(1_000_000, 5_000_000, n_days)
        rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "ticker": t, "open": open_,
                                  "high": high, "low": low, "close": close, "volume": vol}))
    p = pd.concat(rows, ignore_index=True)
    late = (p["ticker"] == "EEE") & (p["date"] < dates[50].strftime("%Y-%m-%d"))
    gap = (p["ticker"] == "EEE") & (p["date"] == dates[n_days * 3 // 4].strftime("%Y-%m-%d"))
    return p[~late & ~gap].sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic_prices()


# ---------- Part 1: logic on synthetic data ----------

def test_factor_columns_match_spec():
    assert FACTOR_FILE_COLS == ["date", "ticker", "mom_5d", "mom_20d", "mom_60d", "rsi_14", "macd",
                                "vol_ratio_20d", "pos_52w", "vol_20d"]
    assert LABEL_FILE_COLS == ["date", "ticker", "fwd_ret_5d", "fwd_ret_5d_rel"]


def test_factors_do_not_depend_on_future(panel):
    """From sp100-rank: garbage prices from a cutoff on must not move any factor before it."""
    dates = sorted(panel["date"].unique())
    cutoff = dates[300]
    clean = compute_factors(panel, start=None)
    bad = panel.copy()
    future = bad["date"] >= cutoff
    bad.loc[future, ["open", "high", "low", "close"]] = 99999.0
    bad.loc[future, "volume"] = 99_999_999
    dirty = compute_factors(bad, start=None)
    before = clean["date"] < cutoff
    pd.testing.assert_frame_equal(clean[before], dirty[before], check_exact=False, rtol=1e-12)


def test_factors_ignore_open_high_low(panel):
    """Factors use close and volume only, so the next morning's open can't sneak in."""
    moved = panel.copy()
    moved[["open", "high", "low"]] *= 1.37
    pd.testing.assert_frame_equal(compute_factors(panel, start=None), compute_factors(moved, start=None))


def test_factors_ranked_within_date(panel):
    f = compute_factors(panel, start=None)
    vals = f[FACTOR_COLS]
    assert ((vals > 0) & (vals <= 1) | vals.isna()).all().all()
    # On a day where all 5 names have a value, the ranks are 0.2, 0.4, ..., 1.0
    day = f[f["date"] == sorted(f["date"].unique())[-1]]
    assert sorted(day["mom_20d"]) == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_factor_formulas_by_hand(panel):
    """Raw values for one ticker on one day, recomputed from scratch."""
    raw = raw_factors(panel)
    aaa = panel[panel["ticker"] == "AAA"].set_index("date")
    day = aaa.index[200]
    c = aaa["close"]
    i = c.index.get_loc(day)
    ts = pd.Timestamp(day)
    assert raw["mom_20d"].loc[ts, "AAA"] == pytest.approx(c.iloc[i] / c.iloc[i - 20] - 1)
    assert raw["mom_5d"].loc[ts, "AAA"] == pytest.approx(c.iloc[i] / c.iloc[i - 5] - 1)
    v = aaa["volume"].astype(float)
    assert raw["vol_ratio_20d"].loc[ts, "AAA"] == pytest.approx(v.iloc[i] / v.iloc[i - 19:i + 1].mean())
    window = c.iloc[i - 251:i + 1] if i >= 251 else None
    if window is not None:
        expect = (c.iloc[i] - window.min()) / (window.max() - window.min())
        assert raw["pos_52w"].loc[ts, "AAA"] == pytest.approx(expect)
    rets = c.pct_change().iloc[i - 19:i + 1]
    assert raw["vol_20d"].loc[ts, "AAA"] == pytest.approx(rets.std(ddof=0))
    assert 0 <= raw["rsi_14"].loc[ts, "AAA"] <= 100


def test_factors_no_rows_invented_and_gaps_kept(panel):
    f = compute_factors(panel, start=None)
    assert len(f) == len(panel)
    assert not f.duplicated(["date", "ticker"]).any()
    # The late lister has no factor until it has enough history of its own.
    eee = f[f["ticker"] == "EEE"].reset_index(drop=True)
    assert eee.loc[:19, "mom_20d"].isna().all()


def test_labels_open_to_open_by_hand(panel):
    lab = build_labels(panel, start=None)
    aaa = panel[panel["ticker"] == "AAA"].reset_index(drop=True)
    i = 100
    expect = aaa.loc[i + 6, "open"] / aaa.loc[i + 1, "open"] - 1
    got = lab[(lab["ticker"] == "AAA") & (lab["date"] == aaa.loc[i, "date"])]["fwd_ret_5d"].item()
    assert got == pytest.approx(expect)


def test_labels_use_only_future_opens(panel):
    """Labels on day t come from opens t+1 and t+6: closes, and anything on day t, don't move them."""
    moved = panel.copy()
    moved[["high", "low", "close"]] *= 1.5
    moved["volume"] *= 3
    pd.testing.assert_frame_equal(build_labels(panel, start=None), build_labels(moved, start=None))


def test_labels_relative_sums_to_zero(panel):
    lab = build_labels(panel, start=None).dropna()
    assert lab.groupby("date")["fwd_ret_5d_rel"].sum().abs().max() < 1e-12


def test_missing_day_gets_no_factor_or_label(panel):
    """EEE has no row on its gap day: it must not be ranked or averaged in that day."""
    dates = sorted(panel["date"].unique())
    gap_day = dates[len(dates) * 3 // 4]
    f = compute_factors(panel, start=None)
    day = f[f["date"] == gap_day]
    assert "EEE" not in set(day["ticker"])
    assert sorted(day["rsi_14"]) == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_labels_empty_at_the_end_and_around_gaps(panel):
    lab = build_labels(panel, start=None)
    last6 = sorted(lab["date"].unique())[-6:]
    assert lab[lab["date"].isin(last6)]["fwd_ret_5d"].isna().all()
    assert lab[~lab["date"].isin(last6) & (lab["ticker"] == "AAA")]["fwd_ret_5d"].notna().all()


# ---------- Part 2: the real files ----------

@pytest.fixture(scope="module")
def real():
    for p in (PRICES_CSV, FACTORS_CSV, LABELS_CSV):
        if not p.exists():
            pytest.skip(f"{p.relative_to(ROOT)} not present yet")
    return pd.read_csv(PRICES_CSV), pd.read_csv(FACTORS_CSV), pd.read_csv(LABELS_CSV)


def test_real_columns_and_keys(real):
    prices, factors, labels = real
    assert list(factors.columns) == FACTOR_FILE_COLS
    assert list(labels.columns) == LABEL_FILE_COLS
    for df in (factors, labels):
        assert not df[["date", "ticker"]].isna().any().any()
        assert not df.duplicated(["date", "ticker"]).any()
        assert df["date"].min() >= "2018-01-01"
    grid = prices[prices["date"] >= "2018-01-01"][["date", "ticker"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(factors[["date", "ticker"]], grid)
    pd.testing.assert_frame_equal(labels[["date", "ticker"]], grid)


def test_real_factors_in_unit_range(real):
    _, factors, _ = real
    v = factors[FACTOR_COLS]
    assert ((v > 0) & (v <= 1) | v.isna()).all().all()
    assert v.notna().mean().min() > 0.99


def test_real_files_match_a_fresh_build(real):
    prices, factors, labels = real
    pd.testing.assert_frame_equal(compute_factors(prices), factors, check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(build_labels(prices), labels, check_exact=False, rtol=1e-9)


def test_real_labels_missing_only_without_both_opens(real):
    """A label is empty only when the stock has no open on t+1 or t+6 (end of data, late rows)."""
    prices, _, labels = real
    cal = sorted(prices["date"].unique())
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None, cal[i + 6] if i + 6 < len(cal) else None)
           for i, d in enumerate(cal)}
    have = set(zip(prices["date"], prices["ticker"]))
    for d, t in labels.loc[labels["fwd_ret_5d"].isna(), ["date", "ticker"]].itertuples(index=False):
        a, b = nxt[d]
        assert a is None or b is None or (a, t) not in have or (b, t) not in have, (d, t)
