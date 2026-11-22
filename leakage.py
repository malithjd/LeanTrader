"""Leakage detectors used by the leakage gate (tests/test_model_table.py).

Two independent checks, so a leak has to beat both:

1. lookahead_columns: rebuild features from prices cut off at day T and compare with
   the full build. A feature that only uses data up to the close of its own day
   can't change on days <= T. One that peeks at day T+1 (the canary) does.

2. ic_outliers: mean daily Spearman correlation of each feature with the label.
   Real price factors sit around |0.01-0.05|. A feature far above that is almost
   certainly built from the future.

The canary: tests plant `canary = close[t+1] / close[t] - 1` (tomorrow's return)
and require both checks to flag it and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from factors import wide

MAX_ABS_IC = 0.10  # real factors are far below this; the canary is far above it


def canary_feature(prices: pd.DataFrame) -> pd.DataFrame:
    """Tomorrow's close-to-close return, keyed (date, ticker). Leaks on purpose."""
    close = wide(prices, "close")
    c = (close.shift(-1) / close - 1.0).stack(future_stack=True).rename("canary").reset_index()
    c["date"] = c["date"].dt.strftime("%Y-%m-%d")
    return c


def lookahead_columns(build: Callable[[pd.DataFrame], pd.DataFrame], prices: pd.DataFrame,
                      cutoff: str, feature_cols: list[str], rtol: float = 1e-9) -> list[str]:
    """Feature columns whose values on dates <= cutoff change when prices after cutoff are removed.

    build(prices) must return a frame with date (YYYY-MM-DD), ticker and feature_cols.
    """
    full = build(prices)
    cut = build(prices[prices["date"].astype(str) <= cutoff])
    keys = ["date", "ticker"]
    a = full[full["date"] <= cutoff].set_index(keys).sort_index()
    b = cut[cut["date"] <= cutoff].set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("builds disagree on (date, ticker) rows up to the cutoff")
    leaks = []
    for col in feature_cols:
        x, y = a[col].to_numpy(dtype="float64"), b[col].to_numpy(dtype="float64")
        same = np.isclose(x, y, rtol=rtol, atol=0.0, equal_nan=True)
        if not same.all():
            leaks.append(col)
    return leaks


def daily_rank_ic(table: pd.DataFrame, feature_cols: list[str], label: str = "fwd_ret_5d_rel",
                  min_names: int = 20) -> pd.Series:
    """Mean across dates of the Spearman correlation between each feature and the label."""
    t = table[["date", label, *feature_cols]].dropna(subset=[label])
    ranks = t.groupby("date")[[label, *feature_cols]].rank()
    ranks["date"] = t["date"]
    out = {}
    for col in feature_cols:
        sub = ranks[["date", label, col]].dropna()
        counts = sub.groupby("date")[col].transform("size")
        sub = sub[counts >= min_names]
        if sub.empty or sub[col].nunique() <= 1:
            out[col] = np.nan
            continue
        ic = sub.groupby("date")[[label, col]].corr().xs(col, level=1)[label]
        out[col] = ic.mean()
    return pd.Series(out, name="mean_rank_ic")


def ic_outliers(table: pd.DataFrame, feature_cols: list[str], label: str = "fwd_ret_5d_rel",
                max_abs_ic: float = MAX_ABS_IC) -> list[str]:
    ic = daily_rank_ic(table, feature_cols, label)
    return ic[ic.abs() > max_abs_ic].index.tolist()
