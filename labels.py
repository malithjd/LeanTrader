"""Labels: the 5-day forward return the ranker predicts.

Owner: MJ. Spec: product spec, "Derived tables" (labels.csv) and "The dating rule".

Input:   data/interim/prices.csv
Output:  data/interim/labels.csv  one row per (date, ticker) in prices.csv from 2018-01-01

Columns:
    fwd_ret_5d      open[t+6] / open[t+1] - 1. We decide after the close on day t,
                    fill at the next open (t+1) and hold 5 trading days.
    fwd_ret_5d_rel  fwd_ret_5d minus that date's average across the stocks that have one.
                    The training target and the Rank IC label. NEVER a feature.

This is the only module allowed to look forward (shift(-n)). Rows whose exit open
doesn't exist yet (the last 6 trading days) or whose stock has a gap in the window
have empty labels; they stay in the file so the grid is complete.

Usage:
    python labels.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factors import START, load_prices, wide

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "data" / "interim" / "labels.csv"
HORIZON = 5  # trading days held
ENTRY_LAG = 1  # fill at the next open

LABEL_COLS = ["fwd_ret_5d", "fwd_ret_5d_rel"]
COLUMNS = ["date", "ticker", *LABEL_COLS]


def build_labels(prices: pd.DataFrame, start: str | None = START) -> pd.DataFrame:
    """labels.csv as a DataFrame."""
    open_ = wide(prices, "open")
    entry = open_.shift(-ENTRY_LAG)  # open t+1
    exit_ = open_.shift(-(ENTRY_LAG + HORIZON))  # open t+6
    # Only stocks with a price on day t get a label (and count in that day's average).
    fwd = (exit_ / entry - 1.0).where(open_.notna())
    rel = fwd.sub(fwd.mean(axis=1, skipna=True), axis=0)

    long = pd.concat(
        {"fwd_ret_5d": fwd.stack(future_stack=True), "fwd_ret_5d_rel": rel.stack(future_stack=True)},
        axis=1,
    ).reset_index()
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")

    keys = prices[["date", "ticker"]].astype({"date": str})
    out = keys.merge(long, on=["date", "ticker"], how="left")
    if start is not None:
        out = out[out["date"] >= start]
    return out[COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    labels = build_labels(load_prices())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(labels):,} rows to {OUT_CSV}")
    empty = labels["fwd_ret_5d"].isna()
    print(f"Rows with no label: {empty.sum():,} (dates {sorted(labels.loc[empty, 'date'].unique())[-6:]} at the end)")


if __name__ == "__main__":
    main()
