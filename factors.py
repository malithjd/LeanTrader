"""Price factors for the ranker: 8 features per (date, ticker), each ranked 0-1 within its date.

Owner: MJ. Spec: product spec, "Derived tables" (factors.csv).

Input:   data/interim/prices.csv   (Bereket)
Output:  data/interim/factors.csv  one row per (date, ticker) in prices.csv from 2018-01-01

Columns:
    mom_5d, mom_20d, mom_60d  past return over 5, 20, 60 trading days
    rsi_14                    14-day Wilder RSI
    macd                      MACD line (EMA12 - EMA26) minus its 9-day signal line
    vol_ratio_20d             today's volume over its 20-day average
    pos_52w                   where today's close sits in its 252-day close range (0 = low, 1 = high)
    vol_20d                   20-day standard deviation of daily close-to-close returns

Rule: a factor for day t uses data up to the close on day t only. Every window runs
backward (rolling, shift(+n), ewm), never shift(-n). The raw values are then ranked
within each date (rank(pct=True)), so every day counts the same.

Shifts run on a wide (date x ticker) frame over the full prices calendar, so "n days
ago" always means n trading days, and a missing (date, ticker) stays missing: no gap
is filled. Formulas are reused from the sp100-rank feature code (features/technical.py).

Usage:
    python factors.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRICES_CSV = ROOT / "data" / "interim" / "prices.csv"
OUT_CSV = ROOT / "data" / "interim" / "factors.csv"
START = "2018-01-01"  # spec: factors and labels from 2018; prices from 2017 for warm-up

FACTOR_COLS = ["mom_5d", "mom_20d", "mom_60d", "rsi_14", "macd", "vol_ratio_20d", "pos_52w", "vol_20d"]
COLUMNS = ["date", "ticker", *FACTOR_COLS]


def wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    """One column of prices as a date x ticker frame over every date in prices."""
    w = prices.pivot(index="date", columns="ticker", values=col).sort_index()
    w.index = pd.DatetimeIndex(pd.to_datetime(w.index), name="date")
    return w.astype("float64")


# --- raw factors on a wide frame (one column per ticker), all backward-looking ---

def momentum(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1.0


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder RSI (exponential smoothing with alpha = 1/window)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0.0, np.nan))
    return out.mask((avg_loss == 0) & avg_gain.notna(), 100.0)


def macd_hist(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line minus its signal line."""
    line = (close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            - close.ewm(span=slow, adjust=False, min_periods=slow).mean())
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def volume_ratio(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    avg = volume.rolling(window, min_periods=window).mean()
    return volume / avg.replace(0.0, np.nan)


def pos_52w(close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    lo = close.rolling(window, min_periods=window).min()
    hi = close.rolling(window, min_periods=window).max()
    return (close - lo) / (hi - lo).replace(0.0, np.nan)


def realized_vol(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    # fill_method=None: a missing day stays missing instead of being forward-filled.
    return close.pct_change(fill_method=None).rolling(window, min_periods=window).std(ddof=0)


def raw_factors(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Unranked factors, each a date x ticker frame."""
    close = wide(prices, "close")
    volume = wide(prices, "volume")
    raw = {
        "mom_5d": momentum(close, 5),
        "mom_20d": momentum(close, 20),
        "mom_60d": momentum(close, 60),
        "rsi_14": rsi(close, 14),
        "macd": macd_hist(close),
        "vol_ratio_20d": volume_ratio(volume, 20),
        "pos_52w": pos_52w(close, 252),
        "vol_20d": realized_vol(close, 20),
    }
    # No price that day -> no factor that day. ewm (RSI, MACD) would otherwise carry the
    # last value through a missing day and rank a stock that has no row.
    has_row = close.notna()
    return {k: v.where(has_row) for k, v in raw.items()}


def compute_factors(prices: pd.DataFrame, start: str | None = START) -> pd.DataFrame:
    """factors.csv as a DataFrame: ranked 0-1 within each date, one row per prices row.

    A factor that can't be computed yet (too little history) is left empty and
    is not counted in that day's ranking.
    """
    raw = raw_factors(prices)
    long = None
    for name in FACTOR_COLS:
        ranked = raw[name].rank(axis=1, pct=True, method="average")
        s = ranked.stack(future_stack=True).rename(name)
        long = s.to_frame() if long is None else long.join(s)
    long = long.reset_index()
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")

    # Keep exactly the (date, ticker) rows that exist in prices: no invented rows.
    keys = prices[["date", "ticker"]].astype({"date": str})
    out = keys.merge(long, on=["date", "ticker"], how="left")
    if start is not None:
        out = out[out["date"] >= start]
    return out[COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)


def load_prices(path: Path = PRICES_CSV) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run prices.py or copy Bereket's file into data/interim/.")
    return pd.read_csv(path)


def main() -> None:
    prices = load_prices()
    factors = compute_factors(prices)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(factors):,} rows to {OUT_CSV}")
    print("Share of rows with each factor filled:")
    print(factors[FACTOR_COLS].notna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
