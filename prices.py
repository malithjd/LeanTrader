"""Pull daily OHLCV for the LeanTrader universe and the SPY benchmark.

Owner: Bereket. Spec: product spec, "Prices — Bereket".

Output:
    data/interim/prices.csv          one row per (date, ticker), 2017-01-01 to today
    data/interim/benchmark.csv       one row per date, SPY, same calendar
    data/interim/prices_coverage.csv one row per ticker: first/last trading day, n_rows
    data/raw/yfinance/<SYMBOL>.csv   per-symbol raw cache so nobody pulls twice

Rules (from the spec):
    yfinance with auto_adjust=True.
    Never fill gaps. Tickers listed after 2017-01-01 start on their first trading day.
    Past tickers (BNY was BK, META was FB, BKNG was PCLN, RTX was UTX) are pulled under
    the old symbol too and stitched onto the current ticker. The current ticker wins on
    any overlap day.

Usage:
    python prices.py              # cached, fast after the first run
    python prices.py --no-cache   # force a fresh pull
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from universe import load_universe

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "yfinance"
OUT_DIR = ROOT / "data" / "interim"
START = "2017-01-01"
BENCH = "SPY"
PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
BENCH_COLS = ["date", "spy_open", "spy_close"]
SLEEP_S = 0.15  # be gentle to Yahoo


def _pull_symbol(symbol: str, end: str, use_cache: bool) -> pd.DataFrame:
    """Download OHLCV for one Yahoo symbol, cache raw, return a clean frame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{symbol}.csv"
    if use_cache and cache.exists() and cache.stat().st_size > 0:
        df = pd.read_csv(cache, parse_dates=["Date"])
    else:
        df = yf.download(
            symbol,
            start=START,
            end=end,
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
        if df.empty:
            cache.touch()  # mark as attempted so we don't retry on every run
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.to_csv(cache, index=False)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"])


def _pull_ticker(current: str, past: list[str], end: str, use_cache: bool) -> pd.DataFrame:
    """Pull current symbol plus any past tickers, stitch on date, label under current."""
    frames = []
    for sym in past + [current]:
        if not sym:
            continue
        df = _pull_symbol(sym, end, use_cache=use_cache)
        if df.empty:
            print(f"    warn: no rows for {sym}")
            continue
        frames.append(df)
        time.sleep(SLEEP_S)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Current ticker wins on overlap days (last frame appended).
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    out["ticker"] = current
    return out[PRICE_COLS]


def build(use_cache: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.utcnow().normalize()
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    uni = load_universe()
    frames: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for i, row in uni.iterrows():
        t = row["ticker"]
        past = [p for p in (row["past_tickers"] or "").split(";") if p]
        tag = f"past={past}" if past else ""
        print(f"[{i + 1:3d}/100] {t:<6s} {tag}")
        df = _pull_ticker(t, past, end, use_cache)
        if df.empty:
            coverage.append({"ticker": t, "first_date": "", "last_date": "", "n_rows": 0})
            continue
        frames.append(df)
        coverage.append({
            "ticker": t,
            "first_date": df["date"].min().strftime("%Y-%m-%d"),
            "last_date": df["date"].max().strftime("%Y-%m-%d"),
            "n_rows": int(len(df)),
        })

    prices = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close"]:
        prices[c] = prices[c].astype("float64")
    # auto_adjust multiplies by split/div factors; the product can leave close
    # ~1e-14 outside [low, high]. Clamp so close <= high and low <= close hold.
    prices["close"] = prices["close"].clip(lower=prices["low"], upper=prices["high"])
    prices["volume"] = prices["volume"].round().astype("int64")
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    prices = prices[PRICE_COLS]
    prices.to_csv(OUT_DIR / "prices.csv", index=False)
    print(f"\nWrote {len(prices):,} rows to {OUT_DIR / 'prices.csv'}")

    spy = _pull_symbol(BENCH, end, use_cache)
    if spy.empty:
        raise SystemExit("Could not pull SPY; benchmark.csv not written.")
    bench = pd.DataFrame({
        "date": spy["date"].dt.strftime("%Y-%m-%d"),
        "spy_open": spy["open"].astype("float64"),
        "spy_close": spy["close"].astype("float64"),
    })[BENCH_COLS].sort_values("date").reset_index(drop=True)
    bench.to_csv(OUT_DIR / "benchmark.csv", index=False)
    print(f"Wrote {len(bench):,} rows to {OUT_DIR / 'benchmark.csv'}")

    cov = pd.DataFrame(coverage)
    cov.to_csv(OUT_DIR / "prices_coverage.csv", index=False)
    empty = cov[cov["n_rows"] == 0]
    late = cov[(cov["n_rows"] > 0) & (cov["first_date"] > START)].sort_values("first_date")
    print("\n--- Coverage note ---")
    print(f"Tickers with no data: {empty['ticker'].tolist() or 'none'}")
    print(f"Tickers whose first trading day is after {START}:")
    if late.empty:
        print("  none")
    else:
        for _, r in late.iterrows():
            print(f"  {r['ticker']:<6s} first day {r['first_date']}  ({r['n_rows']} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pull OHLCV for the universe + SPY.")
    ap.add_argument("--no-cache", action="store_true", help="Force a fresh pull, ignore the raw cache")
    build(use_cache=not ap.parse_args().no_cache)
