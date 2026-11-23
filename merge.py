"""Merge every source into the stock-day table the rankers train on.

Owner: MJ. Spec: product spec, "model_table.csv (MJ)" and "Evals -> Leakage gate".

Inputs (data/interim/):
    prices.csv       the grid: every (date, ticker) row from 2018-01-01
    factors.csv      factors.py
    sec_daily.csv    sec_edgar.py            missing -> 0 (days_since_* -> 90, the cap)
    news_daily.csv   Nati's news_daily.py    optional; one --news-model is used.
                     No file or no row -> has_news = 0, counts 0, sentiment empty.
    labels.csv       labels.py
    configs/universe.csv for sector

Output: data/interim/model_table.csv, columns MODEL_COLS in that order.

Rankers import FEATURE_COLS from here and must never train on LABEL_COLS.
tests/test_model_table.py is the leakage gate: run it before anything trains.

Usage:
    python merge.py                          # news model llama32_3b_ft, if news_daily.csv exists
    python merge.py --news-model finbert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factors import FACTOR_COLS, START
from labels import LABEL_COLS
from universe import load_universe

ROOT = Path(__file__).resolve().parent
INTERIM = ROOT / "data" / "interim"
OUT_CSV = INTERIM / "model_table.csv"
DEFAULT_NEWS_MODEL = "llama32_3b_ft"
NEWS_MODELS = ["finbert", "llama32_3b_zs", "llama32_3b_ft", "gpt_oss_20b", "olmo3_7b_think"]

KEY_COLS = ["date", "ticker"]
META_COLS = ["sector"]  # for the portfolio's sector cap; not a ranker feature
SEC_COLS = ["has_8k", "i_202", "i_502", "i_101", "i_201", "i_701", "i_801", "has_periodic",
            "days_since_202", "days_since_8k", "n_8k_20d", "post_earn_5d"]
SEC_CAP_COLS = ["days_since_202", "days_since_8k"]  # missing means "none in 90 days"
SEC_CAP = 90
NEWS_COUNT_COLS = ["has_news", "n_news", "n_pos", "n_neg",
                   "evt_earnings", "evt_guidance", "evt_analyst", "evt_mna",
                   "evt_legal", "evt_management", "evt_capital"]
NEWS_SENT_COLS = ["sent_mean", "sent_conf_wtd", "sent_extreme", "sent_mean_5d"]
NEWS_COLS = ["has_news", "n_news", "sent_mean", "sent_conf_wtd", "sent_extreme", "n_pos", "n_neg",
             "sent_mean_5d", "evt_earnings", "evt_guidance", "evt_analyst", "evt_mna",
             "evt_legal", "evt_management", "evt_capital"]  # spec order

FEATURE_COLS = [*FACTOR_COLS, *SEC_COLS, *NEWS_COLS]
MODEL_COLS = [*KEY_COLS, *META_COLS, *FEATURE_COLS, *LABEL_COLS]


def _read(name: str) -> pd.DataFrame:
    path = INTERIM / name
    if not path.exists():
        raise SystemExit(f"{path} not found. Build it first (see this module's docstring).")
    return pd.read_csv(path)


def _news_for_model(news_daily: pd.DataFrame | None, model: str) -> pd.DataFrame:
    """One model's news features keyed on (date, ticker); empty frame if there are none."""
    if news_daily is None or news_daily.empty:
        return pd.DataFrame(columns=[*KEY_COLS, *NEWS_COLS])
    n = news_daily[news_daily["model"] == model]
    if n.empty:
        print(f"warn: news_daily has no rows for model {model!r}; news features left as no news")
    if n.duplicated(KEY_COLS).any():
        raise ValueError(f"news_daily has duplicate (date, ticker) rows for model {model!r}")
    return n[[*KEY_COLS, *NEWS_COLS]]


def build_model_table(prices: pd.DataFrame, factors: pd.DataFrame, sec_daily: pd.DataFrame,
                      labels: pd.DataFrame, universe: pd.DataFrame,
                      news_daily: pd.DataFrame | None = None,
                      news_model: str = DEFAULT_NEWS_MODEL, start: str | None = START) -> pd.DataFrame:
    """Left-join every source onto the prices grid. Returns MODEL_COLS."""
    grid = prices[KEY_COLS].astype({"date": str})
    if start is not None:
        grid = grid[grid["date"] >= start]

    def join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], what: str) -> pd.DataFrame:
        right = right[[*KEY_COLS, *cols]].astype({"date": str})
        if right.duplicated(KEY_COLS).any():
            raise ValueError(f"{what} has duplicate (date, ticker) rows")
        return left.merge(right, on=KEY_COLS, how="left", validate="one_to_one")

    t = join(grid, factors, FACTOR_COLS, "factors")
    t = join(t, sec_daily, SEC_COLS, "sec_daily")
    t[SEC_CAP_COLS] = t[SEC_CAP_COLS].fillna(SEC_CAP)
    t[SEC_COLS] = t[SEC_COLS].fillna(0).astype("int64")

    t = join(t, _news_for_model(news_daily, news_model), NEWS_COLS, "news_daily")
    t[NEWS_COUNT_COLS] = t[NEWS_COUNT_COLS].fillna(0).astype("int64")
    t[NEWS_SENT_COLS] = t[NEWS_SENT_COLS].astype("float64")

    t = join(t, labels, LABEL_COLS, "labels")
    t = t.merge(universe[["ticker", "sector"]], on="ticker", how="left", validate="many_to_one")
    return t[MODEL_COLS].sort_values(KEY_COLS).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data/interim/model_table.csv")
    ap.add_argument("--news-model", default=DEFAULT_NEWS_MODEL, choices=NEWS_MODELS)
    args = ap.parse_args()

    news_path = INTERIM / "news_daily.csv"
    news = pd.read_csv(news_path) if news_path.exists() else None
    if news is None:
        print("news_daily.csv not found: has_news = 0 everywhere, sentiment empty. Rerun when it lands.")

    table = build_model_table(
        prices=_read("prices.csv"), factors=_read("factors.csv"), sec_daily=_read("sec_daily.csv"),
        labels=_read("labels.csv"), universe=load_universe(), news_daily=news, news_model=args.news_model,
    )
    table.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(table):,} rows x {table.shape[1]} columns to {OUT_CSV}")
    print(f"Dates {table['date'].min()} to {table['date'].max()}, {table['ticker'].nunique()} tickers")
    print(f"Rows with news ({args.news_model}): {int(table['has_news'].sum()):,}")
    print(f"Rows with a label: {int(table['fwd_ret_5d_rel'].notna().sum()):,}")


if __name__ == "__main__":
    main()
