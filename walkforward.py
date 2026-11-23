"""Walk-forward splits and feature sets shared by every ranker.

Owner: MJ. Spec: product spec, "Rankers and trading" and "Evals -> Ranking".

Folds are calendar years. A fold for year Y trains on every labelled row from
2018-01-01 up to the end of year Y-1 and predicts year Y:

    validation   2022, 2023, 2024   (tuning happens here only)
    test         2025, 2026         (one run; refused unless allow_test=True)

Purge: a label for day t runs from the open of t+1 to the open of t+6, so the
last PURGE_DAYS trading days before a fold have labels that reach into it. They
are dropped from that fold's training rows.

COVID-drop run (robustness): drop_covid=True removes Feb 1 - Jun 30 2020 from
training rows only; predictions still cover every day of the fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from factors import FACTOR_COLS
from labels import ENTRY_LAG, HORIZON, LABEL_COLS
from merge import NEWS_COLS, SEC_COLS

ROOT = Path(__file__).resolve().parent
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "data" / "results"

TARGET = "fwd_ret_5d_rel"
TRAIN_START = "2018-01-01"
VAL_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025, 2026]
TEST_START = "2025-01-01"
PURGE_DAYS = ENTRY_LAG + HORIZON  # 6 trading days
COVID_DROP = ("2020-02-01", "2020-06-30")

BASE_FEATURES = [*FACTOR_COLS, *SEC_COLS]  # factors + SEC: the baseline
NEWS_FEATURES = list(NEWS_COLS)


def feature_cols(news_model: str | None) -> list[str]:
    """Baseline features, plus the news block when a news model is named."""
    cols = BASE_FEATURES + (NEWS_FEATURES if news_model else [])
    assert not set(cols) & set(LABEL_COLS), "a label column slipped into the features"
    return cols


def variant_name(news_model: str | None) -> str:
    return f"base+{news_model}" if news_model else "base"


@dataclass(frozen=True)
class Fold:
    year: int
    train_dates: tuple[str, ...]
    pred_dates: tuple[str, ...]


def make_folds(dates: list[str] | pd.Series, years: list[int], drop_covid: bool = False,
               allow_test: bool = False) -> list[Fold]:
    """One Fold per year. dates = every trading date in the table (YYYY-MM-DD)."""
    if not allow_test and any(y >= int(TEST_START[:4]) for y in years):
        raise ValueError(f"years {years} reach the test period ({TEST_START} on); pass allow_test=True "
                         "only for the single test run")
    days = sorted(set(pd.Series(dates).astype(str)))
    folds = []
    for y in years:
        start, end = f"{y}-01-01", f"{y}-12-31"
        before = [d for d in days if TRAIN_START <= d < start]
        train = before[:-PURGE_DAYS] if len(before) > PURGE_DAYS else []
        if drop_covid:
            train = [d for d in train if not (COVID_DROP[0] <= d <= COVID_DROP[1])]
        pred = [d for d in days if start <= d <= end]
        if not train or not pred:
            continue
        folds.append(Fold(y, tuple(train), tuple(pred)))
    return folds


def run_gate() -> None:
    """The leakage gate: tests/test_model_table.py must pass before anything trains."""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_model_table.py"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("Leakage gate FAILED: no training until tests/test_model_table.py passes")
    print(f"Leakage gate passed ({tail[0] if tail else 'ok'})")


def load_table(news_model: str | None = None) -> pd.DataFrame:
    """The stock-day table. Baseline reads model_table.csv; a news variant rebuilds it in
    memory with that model's rows from news_daily.csv (merge.py writes only one model)."""
    if news_model is None:
        path = INTERIM / "model_table.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found: run python merge.py first")
        t = pd.read_csv(path)
    else:
        from merge import _read, build_model_table
        from universe import load_universe

        news_path = INTERIM / "news_daily.csv"
        if not news_path.exists():
            raise SystemExit(f"{news_path} not found: news variants need Nati's news_daily.py output")
        news = pd.read_csv(news_path)
        if news_model not in set(news["model"]):
            raise SystemExit(f"news_daily.csv has no rows for model {news_model!r}")
        t = build_model_table(prices=_read("prices.csv"), factors=_read("factors.csv"),
                              sec_daily=_read("sec_daily.csv"), labels=_read("labels.csv"),
                              universe=load_universe(), news_daily=news, news_model=news_model)
    t["date"] = t["date"].astype(str)
    return t
