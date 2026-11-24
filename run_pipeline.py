"""One command from news scores to ranker results. Rerun it whenever news changes.

Owner: MJ.

Steps
    1. news_daily.py (Nati), if it exists and news_scores.csv exists: scores -> stock-day features
    2. merge.py: rebuild model_table.csv
    3. Leakage gate (tests/test_model_table.py); stops here if it fails
    4. rank_lgbm.py and rank_gru.py: baseline, then one variant per news model found in
       news_daily.csv (tuned on 2022-2024; --test uses the saved configs instead)
    5. --drop-covid: also the robustness run for every variant
    6. compare.py: news vs baseline Rank IC with 95% intervals

Usage:
    python run_pipeline.py                     # validation, every variant
    python run_pipeline.py --skip-gru          # LightGBM only (fast)
    python run_pipeline.py --drop-covid        # add the COVID-drop runs
    python run_pipeline.py --test --drop-covid # the single test run (Tue Nov 24)
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from walkforward import INTERIM, ROOT, run_gate


def sh(*args: str) -> None:
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def news_models() -> list[str]:
    path = INTERIM / "news_daily.csv"
    if not path.exists():
        return []
    return sorted(pd.read_csv(path, usecols=["model"])["model"].unique())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--drop-covid", action="store_true")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--skip-news-daily", action="store_true")
    args = ap.parse_args()

    if not args.skip_news_daily and (ROOT / "news_daily.py").exists() and (INTERIM / "news_scores.csv").exists():
        sh("news_daily.py")
    sh("merge.py")
    run_gate()

    models = news_models()
    print(f"News models in news_daily.csv: {models or 'none (baseline only)'}")
    rankers = ["rank_lgbm.py"] + ([] if args.skip_gru else ["rank_gru.py"])
    extra = ["--test"] if args.test else []
    for script in rankers:
        for m in [None, *models]:
            news = ["--news-model", m] if m else []
            sh(script, "--skip-gate", *news, *extra)
            if args.drop_covid:
                sh(script, "--skip-gate", "--drop-covid", *news, *extra)
    sh("compare.py", *extra)


if __name__ == "__main__":
    main()
