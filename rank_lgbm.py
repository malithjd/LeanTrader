"""LightGBM ranker: the main model on the stock-day feature table.

Owner: MJ. Spec: product spec, "Rankers and trading", "Evals -> Ranking / Portfolio / Robustness".

What it does
    1. Runs the leakage gate (tests/test_model_table.py). Nothing trains if it fails.
    2. Tunes on validation years 2022-2024 only: a small grid over two objectives
         regression   predict the day's rank-normalized fwd_ret_5d_rel
         lambdarank   learn the daily order directly (label = the day's return bucket, 0-19)
       The config with the best mean of the three per-year Rank ICs wins and is saved to
       data/results/configs/lgbm_{variant}.json.
    3. Saves validation predictions, Rank IC and backtest tables (evaluate.py).
    --drop-covid reruns the saved config with Feb-Jun 2020 removed from training.
    --test is the single test run (2025 on) with the saved config; it never re-tunes.

Usage:
    python rank_lgbm.py                              # baseline: factors + SEC
    python rank_lgbm.py --news-model finbert         # baseline + FinBERT news
    python rank_lgbm.py --drop-covid                 # robustness run
    python rank_lgbm.py --test [--news-model ...] [--drop-covid]   # ONE time, Tue Nov 24
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluate import evaluate_and_save
from metrics import daily_ic
from walkforward import (RESULTS, TARGET, TEST_YEARS, VAL_YEARS, feature_cols, load_table,
                         make_folds, run_gate, variant_name)

MODEL = "lgbm"
N_BUCKETS = 20
SEED = 0
BASE_PARAMS = {"learning_rate": 0.02, "feature_fraction": 0.8, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": SEED,
               "deterministic": True, "force_row_wise": True, "num_threads": 4}
GRID = {"objective": ["regression", "lambdarank"], "num_leaves": [15, 63],
        "min_child_samples": [500, 2000], "n_estimators": [300, 800]}


def rank_gauss(t: pd.DataFrame) -> pd.Series:
    """The day's target as a centered rank in [-0.5, 0.5]: same scale every day."""
    return t.groupby("date")[TARGET].rank(pct=True) - 0.5


def buckets(t: pd.DataFrame) -> pd.Series:
    """The day's target as an integer bucket 0..N_BUCKETS-1 (lambdarank relevance)."""
    pct = t.groupby("date")[TARGET].rank(pct=True, method="first")
    return np.minimum((pct * N_BUCKETS).astype(int), N_BUCKETS - 1)


def fit_predict(table: pd.DataFrame, fold, feats: list[str], cfg: dict) -> pd.DataFrame:
    train = table[table["date"].isin(set(fold.train_dates))].dropna(subset=[TARGET])
    train = train.sort_values(["date", "ticker"])
    pred = table[table["date"].isin(set(fold.pred_dates))].copy()
    params = {**BASE_PARAMS, **{k: v for k, v in cfg.items() if k != "n_estimators"}}
    if cfg["objective"] == "lambdarank":
        params.update({"metric": "ndcg", "label_gain": list(range(N_BUCKETS))})
        ds = lgb.Dataset(train[feats], label=buckets(train), group=train.groupby("date").size().to_numpy(),
                         free_raw_data=True)
    else:
        ds = lgb.Dataset(train[feats], label=rank_gauss(train), free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=cfg["n_estimators"])
    pred["score"] = booster.predict(pred[feats])
    pred["fold"] = fold.year
    return pred[["date", "ticker", "sector", "fold", "score", TARGET]], booster


def run_folds(table, folds, feats, cfg) -> tuple[pd.DataFrame, list]:
    parts, boosters = [], []
    for f in folds:
        p, b = fit_predict(table, f, feats, cfg)
        parts.append(p)
        boosters.append(b)
    return pd.concat(parts, ignore_index=True), boosters


def mean_fold_ic(preds: pd.DataFrame) -> tuple[float, list[float]]:
    per = [daily_ic(g).mean() for _, g in preds.groupby("fold")]
    return float(np.mean(per)), [float(x) for x in per]


def tune(table, feats, variant) -> dict:
    folds = make_folds(table["date"], VAL_YEARS)
    rows, best = [], None
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, values))
        t0 = time.time()
        preds, _ = run_folds(table, folds, feats, cfg)
        score, per = mean_fold_ic(preds)
        rows.append({**cfg, "mean_fold_ic": score, **{f"ic_{y}": v for y, v in zip(VAL_YEARS, per)},
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {cfg}  mean IC {score:+.4f}  per year {[round(v, 4) for v in per]}")
        if best is None or score > best[0]:
            best = (score, cfg, preds)
    grid = pd.DataFrame(rows).sort_values("mean_fold_ic", ascending=False)
    (RESULTS / "configs").mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS / f"tuning_{MODEL}_{variant}.csv", index=False)
    cfg = best[1]
    (RESULTS / "configs" / f"{MODEL}_{variant}.json").write_text(json.dumps(cfg, indent=2))
    print(f"Best config: {cfg} (mean fold IC {best[0]:+.4f})")
    return {"cfg": cfg, "preds": best[2]}


def load_cfg(variant: str) -> dict:
    path = RESULTS / "configs" / f"{MODEL}_{variant}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: tune first (python rank_lgbm.py without --test)")
    return json.loads(path.read_text())


def importance(boosters, feats) -> pd.DataFrame:
    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    return pd.DataFrame({"feature": feats, "gain": gain / gain.sum()}).sort_values("gain", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--news-model", default=None, help="add this model's news features (news_daily.csv)")
    ap.add_argument("--drop-covid", action="store_true", help="drop Feb-Jun 2020 from training")
    ap.add_argument("--test", action="store_true", help="the single test run, 2025 on, saved config")
    ap.add_argument("--skip-gate", action="store_true", help="tests only")
    args = ap.parse_args()

    if not args.skip_gate:
        run_gate()
    variant = variant_name(args.news_model)
    feats = feature_cols(args.news_model)
    table = load_table(args.news_model)
    run = "nocovid" if args.drop_covid else "main"
    print(f"{MODEL} {variant} run={run}: {len(table):,} rows, {len(feats)} features")

    if args.test:
        cfg = load_cfg(variant)
        folds = make_folds(table["date"], TEST_YEARS, drop_covid=args.drop_covid, allow_test=True)
        preds, boosters = run_folds(table, folds, feats, cfg)
        stage = "test"
    elif args.drop_covid:
        cfg = load_cfg(variant)
        preds, boosters = run_folds(table, make_folds(table["date"], VAL_YEARS, drop_covid=True), feats, cfg)
        stage = "val"
    else:
        cfg = tune(table, feats, variant)["cfg"]
        preds, boosters = run_folds(table, make_folds(table["date"], VAL_YEARS), feats, cfg)
        stage = "val"

    out = evaluate_and_save(preds, stage, MODEL, variant, run)
    importance(boosters, feats).to_csv(RESULTS / f"importance_{stage}_{MODEL}_{variant}_{run}.csv", index=False)
    print(out["ic"].round(4).to_string(index=False))
    print(out["portfolio"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
