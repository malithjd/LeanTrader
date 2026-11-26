"""Rankers, metrics and backtest on small synthetic data (no data files needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import pick, rebalance_dates, run_backtest, summarize
from labels import LABEL_COLS
from metrics import bootstrap_ci, daily_ic, ic_diff_ci
from walkforward import BASE_FEATURES, COVID_DROP, PURGE_DAYS, feature_cols, make_folds

SECTORS = ["Tech", "Health", "Fin", "Ind", "Energy"]


def bdays(start: str, end: str) -> list[str]:
    return list(pd.bdate_range(start, end).strftime("%Y-%m-%d"))


def panel(n_days=60, n_names=40, signal=1.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = bdays("2022-01-03", "2023-12-29")[:n_days]
    rows = []
    for d in days:
        y = rng.normal(size=n_names)
        s = signal * y + rng.normal(size=n_names)
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i:02d}", "sector": SECTORS[i % 5],
                         "score": s[i], "fwd_ret_5d_rel": y[i]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ walk-forward
def test_labels_never_features():
    for m in [None, "finbert"]:
        assert not set(feature_cols(m)) & set(LABEL_COLS)
    assert set(BASE_FEATURES) < set(feature_cols("finbert"))


def test_folds_purge_and_order():
    days = bdays("2018-01-01", "2024-12-31")
    for f in make_folds(days, [2022, 2023, 2024]):
        assert max(f.train_dates) < min(f.pred_dates)
        gap = [d for d in days if max(f.train_dates) < d < min(f.pred_dates)]
        assert len(gap) == PURGE_DAYS  # labels of the last 6 days would reach into the fold
        assert all(d[:4] == str(f.year) for d in f.pred_dates)


def test_test_years_refused_without_flag():
    days = bdays("2018-01-01", "2026-06-30")
    with pytest.raises(ValueError):
        make_folds(days, [2025])
    assert make_folds(days, [2025], allow_test=True)[0].year == 2025


def test_drop_covid_only_touches_training():
    days = bdays("2018-01-01", "2022-12-31")
    f = make_folds(days, [2022], drop_covid=True)[0]
    assert not [d for d in f.train_dates if COVID_DROP[0] <= d <= COVID_DROP[1]]
    assert len(f.pred_dates) == len([d for d in days if d.startswith("2022")])


# ------------------------------------------------------------------ metrics
def test_perfect_and_inverse_ic():
    p = panel(signal=1.0)
    p["score"] = p["fwd_ret_5d_rel"]
    assert np.allclose(daily_ic(p), 1.0)
    p["score"] = -p["fwd_ret_5d_rel"]
    assert np.allclose(daily_ic(p), -1.0)


def test_ic_matches_pandas_spearman():
    p = panel(signal=0.5)
    ref = p.groupby("date").apply(lambda g: g["score"].corr(g["fwd_ret_5d_rel"], method="spearman"))
    assert np.allclose(daily_ic(p).to_numpy(), ref.sort_index().to_numpy())


def test_bootstrap_ci_covers_mean_and_diff_sign():
    good, noise = daily_ic(panel(signal=1.0, seed=1)), daily_ic(panel(signal=0.0, seed=2))
    m, lo, hi = bootstrap_ci(good)
    assert lo < m < hi and lo > 0
    d = ic_diff_ci(good, noise)
    assert d["significant"] and d["delta_ic"] > 0


# ------------------------------------------------------------------ backtest
def test_rebalance_is_first_trading_day_of_week():
    days = [d for d in bdays("2024-01-01", "2024-01-31") if d not in ("2024-01-01", "2024-01-15")]  # holidays
    r = rebalance_dates(days)
    assert r[0] == "2024-01-02" and "2024-01-16" in r and len(r) == 5


def test_pick_respects_sector_cap():
    day = pd.DataFrame({"ticker": [f"T{i}" for i in range(20)],
                        "sector": ["Tech"] * 10 + ["Health"] * 10,
                        "score": list(range(40, 30, -1)) + list(range(10, 0, -1))})
    names = pick(day, top_n=10, sector_cap=0.30)
    assert sum(n in {f"T{i}" for i in range(10)} for n in names) == 3
    assert len(names) == 6  # only two sectors, 3 each


def toy_market(n_days=30, n_names=30):
    days = bdays("2024-01-01", "2024-03-29")[:n_days]
    tick = [f"T{i:02d}" for i in range(n_names)]
    growth = {t: 1 + 0.001 * i for i, t in enumerate(tick)}  # higher index grows faster
    prices = pd.DataFrame([{"date": d, "ticker": t, "open": 100 * growth[t] ** k}
                           for k, d in enumerate(days) for t in tick])
    bench = pd.DataFrame({"date": days, "spy_open": [100 * 1.0005 ** k for k in range(len(days))]})
    scores = pd.DataFrame([{"date": d, "ticker": t, "score": i, "sector": SECTORS[i % 5]}
                           for d in days for i, t in enumerate(tick)])
    return days, prices, bench, scores


def test_backtest_costs_and_turnover():
    days, prices, bench, scores = toy_market()
    bt0 = run_backtest(scores, prices, bench, cost_bps=0)
    bt10 = run_backtest(scores, prices, bench, cost_bps=10)
    # same top 10 every week: buy once, then only small trades back to equal weight after drift
    assert bt10["turnover"].iloc[0] == pytest.approx(1.0)
    assert (bt10["names"] == bt10["names"].iloc[0]).all()
    assert (bt10["turnover"].iloc[1:] < 0.01).all()
    assert bt10["cost"].iloc[0] == pytest.approx(10 / 1e4)
    assert bt0["gross"].equals(bt10["gross"])
    assert (bt10["net"].iloc[0] < bt0["net"].iloc[0])
    assert (bt10["max_sector_share"] <= 0.30 + 1e-9).all()
    # picks the 10 fastest growers allowed by the cap, so it beats the equal-weight universe
    assert (bt0["gross"] > bt0["ew_universe"]).all()
    s = summarize(bt10)
    assert s["ann_return"] > s["ew_ann_return"] and s["max_drawdown"] <= 0


def test_backtest_fills_at_next_open():
    days, prices, bench, scores = toy_market()
    bt = run_backtest(scores, prices, bench, cost_bps=0)
    for _, r in bt.iterrows():
        assert days.index(r["entry"]) == days.index(r["decision_date"]) + 1


# ------------------------------------------------------------------ rankers (smoke)
def _toy_table(n_years=(2020, 2022)):
    rng = np.random.default_rng(0)
    days = bdays(f"{n_years[0]}-01-01", f"{n_years[1]}-12-31")
    n = 30
    rows = []
    for d in days:
        x = rng.normal(size=n)
        y = 0.3 * x + rng.normal(size=n)
        for i in range(n):
            rows.append({"date": d, "ticker": f"T{i:02d}", "sector": SECTORS[i % 5],
                         "f1": x[i], "f2": rng.normal(), "fwd_ret_5d_rel": y[i]})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("objective", ["regression", "lambdarank"])
def test_lgbm_learns_a_planted_signal(objective):
    pytest.importorskip("lightgbm")
    import rank_lgbm

    t = _toy_table()
    fold = make_folds(t["date"], [2022])[0]
    cfg = {"objective": objective, "num_leaves": 7, "min_child_samples": 50, "n_estimators": 50}
    pred, _ = rank_lgbm.fit_predict(t, fold, ["f1", "f2"], cfg)
    assert daily_ic(pred).mean() > 0.15


GRU_SCRIPT = """
import torch, rank_gru
from metrics import daily_ic
from walkforward import make_folds
from tests.test_rankers import _toy_table
rank_gru.SEEDS, rank_gru.MAX_EPOCHS = [0], 3
t = _toy_table()
p = rank_gru.Panel(t, ["f1", "f2"])
fold = make_folds(t["date"], [2022])[0]
for cell in ["gru", "lstm"]:
    pred = rank_gru.fit_predict(p, t, fold, {"hidden": 8, "lr": 3e-3, "cell": cell}, torch.device("cpu"))
    assert pred["score"].notna().mean() > 0.9, (cell, pred["score"].notna().mean())
    ic = daily_ic(pred).mean()
    assert ic > 0.1, (cell, ic)
"""


def test_gru_learns_a_planted_signal():
    """GRU and LSTM both learn a planted signal.

    Runs in its own process: on macOS, torch and lightgbm ship clashing OpenMP runtimes
    and torch aborts if lightgbm was imported first (the rankers run as separate scripts)."""
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed (requirements-ml.txt)")
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-c", GRU_SCRIPT], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
