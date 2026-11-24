"""Score a ranker's predictions the same way for every model and variant.

Owner: MJ. Used by rank_lgbm.py, rank_gru.py and compare.py.

Inputs: a predictions frame (date, ticker, score, fwd_ret_5d_rel, sector, fold).
Outputs (data/results/):
    preds_{stage}_{model}_{variant}.csv   the predictions
    ic_{stage}.csv                         one row per (model, variant, run, period): Rank IC summary
    portfolio_{stage}.csv                  one row per (model, variant, run, cost_bps): backtest summary
Rows for the same (model, variant, run) are replaced, so reruns don't duplicate.
"""

from __future__ import annotations

import pandas as pd

from backtest import run_backtest, summarize
from metrics import bootstrap_ci, daily_ic, ic_summary
from walkforward import INTERIM, RESULTS

COSTS = [5.0, 10.0, 20.0]


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(INTERIM / "prices.csv", usecols=["date", "ticker", "open"])
    bench = pd.read_csv(INTERIM / "benchmark.csv", usecols=["date", "spy_open"])
    return prices, bench


def ic_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Rank IC per fold year and over all folds, with a 95% block-bootstrap CI."""
    ic = daily_ic(preds)
    years = pd.Series(ic.index.str[:4], index=ic.index)
    rows = []
    for period, sub in [("all", ic), *[(y, ic[years == y]) for y in sorted(years.unique())]]:
        s = ic_summary(sub)
        _, lo, hi = bootstrap_ci(sub)
        rows.append({"period": period, **s, "ic_ci_low": lo, "ic_ci_high": hi})
    return pd.DataFrame(rows)


def portfolio_table(preds: pd.DataFrame, prices: pd.DataFrame, bench: pd.DataFrame,
                    costs: list[float] = COSTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest summary at each cost, plus the 10 bps period-by-period record."""
    rows, record = [], None
    for c in costs:
        bt = run_backtest(preds[["date", "ticker", "score", "sector"]], prices, bench, cost_bps=c)
        rows.append({"cost_bps": c, **summarize(bt)})
        if c == 10.0:
            record = bt
    return pd.DataFrame(rows), record


def _upsert(path, new: pd.DataFrame, keys: dict) -> None:
    new = new.assign(**keys)[[*keys, *new.columns]]
    if path.exists():
        old = pd.read_csv(path, dtype={"period": str})
        mask = pd.Series(True, index=old.index)
        for k, v in keys.items():
            mask &= old[k].astype(str) == str(v)
        new = pd.concat([old[~mask], new], ignore_index=True)
    new.to_csv(path, index=False)


def evaluate_and_save(preds: pd.DataFrame, stage: str, model: str, variant: str, run: str = "main",
                      prices: pd.DataFrame | None = None, bench: pd.DataFrame | None = None) -> dict:
    """Write preds + IC + portfolio tables for one run. Returns the two summary frames."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if prices is None or bench is None:
        prices, bench = load_prices()
    tag = f"{stage}_{model}_{variant}" + ("" if run == "main" else f"_{run}")
    preds.to_csv(RESULTS / f"preds_{tag}.csv", index=False)
    ic = ic_table(preds)
    port, record = portfolio_table(preds, prices, bench)
    if record is not None:
        record.to_csv(RESULTS / f"trades_{tag}.csv", index=False)
    keys = {"model": model, "variant": variant, "run": run}
    _upsert(RESULTS / f"ic_{stage}.csv", ic, keys)
    _upsert(RESULTS / f"portfolio_{stage}.csv", port, keys)
    return {"ic": ic, "portfolio": port}
