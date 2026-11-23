"""Portfolio backtest: the spec's trading rules applied to any ranker's scores.

Owner: MJ. Spec: product spec, "Rankers and trading" and "Evals -> Portfolio".

Rules:
    - Rebalance weekly: decide after the close on the first trading day of each week (t),
      fill at the next open (t+1), the same timing as the label.
    - Hold the top 10 by score, equal weight, at most 30% in one sector (3 names).
    - Cost: COST_BPS on every dollar bought or sold (10 bps; 5 and 20 for robustness).
      Reported turnover is one-way (half of buys + sells).
    - Between rebalances the weights drift with prices (buy and hold).

Returns are open-to-open from prices.csv (adjusted). SPY uses benchmark.csv spy_open over
the same holding periods. "ew_universe" holds all stocks with a score, equal weight, as a
second reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_N = 10
SECTOR_CAP = 0.30
COST_BPS = 10.0
PERIODS_PER_YEAR = 52


def rebalance_dates(dates: list[str] | pd.Series) -> list[str]:
    """First trading day of each ISO week among the given dates."""
    d = pd.to_datetime(pd.Series(sorted(set(pd.Series(dates).astype(str)))))
    week = d.dt.isocalendar()
    first = d.groupby([week["year"], week["week"]]).min()
    return sorted(first.dt.strftime("%Y-%m-%d"))


def pick(day: pd.DataFrame, top_n: int = TOP_N, sector_cap: float = SECTOR_CAP) -> list[str]:
    """Top names by score, skipping any that would put a sector over the cap."""
    max_per_sector = int(np.floor(sector_cap * top_n + 1e-9))
    chosen, per_sector = [], {}
    for tk, sec in day.sort_values(["score", "ticker"], ascending=[False, True])[["ticker", "sector"]].itertuples(index=False):
        if per_sector.get(sec, 0) >= max_per_sector:
            continue
        chosen.append(tk)
        per_sector[sec] = per_sector.get(sec, 0) + 1
        if len(chosen) == top_n:
            break
    return chosen


def run_backtest(scores: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame,
                 cost_bps: float = COST_BPS, top_n: int = TOP_N, sector_cap: float = SECTOR_CAP) -> pd.DataFrame:
    """One row per holding period.

    scores: date, ticker, score, sector (only dates you are allowed to trade on).
    prices: date, ticker, open.  benchmark: date, spy_open.
    """
    opens = prices.assign(date=prices["date"].astype(str)).pivot(index="date", columns="ticker", values="open").sort_index()
    spy = benchmark.assign(date=benchmark["date"].astype(str)).set_index("date")["spy_open"].sort_index()
    all_days = list(opens.index)
    pos = {d: i for i, d in enumerate(all_days)}
    sc = scores.assign(date=scores["date"].astype(str)).dropna(subset=["score"])
    rebal = [d for d in rebalance_dates(sc["date"]) if d in pos]

    rows, held = [], pd.Series(dtype=float)  # held = drifted weights just before the next trade
    for k, d in enumerate(rebal):
        entry_i = pos[d] + 1
        exit_i = pos[rebal[k + 1]] + 1 if k + 1 < len(rebal) else None
        if exit_i is None or exit_i >= len(all_days) or entry_i >= len(all_days):
            break  # last period has no exit open yet
        entry, exit_ = all_days[entry_i], all_days[exit_i]
        day = sc[sc["date"] == d]
        ok = opens.loc[entry, day["ticker"]].notna() & opens.loc[exit_, day["ticker"]].notna()
        day = day[ok.to_numpy()]
        names = pick(day, top_n, sector_cap)
        if not names:
            continue
        target = pd.Series(1.0 / len(names), index=names)
        traded = target.sub(held, fill_value=0.0).abs().sum()  # buys + sells, share of capital
        turnover = traded / 2 if len(held) else traded  # one-way turnover (reported)
        cost = traded * cost_bps / 1e4  # every buy and every sell pays cost_bps

        gross_by_name = opens.loc[exit_, names] / opens.loc[entry, names]
        gross = float((target * gross_by_name).sum() - 1.0)
        net = (1 - cost) * (1 + gross) - 1
        held = target * gross_by_name / (1 + gross)

        univ = day["ticker"].tolist()
        ew = float((opens.loc[exit_, univ] / opens.loc[entry, univ]).mean() - 1.0)
        spy_ret = float(spy.get(exit_, np.nan) / spy.get(entry, np.nan) - 1.0)
        sectors = day.set_index("ticker").loc[names, "sector"].value_counts()
        rows.append({"decision_date": d, "entry": entry, "exit": exit_, "n_names": len(names),
                     "max_sector_share": float(sectors.max() / len(names)), "turnover": float(turnover),
                     "cost": cost, "gross": gross, "net": net, "spy": spy_ret, "ew_universe": ew,
                     "names": ";".join(names)})
    return pd.DataFrame(rows)


def _ann(r: pd.Series) -> tuple[float, float, float, float]:
    """CAGR, annual vol, Sharpe (rf = 0), max drawdown from per-period returns."""
    r = r.dropna()
    if r.empty:
        return (np.nan,) * 4
    wealth = (1 + r).cumprod()
    years = len(r) / PERIODS_PER_YEAR
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    vol = r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR) if r.std(ddof=1) > 0 else np.nan
    mdd = float((wealth / wealth.cummax() - 1).min())
    return float(cagr), float(vol), float(sharpe), mdd


def summarize(bt: pd.DataFrame) -> dict:
    """Annual return, Sharpe, max drawdown, turnover and excess over SPY, after costs."""
    if bt.empty:
        return {}
    cagr, vol, sharpe, mdd = _ann(bt["net"])
    spy_cagr, _, spy_sharpe, spy_mdd = _ann(bt["spy"])
    ew_cagr, _, ew_sharpe, _ = _ann(bt["ew_universe"])
    return {"start": bt["entry"].iloc[0], "end": bt["exit"].iloc[-1], "n_periods": int(len(bt)),
            "ann_return": cagr, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd,
            "avg_turnover": float(bt["turnover"].iloc[1:].mean()) if len(bt) > 1 else np.nan,
            "spy_ann_return": spy_cagr, "spy_sharpe": spy_sharpe, "spy_max_drawdown": spy_mdd,
            "excess_vs_spy": cagr - spy_cagr, "ew_ann_return": ew_cagr, "ew_sharpe": ew_sharpe,
            "max_sector_share": float(bt["max_sector_share"].max())}
