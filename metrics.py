"""Ranking metrics: daily Rank IC, ICIR and block-bootstrap intervals.

Owner: MJ. Spec: product spec, "Evals -> Ranking":
    mean daily Rank IC (Spearman of prediction vs fwd_ret_5d_rel), ICIR, and the change vs
    factors-only with a 95% block-bootstrap interval (20-day blocks).

Labels overlap (5-day returns, one per day), so neighbouring daily ICs are correlated.
The circular block bootstrap resamples 20-day blocks to keep that correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES = 20  # a day needs this many stocks with a prediction and a label
BLOCK = 20
N_BOOT = 2000


def daily_ic(df: pd.DataFrame, pred: str = "score", label: str = "fwd_ret_5d_rel",
             min_names: int = MIN_NAMES) -> pd.Series:
    """Spearman correlation of pred vs label for each date. Index = date."""
    t = df[["date", pred, label]].dropna()
    t = t[t.groupby("date")[pred].transform("size") >= min_names]
    r = t.groupby("date")[[pred, label]].rank()
    r["date"] = t["date"].to_numpy()
    g = r.groupby("date")
    cov = (r[pred] * r[label]).groupby(r["date"]).mean() - g[pred].mean() * g[label].mean()
    ic = cov / (g[pred].std(ddof=0) * g[label].std(ddof=0))
    return ic.rename("ic").sort_index()


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    sd = ic.std(ddof=1)
    return {"n_days": int(len(ic)), "mean_ic": float(ic.mean()), "ic_sd": float(sd),
            "icir": float(ic.mean() / sd) if sd > 0 else np.nan,
            "icir_annual": float(ic.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "hit_rate": float((ic > 0).mean())}


def _block_means(x: np.ndarray, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap of the mean of x."""
    n = len(x)
    if n == 0:
        return np.full(n_boot, np.nan)
    b = min(block, n)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def bootstrap_ci(ic: pd.Series, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) of the mean daily IC."""
    x = ic.dropna().sort_index().to_numpy(dtype=float)
    means = _block_means(x, block, n_boot, np.random.default_rng(seed))
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def ic_diff_ci(ic_a: pd.Series, ic_b: pd.Series, **kw) -> dict:
    """Change in mean daily IC, a minus b, on the dates both have; 95% block-bootstrap CI."""
    both = pd.concat({"a": ic_a, "b": ic_b}, axis=1).dropna().sort_index()
    mean, lo, hi = bootstrap_ci(both["a"] - both["b"], **kw)
    return {"n_days": int(len(both)), "delta_ic": mean, "ci_low": lo, "ci_high": hi,
            "significant": bool(lo > 0 or hi < 0)}
