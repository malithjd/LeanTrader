"""GRU ranker (and LSTM for comparison): reads each stock's last 60 trading days of features.

Owner: MJ. Spec: product spec, "Rankers and trading" (GRU over a 60-day window, same
walk-forward setup as LightGBM). --cell lstm swaps the GRU for an LSTM with everything
else identical (features, loss, folds, seeds, grid), so the two differ only in the cell.

Model
    GRU or LSTM (features -> hidden) over the 60-day window, last hidden state -> linear -> score.
    Loss: per day, minus the Pearson correlation between the scores and the day's centered
    target rank (a smooth stand-in for Rank IC). A batch is several days of 100 stocks.
    Features are clipped at the training 1st/99th percentiles and z-scored with training
    statistics only; missing values and days a stock wasn't listed become 0.
    A stock-day is scored only if the stock has a full 60-day window.

Training per fold
    The last INNER_VAL_DAYS of the fold's training dates (after a 6-day purge) pick the
    epoch (early stopping); the score is the mean over SEEDS runs. A small grid on the
    validation years picks hidden size and learning rate, saved to
    data/results/configs/{gru|lstm}_{variant}.json.

Usage (same flags as rank_lgbm.py):
    python rank_gru.py [--news-model finbert] [--drop-covid] [--test]
    python rank_gru.py --cell lstm          # LSTM comparison, same setup
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluate import evaluate_and_save
from metrics import daily_ic
from walkforward import (PURGE_DAYS, RESULTS, TARGET, TEST_YEARS, VAL_YEARS, feature_cols, load_table,
                         make_folds, run_gate, variant_name)

MODEL = "gru"
WINDOW = 60
INNER_VAL_DAYS = 126
DATES_PER_BATCH = 16
MAX_EPOCHS = 20
PATIENCE = 3
SEEDS = [0, 1, 2]
GRID = {"hidden": [32, 64], "lr": [1e-3, 3e-4]}
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


CELLS = {"gru": nn.GRU, "lstm": nn.LSTM}


class RNNRanker(nn.Module):
    """One recurrent layer (GRU or LSTM) over the window; its last hidden state -> score."""

    def __init__(self, n_features: int, hidden: int, cell: str = "gru"):
        super().__init__()
        self.cell = cell
        self.rnn = CELLS[cell](n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [batch, WINDOW, F]
        _, state = self.rnn(x)
        h = state[0] if self.cell == "lstm" else state  # LSTM returns (h, c)
        return self.head(h[-1]).squeeze(-1)


GRURanker = RNNRanker  # old name


class Panel:
    """The table as dense arrays: X[date, ticker, feature], y[date, ticker], present mask."""

    def __init__(self, table: pd.DataFrame, feats: list[str]):
        self.dates = sorted(table["date"].unique())
        self.tickers = sorted(table["ticker"].unique())
        di = {d: i for i, d in enumerate(self.dates)}
        ti = {t: i for i, t in enumerate(self.tickers)}
        T, N, F = len(self.dates), len(self.tickers), len(feats)
        r, c = table["date"].map(di).to_numpy(), table["ticker"].map(ti).to_numpy()
        self.raw = np.full((T, N, F), np.nan, dtype=np.float32)
        self.raw[r, c] = table[feats].to_numpy(dtype=np.float32)
        self.y = np.full((T, N), np.nan, dtype=np.float32)
        self.y[r, c] = table[TARGET].to_numpy(dtype=np.float32)
        self.present = np.zeros((T, N), dtype=bool)
        self.present[r, c] = True
        cum = np.vstack([np.zeros((1, N)), np.cumsum(self.present, axis=0)])
        self.full_window = np.zeros((T, N), dtype=bool)
        self.full_window[WINDOW - 1:] = (cum[WINDOW:] - cum[:-WINDOW]) == WINDOW
        self.index = di
        self.X = None

    def fit_scaler(self, train_idx: np.ndarray) -> None:
        """Clip + z-score with training-date statistics only, then fill gaps with 0."""
        tr = self.raw[train_idx].reshape(-1, self.raw.shape[2])
        lo, hi = np.nanpercentile(tr, 1, axis=0), np.nanpercentile(tr, 99, axis=0)
        clipped = np.clip(tr, lo, hi)
        mu, sd = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        X = (np.clip(self.raw, lo, hi) - mu) / sd
        self.X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    def batch(self, day_idx: list[int], need_label: bool):
        """Windows for every scoreable stock on each day: x [n, WINDOW, F], y [n], day id [n]."""
        xs, ys, gs, cells = [], [], [], []
        for g, i in enumerate(day_idx):
            ok = self.full_window[i] & (~np.isnan(self.y[i]) if need_label else self.present[i])
            cols = np.flatnonzero(ok)
            if len(cols) == 0:
                continue
            xs.append(self.X[i - WINDOW + 1:i + 1, cols].transpose(1, 0, 2))
            ys.append(self.y[i, cols])
            gs.append(np.full(len(cols), g))
            cells.extend((i, c) for c in cols)
        if not xs:
            return None
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), cells


def corr_loss(pred: torch.Tensor, y: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Minus the mean per-day Pearson correlation of pred with the day's centered target rank."""
    losses = []
    for g in torch.unique(groups):
        m = groups == g
        if m.sum() < 10:
            continue
        p, t = pred[m], y[m]
        t = torch.argsort(torch.argsort(t)).float()
        p, t = p - p.mean(), t - t.mean()
        losses.append(-(p * t).sum() / (p.norm() * t.norm() + 1e-8))
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def _days(panel: Panel, dates) -> list[int]:
    return [panel.index[d] for d in dates if d in panel.index and panel.index[d] >= WINDOW - 1]


def _predict(model, panel, day_idx, dev, need_label=False) -> pd.DataFrame:
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(day_idx), DATES_PER_BATCH * 4):
            b = panel.batch(day_idx[k:k + DATES_PER_BATCH * 4], need_label)
            if b is None:
                continue
            x, _, _, cells = b
            s = model(torch.from_numpy(x).to(dev)).cpu().numpy()
            out.append(pd.DataFrame({"date": [panel.dates[i] for i, _ in cells],
                                     "ticker": [panel.tickers[c] for _, c in cells], "score": s}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "ticker", "score"])


def train_one(panel, train_days, val_days, val_labels, cfg, seed, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RNNRanker(panel.X.shape[2], cfg["hidden"], cfg.get("cell", "gru")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    best, best_state, bad = -np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(train_days)
        for k in range(0, len(order), DATES_PER_BATCH):
            b = panel.batch(list(order[k:k + DATES_PER_BATCH]), need_label=True)
            if b is None:
                continue
            x, y, g, _ = b
            loss = corr_loss(model(torch.from_numpy(x).to(dev)), torch.from_numpy(y).to(dev),
                             torch.from_numpy(g).to(dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p = _predict(model, panel, val_days, dev, need_label=True).merge(val_labels, on=["date", "ticker"])
        ic = daily_ic(p).mean()
        if ic > best + 1e-5:
            best, bad = ic, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best)


def _labels(panel, day_idx) -> pd.DataFrame:
    rows = [(panel.dates[i], panel.tickers[c], panel.y[i, c]) for i in day_idx
            for c in np.flatnonzero(~np.isnan(panel.y[i]))]
    return pd.DataFrame(rows, columns=["date", "ticker", TARGET])


def fit_predict(panel, table, fold, cfg, dev) -> pd.DataFrame:
    train_dates = list(fold.train_dates)
    inner_val = train_dates[-INNER_VAL_DAYS:]
    inner_train = train_dates[:-(INNER_VAL_DAYS + PURGE_DAYS)]
    panel.fit_scaler(np.array([panel.index[d] for d in inner_train]))
    tr, va, pr = _days(panel, inner_train), _days(panel, inner_val), _days(panel, fold.pred_dates)
    val_labels = _labels(panel, va)
    scores = []
    for seed in SEEDS:
        model, _ = train_one(panel, tr, va, val_labels, cfg, seed, dev)
        s = _predict(model, panel, pr, dev).set_index(["date", "ticker"])["score"]
        scores.append(s.groupby(level=0).rank(pct=True))  # put seeds on one scale before averaging
    score = pd.concat(scores, axis=1).mean(axis=1).rename("score").reset_index()
    keep = table[table["date"].isin(set(fold.pred_dates))][["date", "ticker", "sector", TARGET]]
    out = keep.merge(score, on=["date", "ticker"], how="left")
    out["fold"] = fold.year
    return out[["date", "ticker", "sector", "fold", "score", TARGET]]


def run_folds(panel, table, folds, cfg, dev) -> pd.DataFrame:
    return pd.concat([fit_predict(panel, table, f, cfg, dev) for f in folds], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--news-model", default=None)
    ap.add_argument("--drop-covid", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--cell", choices=list(CELLS), default=MODEL, help="gru (main) or lstm (comparison)")
    args = ap.parse_args()

    if not args.skip_gate:
        run_gate()
    name = args.cell
    variant = variant_name(args.news_model)
    feats = feature_cols(args.news_model)
    table = load_table(args.news_model)
    panel = Panel(table, feats)
    dev = device()
    run = "nocovid" if args.drop_covid else "main"
    cfg_path = RESULTS / "configs" / f"{name}_{variant}.json"
    print(f"{name} {variant} run={run} on {dev}: {len(panel.dates)} days x {len(panel.tickers)} stocks x {len(feats)} features")

    if args.test or args.drop_covid:
        if not cfg_path.exists():
            raise SystemExit(f"{cfg_path} not found: tune first (python rank_gru.py --cell {name})")
        cfg = {**json.loads(cfg_path.read_text()), "cell": name}
        years, allow = (TEST_YEARS, True) if args.test else (VAL_YEARS, False)
        preds = run_folds(panel, table, make_folds(table["date"], years, args.drop_covid, allow), cfg, dev)
        stage = "test" if args.test else "val"
    else:
        folds = make_folds(table["date"], VAL_YEARS)
        rows, best = [], None
        for values in itertools.product(*GRID.values()):
            cfg = {**dict(zip(GRID, values)), "cell": name}
            t0 = time.time()
            preds = run_folds(panel, table, folds, cfg, dev)
            per = [float(daily_ic(g).mean()) for _, g in preds.groupby("fold")]
            score = float(np.mean(per))
            rows.append({**cfg, "mean_fold_ic": score, **{f"ic_{y}": v for y, v in zip(VAL_YEARS, per)},
                         "seconds": round(time.time() - t0, 1)})
            print(f"  {cfg}  mean IC {score:+.4f}  per year {[round(v, 4) for v in per]}  ({rows[-1]['seconds']}s)")
            if best is None or score > best[0]:
                best = (score, cfg, preds)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("mean_fold_ic", ascending=False).to_csv(
            RESULTS / f"tuning_{name}_{variant}.csv", index=False)
        cfg_path.write_text(json.dumps(best[1], indent=2))
        print(f"Best config: {best[1]} (mean fold IC {best[0]:+.4f})")
        preds, stage = best[2], "val"

    out = evaluate_and_save(preds, stage, name, variant, run)
    print(out["ic"].round(4).to_string(index=False))
    print(out["portfolio"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
