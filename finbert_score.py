"""Score every kept news row with FinBERT (ProsusAI/finbert).

Owner: Bereket. Spec: product spec, Task 3 (news_scores.csv, model = 'finbert').

FinBERT is a 3-class sentiment classifier (positive / negative / neutral).
Per the spec: FinBERT fills `sentiment` and `confidence` only; `event_type`
is left empty and `relevant` is forced to 1.

Input:
    data/interim/news_clean.csv  (Nati's table; only keep == 1 rows are scored)

Outputs:
    data/interim/news_scores.csv  one row per (news_id, ticker, model='finbert'),
                                  appended if the file already has other models.
    data/interim/cost_log.csv     one row per run, appended.

Usage:
    python finbert_score.py                     # score all keep==1 rows
    python finbert_score.py --limit 200         # smoke test
    python finbert_score.py --input path.csv    # custom input (used by tests)
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
import time
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "data" / "interim" / "news_clean.csv"
OUT_PATH = ROOT / "data" / "interim" / "news_scores.csv"
COST_PATH = ROOT / "data" / "interim" / "cost_log.csv"
MODEL_ID = "ProsusAI/finbert"
MODEL_TAG = "finbert"
MAX_LEN = 256
BATCH_SIZE = 32
CHECKPOINT_EVERY = 5000  # append partial rows to news_scores.csv this often

SCORE_COLS = [
    "news_id", "ticker", "model",
    "relevant", "event_type", "sentiment", "confidence",
    "valid_json", "latency_ms", "in_tokens", "out_tokens",
]
COST_COLS = [
    "run_id", "run_date", "owner", "model", "stage",
    "n_rows", "n_stock_days", "wall_seconds",
    "gpu_type", "gpu_hourly_usd", "api_cost_usd",
    "in_tokens", "out_tokens",
]


def _load_news(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Waiting on Nati's news_clean.csv "
            "(spec: Task 'Pull Alpaca news ... first cut').")
    df = pd.read_csv(path)
    missing = {"news_id", "ticker", "title", "summary", "keep"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    df = df[df["keep"] == 1].copy()
    df["summary"] = df["summary"].fillna("")
    df["title"] = df["title"].fillna("")
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def _build_text(df: pd.DataFrame) -> list[str]:
    return [f"{t}. {s}".strip(". ") for t, s in zip(df["title"], df["summary"])]


def _scores(texts: list[str]) -> tuple[list[float], list[float], list[int], float, str]:
    """Returns (sentiment, confidence, in_tokens_per_row, total_wall_seconds, device)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(device)

    # ProsusAI/finbert label order is {0: positive, 1: negative, 2: neutral}
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    i_pos = next(i for i, lab in id2label.items() if lab == "positive")
    i_neg = next(i for i, lab in id2label.items() if lab == "negative")
    i_neu = next(i for i, lab in id2label.items() if lab == "neutral")

    sentiments: list[float] = []
    confidences: list[float] = []
    in_toks: list[int] = []
    n = len(texts)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            mask = enc["attention_mask"].cpu().numpy()
            for row, m in zip(probs, mask):
                sent = float(row[i_pos] - row[i_neg])
                conf = float(max(row[i_pos], row[i_neg], row[i_neu]))
                sentiments.append(sent)
                confidences.append(conf)
                in_toks.append(int(m.sum()))
            if (start // BATCH_SIZE) % 50 == 0:
                done = min(start + BATCH_SIZE, n)
                rate = done / max(time.perf_counter() - t0, 1e-6)
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  [{done:>6d}/{n}]  {rate:5.1f} rows/s  eta {eta/60:5.1f} min", flush=True)
    return sentiments, confidences, in_toks, time.perf_counter() - t0, device


def _upsert_scores(new_rows: pd.DataFrame) -> int:
    """Write news_scores.csv, replacing any existing finbert rows."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH)
        old = old[old["model"] != MODEL_TAG]
        combined = pd.concat([old, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined[SCORE_COLS]
    combined.to_csv(OUT_PATH, index=False)
    return len(combined)


def _append_cost(row: dict) -> None:
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])[COST_COLS]
    df.to_csv(COST_PATH, mode="a", header=not COST_PATH.exists(), index=False)


def _done_keys(path: Path) -> set[tuple[int, str]]:
    """Return (news_id, ticker) pairs already scored for MODEL_TAG in path."""
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=["news_id", "ticker", "model"])
    except Exception:
        return set()
    done = done[done["model"] == MODEL_TAG]
    return set(zip(done["news_id"].astype("int64"), done["ticker"].astype(str)))


def _append_chunk(rows: list[dict]) -> None:
    """Append a chunk of already-scored rows to news_scores.csv."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=SCORE_COLS)
    df.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)


def score(input_path: Path = IN_PATH, limit: int | None = None) -> pd.DataFrame:
    df = _load_news(input_path, limit)
    if df.empty:
        print("No rows to score (news_clean.csv has no keep==1 rows).")
        return df

    # Resume: skip rows we've already scored for this model in a previous run.
    done = _done_keys(OUT_PATH)
    if done:
        key = list(zip(df["news_id"].astype("int64"), df["ticker"].astype(str)))
        mask = [k not in done for k in key]
        skipped = len(df) - sum(mask)
        if skipped:
            print(f"Resuming: {skipped:,} rows already scored, {sum(mask):,} to go.")
        df = df[mask].reset_index(drop=True)
        if df.empty:
            print("Everything already scored.")
            return df

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    i_pos = next(i for i, lab in id2label.items() if lab == "positive")
    i_neg = next(i for i, lab in id2label.items() if lab == "negative")
    i_neu = next(i for i, lab in id2label.items() if lab == "neutral")

    texts = _build_text(df)
    n = len(texts)
    buf: list[dict] = []
    total_in_toks = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            mask = enc["attention_mask"].cpu().numpy()
            for j, (row, m) in enumerate(zip(probs, mask)):
                idx = start + j
                buf.append({
                    "news_id": int(df.iat[idx, df.columns.get_loc("news_id")]),
                    "ticker": str(df.iat[idx, df.columns.get_loc("ticker")]),
                    "model": MODEL_TAG,
                    "relevant": 1,
                    "event_type": "",
                    "sentiment": float(row[i_pos] - row[i_neg]),
                    "confidence": float(max(row[i_pos], row[i_neg], row[i_neu])),
                    "valid_json": 1,
                    "latency_ms": 0,  # filled below
                    "in_tokens": int(m.sum()),
                    "out_tokens": 0,
                })
                total_in_toks += int(m.sum())
            if len(buf) >= CHECKPOINT_EVERY:
                _append_chunk(buf)
                buf = []
            if (start // BATCH_SIZE) % 50 == 0:
                done_n = min(start + BATCH_SIZE, n)
                rate = done_n / max(time.perf_counter() - t0, 1e-6)
                eta = (n - done_n) / rate if rate > 0 else 0
                print(f"  [{done_n:>6d}/{n}]  {rate:5.1f} rows/s  eta {eta/60:5.1f} min", flush=True)
    if buf:
        _append_chunk(buf)

    wall = time.perf_counter() - t0
    # Fill latency_ms as a flat per-row average, then rewrite file.
    scored = pd.read_csv(OUT_PATH)
    per_row = max(1, int(wall * 1000 / max(n, 1)))
    scored.loc[scored["model"] == MODEL_TAG, "latency_ms"] = per_row
    scored[SCORE_COLS].to_csv(OUT_PATH, index=False)
    _append_cost({
        "run_id": uuid.uuid4().hex[:8],
        "run_date": date.today().isoformat(),
        "owner": os.environ.get("USER", getpass.getuser()),
        "model": MODEL_TAG,
        "stage": "score",
        "n_rows": n,
        "n_stock_days": df["ticker"].nunique(),
        "wall_seconds": round(wall, 2),
        "gpu_type": device,
        "gpu_hourly_usd": 0.0,
        "api_cost_usd": 0.0,
        "in_tokens": int(total_in_toks),
        "out_tokens": 0,
    })
    rate = n / wall if wall > 0 else math.nan
    print(f"Scored {n:,} rows in {wall:.1f}s ({rate:.1f} rows/s). "
          f"news_scores.csv has {len(scored):,} rows across all models.")
    return scored


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score news with FinBERT.")
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--limit", type=int, default=None, help="Score at most N rows (smoke test).")
    args = ap.parse_args()
    score(args.input, args.limit)
