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


def _scores(texts: list[str]) -> tuple[list[float], list[float], list[int], float]:
    """Returns (sentiment, confidence, in_tokens_per_row, total_wall_seconds)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    # ProsusAI/finbert label order is {0: positive, 1: negative, 2: neutral}
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    i_pos = next(i for i, lab in id2label.items() if lab == "positive")
    i_neg = next(i for i, lab in id2label.items() if lab == "negative")
    i_neu = next(i for i, lab in id2label.items() if lab == "neutral")

    sentiments: list[float] = []
    confidences: list[float] = []
    in_toks: list[int] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row, mask in zip(probs, enc["attention_mask"].cpu().numpy()):
                sent = float(row[i_pos] - row[i_neg])
                conf = float(max(row[i_pos], row[i_neg], row[i_neu]))
                sentiments.append(sent)
                confidences.append(conf)
                in_toks.append(int(mask.sum()))
    return sentiments, confidences, in_toks, time.perf_counter() - t0


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


def score(input_path: Path = IN_PATH, limit: int | None = None) -> pd.DataFrame:
    df = _load_news(input_path, limit)
    if df.empty:
        print("No rows to score (news_clean.csv has no keep==1 rows).")
        return df

    texts = _build_text(df)
    sentiments, confidences, in_toks, wall = _scores(texts)
    total_latency_ms = int(wall * 1000)
    per_row_latency = max(1, total_latency_ms // len(df))

    out = pd.DataFrame({
        "news_id": df["news_id"].astype("int64"),
        "ticker": df["ticker"].astype(str),
        "model": MODEL_TAG,
        "relevant": 1,
        "event_type": "",
        "sentiment": sentiments,
        "confidence": confidences,
        "valid_json": 1,
        "latency_ms": per_row_latency,
        "in_tokens": in_toks,
        "out_tokens": 0,
    })

    n = _upsert_scores(out)
    _append_cost({
        "run_id": uuid.uuid4().hex[:8],
        "run_date": date.today().isoformat(),
        "owner": os.environ.get("USER", getpass.getuser()),
        "model": MODEL_TAG,
        "stage": "score",
        "n_rows": len(df),
        "n_stock_days": df.drop_duplicates(["ticker"]).shape[0],
        "wall_seconds": round(wall, 2),
        "gpu_type": "cpu",
        "gpu_hourly_usd": 0.0,
        "api_cost_usd": 0.0,
        "in_tokens": int(sum(in_toks)),
        "out_tokens": 0,
    })
    rate = len(df) / wall if wall > 0 else math.nan
    print(f"Scored {len(df):,} rows in {wall:.1f}s ({rate:.1f} rows/s). "
          f"news_scores.csv now has {n:,} rows across all models.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score news with FinBERT.")
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--limit", type=int, default=None, help="Score at most N rows (smoke test).")
    args = ap.parse_args()
    score(args.input, args.limit)
