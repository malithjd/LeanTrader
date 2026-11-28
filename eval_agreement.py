"""Compute agreement metrics between a scored model and the teacher labels.

Owner: Bereket. Spec: product spec, Task 5 (QLoRA agreement on 2023 check set).

Reads:
    data/interim/teacher_labels.csv   (ground truth, from Nati)
    data/interim/news_scores.csv      (model predictions, filtered by --model)

Writes a one-row summary to stdout plus (optionally) appends to eval_agreement.csv.

Usage:
    python eval_agreement.py --model llama32_3b_ft --split check
    python eval_agreement.py --model finbert --split check   # sentiment only
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def pearson(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    return float(a.corr(b))


def macro_f1(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = ((y_pred == lab) & (y_true == lab)).sum()
        fp = ((y_pred == lab) & (y_true != lab)).sum()
        fn = ((y_pred != lab) & (y_true == lab)).sum()
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. llama32_3b_ft, llama32_3b_zs, finbert")
    ap.add_argument("--split", default="check", choices=["check", "post", "both"],
                    help="which teacher split to evaluate on")
    ap.add_argument("--labels", type=Path, default=Path("data/interim/teacher_labels.csv"))
    ap.add_argument("--scores", type=Path, default=Path("data/interim/news_scores.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/eval_agreement.csv"))
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    scores = pd.read_csv(args.scores)
    scores = scores[scores["model"] == args.model]
    if args.split != "both":
        labels = labels[labels["split"] == args.split]

    m = labels.merge(
        scores, on=["news_id", "ticker"], suffixes=("_true", "_pred"), how="inner"
    )
    if m.empty:
        print(f"No overlap between {args.model} scores and {args.split} split in teacher_labels.")
        return

    n = len(m)
    valid = m["valid_json"].mean() if "valid_json" in m.columns else float("nan")
    sent_r = pearson(m["sentiment_true"], m["sentiment_pred"])
    rel_acc = float((m["relevant_true"] == m["relevant_pred"]).mean()) \
        if "relevant_true" in m.columns and "relevant_pred" in m.columns else float("nan")
    if "event_type_true" in m.columns and "event_type_pred" in m.columns:
        all_events = sorted(set(m["event_type_true"].dropna().unique())
                            | set(m["event_type_pred"].dropna().unique()))
        evt_f1 = macro_f1(m["event_type_true"].fillna(""), m["event_type_pred"].fillna(""), all_events)
    else:
        evt_f1 = float("nan")

    print(f"=== {args.model}  on split={args.split}  (n={n}) ===")
    print(f"  Valid-JSON share            : {valid:.3f}")
    print(f"  Sentiment Pearson (spec ≥.80): {sent_r:+.3f}")
    print(f"  Event-type macro F1 (spec ≥.70): {evt_f1:.3f}")
    print(f"  Relevance accuracy           : {rel_acc:.3f}")

    row = pd.DataFrame([{
        "model": args.model, "split": args.split, "n": n,
        "valid_json": round(valid, 4),
        "sentiment_pearson": round(sent_r, 4),
        "event_f1": round(evt_f1, 4),
        "relevance_accuracy": round(rel_acc, 4),
    }])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
