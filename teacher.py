"""
LeanTrader teacher labels (Nati) -- spec "teacher_labels.csv".

Three steps, run from the repo root:
  python teacher.py sample                       # pick the 6,000 rows -> data/interim/teacher_sample.csv
  python teacher.py label --limit 50             # pilot: label 50 rows, print projected cost
  python teacher.py label                        # label the rest (resumable, safe to stop/restart)
  python teacher.py finalize                     # -> teacher_labels.csv, hand_check.csv, cost_log.csv row

Needs:  pip install -r requirements.txt     and     GEMINI_API_KEY=... in .env (see .env.example)
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from news_prompt import SYSTEM_PROMPT, TEACHER_EXTRA, TEACHER_SCHEMA, format_input, parse_labels

INTERIM = Path("data/interim")
NEWS = INTERIM / "news_clean.csv"
SAMPLE = INTERIM / "teacher_sample.csv"
RAW = INTERIM / "teacher_raw.jsonl"          # one line per labeled row; the resume log
RUNS = INTERIM / "teacher_runs.jsonl"        # one line per `label` run (wall time for cost_log)
LABELS = INTERIM / "teacher_labels.csv"
HAND_CHECK = INTERIM / "hand_check.csv"
COST_LOG = INTERIM / "cost_log.csv"

SPLITS = {  # name: (first year, last year, rows) -- from the spec
    "train": (2018, 2022, 4000),
    "check": (2023, 2023, 1000),
    "post": (2025, 2025, 1000),
}
SEED = 42
RETRYABLE = {429, 500, 502, 503, 504}

LABEL_COLUMNS = ["news_id", "ticker", "split", "relevant", "event_type", "sentiment",
                 "confidence", "rationale", "teacher_model", "cost_usd"]
COST_COLUMNS = ["run_id", "run_date", "owner", "model", "stage", "n_rows", "n_stock_days",
                "wall_seconds", "gpu_type", "gpu_hourly_usd", "api_cost_usd", "in_tokens", "out_tokens"]


# ---------------------------------------------------------------- sample
def cmd_sample(args):
    """Pick rows per split, spread as evenly as possible across tickers.

    Only keep = 1 rows (the ones the pipeline actually uses). Each article is used once,
    so an article tagging GOOG and GOOGL can't be labeled twice. Within a split, every
    ticker gets its 1st row before any ticker gets its 2nd, and so on.
    """
    if SAMPLE.exists() and not args.force:
        sys.exit(f"{SAMPLE} exists (use --force to redraw; labels already made would no longer match).")
    news = pd.read_csv(NEWS, dtype={"ticker": str}, keep_default_na=False)
    news = news[news["keep"] == 1].copy()
    news["year"] = news["decision_date"].str[:4].astype(int)

    picks = []
    for split, (y0, y1, n) in SPLITS.items():
        pool = news[news["year"].between(y0, y1)].sample(frac=1, random_state=SEED)
        pool = pool.drop_duplicates("news_id")
        pool["turn"] = pool.groupby("ticker").cumcount()
        chosen = pool.sort_values("turn", kind="stable").head(n).assign(split=split)
        if len(chosen) < n:
            print(f"  warning: {split} has only {len(chosen)} rows available (wanted {n})")
        picks.append(chosen)
        print(f"  {split:5s} {len(chosen):>5,} rows, {chosen['ticker'].nunique()} tickers, "
              f"years {sorted(chosen['year'].unique().tolist())}")

    sample = pd.concat(picks).sample(frac=1, random_state=SEED)  # mixed order -> pilot covers all splits
    cols = ["news_id", "ticker", "company", "decision_date", "split", "title", "summary", "source"]
    sample[cols].to_csv(SAMPLE, index=False)
    print(f"wrote {SAMPLE} ({len(sample):,} rows)")


# ---------------------------------------------------------------- label
def done_keys(model):
    """Rows already labeled BY THIS MODEL (switching models after a pilot relabels them)."""
    if not RAW.exists():
        return set()
    with RAW.open(encoding="utf-8") as f:
        return {(r["news_id"], r["ticker"]) for r in map(json.loads, f) if r["teacher_model"] == model}


def label_one(client, row, args):
    """Label one article. Retries rate limits and server errors with growing waits,
    and retries a reply that isn't valid labels twice. Tokens from every attempt are
    counted, so the cost is honest."""
    config = {"system_instruction": SYSTEM_PROMPT + TEACHER_EXTRA,
              "response_mime_type": "application/json",
              "response_json_schema": TEACHER_SCHEMA,
              "temperature": 0}
    if args.thinking_level:
        config["thinking_config"] = {"thinking_level": args.thinking_level}
    elif args.thinking_budget is not None:
        config["thinking_config"] = {"thinking_budget": args.thinking_budget}
    text = format_input(row.ticker, row.company, row.title, row.summary)

    tok_in = tok_out = tok_think = bad_replies = 0
    started = time.time()
    for attempt in range(8):
        try:
            resp = client.models.generate_content(model=args.model, contents=text, config=config)
        except Exception as e:                            # noqa: BLE001 -- SDK raises several types
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code in RETRYABLE:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise                                         # bad key, bad model name, bad config: stop
        u = resp.usage_metadata
        tok_in += getattr(u, "prompt_token_count", 0) or 0
        tok_out += getattr(u, "candidates_token_count", 0) or 0
        tok_think += getattr(u, "thoughts_token_count", 0) or 0
        try:
            labels = parse_labels(resp.text or "")
        except (ValueError, TypeError) as e:
            bad_replies += 1
            if bad_replies <= 2:
                continue
            return None, f"{row.news_id}/{row.ticker}: unusable reply ({e})"
        cost = (tok_in * args.price_in + (tok_out + tok_think) * args.price_out) / 1e6
        return {"news_id": int(row.news_id), "ticker": row.ticker, "split": row.split, **labels,
                "teacher_model": args.model, "in_tokens": tok_in, "out_tokens": tok_out,
                "thought_tokens": tok_think, "cost_usd": round(cost, 8),
                "latency_ms": int((time.time() - started) * 1000),
                "labeled_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}, None
    return None, f"{row.news_id}/{row.ticker}: gave up after repeated rate limits / server errors"


def cmd_label(args, client=None):
    if client is None:
        from google import genai
        client = genai.Client()                           # reads GEMINI_API_KEY
    sample = pd.read_csv(SAMPLE, dtype={"ticker": str}, keep_default_na=False)
    done = done_keys(args.model)
    todo = [r for r in sample.itertuples(index=False) if (int(r.news_id), r.ticker) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done):,} already labeled, labeling {len(todo):,} now with {args.model} "
          f"({args.workers} at a time)")
    if not todo:
        return

    lock, n_ok, errors, run_cost = threading.Lock(), 0, [], 0.0
    started = time.time()
    with RAW.open("a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(label_one, client, r, args) for r in todo]
        try:
            for fut in as_completed(futures):
                rec, err = fut.result()
                with lock:
                    if rec:
                        out.write(json.dumps(rec) + "\n")
                        out.flush()                       # every row is saved the moment it's done
                        n_ok += 1
                        run_cost += rec["cost_usd"]
                    else:
                        errors.append(err)
                    if (n_ok + len(errors)) % 50 == 0:
                        print(f"  {n_ok + len(errors):,}/{len(todo):,} done, ${run_cost:.4f} so far")
        except Exception as e:                            # noqa: BLE001
            for f in futures:
                f.cancel()
            sys.exit(f"Stopped: {type(e).__name__}: {e}\n(rows finished before this are saved)")
    wall = time.time() - started
    with RUNS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"model": args.model, "rows": n_ok, "wall_seconds": round(wall, 1)}) + "\n")

    print(f"\nlabeled {n_ok:,} rows in {wall:.0f}s, {len(errors)} failed (rerun to retry them)")
    for e in errors[:5]:
        print("  ", e)
    report(args)


def report(args):
    raw = pd.read_json(RAW, lines=True)
    raw = raw[raw["teacher_model"] == args.model]
    per_row = raw["cost_usd"].mean()
    total = sum(n for *_, n in SPLITS.values())
    print(f"\n{len(raw):,} rows labeled with {args.model}: ${raw['cost_usd'].sum():.4f} total, "
          f"${per_row:.6f} per row -> about ${per_row * total:.2f} for all {total:,}")
    print(f"tokens per row: {raw['in_tokens'].mean():.0f} in, {raw['out_tokens'].mean():.0f} out, "
          f"{raw['thought_tokens'].mean():.0f} thinking")
    print(f"relevant: {raw['relevant'].mean():.0%}   mean sentiment: {raw['sentiment'].mean():+.2f}")
    print("event types:", raw["event_type"].value_counts().to_dict())


# ---------------------------------------------------------------- finalize
def cmd_finalize(args):
    all_raw = pd.read_json(RAW, lines=True, dtype={"ticker": str})
    model = args.model or all_raw["teacher_model"].mode().iat[0]   # default: the model used most
    raw = all_raw[all_raw["teacher_model"] == model].drop_duplicates(["news_id", "ticker"], keep="last")
    print(f"using labels from {model}")
    sample = pd.read_csv(SAMPLE, dtype={"ticker": str}, keep_default_na=False)
    labels = sample[["news_id", "ticker", "company", "decision_date", "title", "summary"]].merge(
        raw, on=["news_id", "ticker"], how="inner")

    missing = len(sample) - len(labels)
    for split, (*_, n) in SPLITS.items():
        print(f"  {split:5s} {int((labels['split'] == split).sum()):>5,} / {n:,}")
    if missing:
        print(f"  warning: {missing:,} sample rows still unlabeled -- run `label` again")

    labels[LABEL_COLUMNS].to_csv(LABELS, index=False)
    print(f"wrote {LABELS}")

    hand = labels.sample(min(100, len(labels)), random_state=SEED)
    hand = hand[["news_id", "ticker", "company", "split", "title", "summary", "relevant",
                 "event_type", "sentiment", "confidence", "rationale"]].copy()
    for col in ["ok_relevant", "ok_event_type", "ok_sentiment", "tag_wrong", "note"]:
        hand[col] = ""                                    # Nati fills these in by hand
    if HAND_CHECK.exists() and not args.force:
        print(f"{HAND_CHECK} exists, not overwritten (your notes are safe; --force to redo)")
    else:
        hand.to_csv(HAND_CHECK, index=False)
        print(f"wrote {HAND_CHECK} (100 rows to check by hand)")

    runs = pd.read_json(RUNS, lines=True) if RUNS.exists() else pd.DataFrame(columns=["wall_seconds"])
    row = {"run_id": f"teacher_label_{model}", "run_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
           "owner": "nati", "model": model, "stage": "label", "n_rows": len(labels),
           "n_stock_days": labels[["ticker", "decision_date"]].drop_duplicates().shape[0],
           "wall_seconds": round(float(runs["wall_seconds"].sum()), 1), "gpu_type": "",
           # every dollar spent on labeling counts, pilots of other models included
           "gpu_hourly_usd": 0.0, "api_cost_usd": round(float(all_raw["cost_usd"].sum()), 4),
           "in_tokens": int(all_raw["in_tokens"].sum()),
           "out_tokens": int(all_raw["out_tokens"].sum() + all_raw["thought_tokens"].sum())}
    log = pd.read_csv(COST_LOG) if COST_LOG.exists() else pd.DataFrame(columns=COST_COLUMNS)
    log = pd.concat([log[log["run_id"] != row["run_id"]], pd.DataFrame([row])], ignore_index=True)
    log[COST_COLUMNS].to_csv(COST_LOG, index=False)
    print(f"cost_log.csv: {row['run_id']} -> ${row['api_cost_usd']} (includes pilot and retries)")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("label")
    p.add_argument("--model", default="gemini-3.5-flash", help="exact Gemini model id")
    p.add_argument("--price-in", type=float, default=1.50, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=9.00, help="USD per 1M output tokens (incl. thinking)")
    p.add_argument("--thinking-level", help="Gemini 3.x: e.g. low or minimal")
    p.add_argument("--thinking-budget", type=int, help="Gemini 2.5: e.g. 0 to turn thinking off")
    p.add_argument("--workers", type=int, default=8, help="requests in parallel")
    p.add_argument("--limit", type=int, help="label only this many (pilot)")
    p = sub.add_parser("finalize")
    p.add_argument("--model", help="whose labels to use (default: the model with the most rows)")
    p.add_argument("--force", action="store_true", help="also overwrite hand_check.csv")
    a = ap.parse_args()
    load_dotenv()  # reads GEMINI_API_KEY from .env; never overrides a key already in the shell
    {"sample": cmd_sample, "label": cmd_label, "finalize": cmd_finalize}[a.cmd](a)
