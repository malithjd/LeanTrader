"""Score news with any of the LeanTrader LLMs through the SHARED prompt and schema.

Owner: Bereket. Spec: product spec, "LLM scoring harness" (Task 4).

Every text model reads the identical system prompt (news_prompt.SYSTEM_PROMPT) and
returns the identical JSON schema (news_prompt.STUDENT_SCHEMA), so agreement and
Rank-IC comparisons between models are apples-to-apples.

Supported models (pass with --model):
    llama32_3b_zs    meta-llama/Llama-3.2-3B-Instruct          zero-shot baseline
    llama32_3b_ft    Llama-3.2-3B-Instruct + LoRA adapter      Bereket's fine-tune
    gpt_oss_20b      openai/gpt-oss-20b                        benchmark zero-shot
    olmo3_7b_think   allenai/OLMo-3-1125-7B-Think              benchmark zero-shot
    qwen_tiny        Qwen/Qwen2.5-0.5B-Instruct                for the smoke test only

Notes:
    - transformers backend; portable (CUDA / MPS / CPU). For full 233k-row
      production scoring, switch to vLLM on CUDA -- the on-disk CSV format is identical.
    - Resumable: on restart, skips rows already written for the same --model.
    - JSON schema is enforced with a strict `parse_labels`; invalid replies are
      retried once and, if still bad, logged with valid_json=0 and zero-valued fields.

Usage:
    # 50-row smoke per the spec (for each of the 3 benchmark models + the fine-tune):
    python score_news.py --model llama32_3b_zs   --limit 50
    python score_news.py --model gpt_oss_20b     --limit 50
    python score_news.py --model olmo3_7b_think  --limit 50
    python score_news.py --model llama32_3b_ft   --limit 50 --adapter path/to/lora

    # Full production run (Task 6, after fine-tune), on CUDA:
    python score_news.py --model llama32_3b_ft --adapter hf://user/llama32-3b-leantrader
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

from news_prompt import STUDENT_SCHEMA, SYSTEM_PROMPT, format_input, parse_labels

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "data" / "interim" / "news_clean.csv"
OUT_PATH = ROOT / "data" / "interim" / "news_scores.csv"
COST_PATH = ROOT / "data" / "interim" / "cost_log.csv"

# Map our short tag -> (HF model id, kwargs). The id stays loadable from the Hub;
# a local path also works ("--base-path ./models/llama-3.2-3b").
MODEL_IDS = {
    "llama32_3b_zs":   "meta-llama/Llama-3.2-3B-Instruct",
    "llama32_3b_ft":   "meta-llama/Llama-3.2-3B-Instruct",    # + --adapter
    "gpt_oss_20b":     "openai/gpt-oss-20b",
    "olmo3_7b_think":  "allenai/OLMo-3-1125-7B-Think",
    "qwen_tiny":       "Qwen/Qwen2.5-0.5B-Instruct",          # smoke test only
}

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

CHECKPOINT_EVERY = 500
MAX_RETRIES = 1  # one retry on invalid JSON, then give up for that row


# --- data IO ----------------------------------------------------------------


def _load_news(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found. Pull news_clean.csv from the team drive.")
    df = pd.read_csv(path, low_memory=False)
    need = {"news_id", "ticker", "title", "summary", "company", "keep"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    df = df[df["keep"] == 1].copy()
    df["title"] = df["title"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["company"] = df["company"].fillna(df["ticker"])
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def _done_keys(path: Path, model_tag: str) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=["news_id", "ticker", "model"])
    except Exception:
        return set()
    done = done[done["model"] == model_tag]
    return set(zip(done["news_id"].astype("int64"), done["ticker"].astype(str)))


def _append_chunk(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=SCORE_COLS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_cost(row: dict) -> None:
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])[COST_COLS]
    df.to_csv(COST_PATH, mode="a", header=not COST_PATH.exists(), index=False)


# --- prompt + JSON extraction ----------------------------------------------


def _user_message(row) -> str:
    return format_input(row.ticker, row.company, row.title, row.summary)


SCHEMA_HINT = (
    "Reply with ONE JSON object only, no prose, matching exactly these fields: "
    + json.dumps(STUDENT_SCHEMA["required"])
)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_labels(text: str) -> tuple[dict, bool]:
    """Pull the first JSON object out of a reply and validate via parse_labels."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return _empty_labels(), False
    try:
        obj = parse_labels(match.group(0))
    except (ValueError, TypeError, json.JSONDecodeError):
        return _empty_labels(), False
    return obj, True


def _empty_labels() -> dict:
    return {"relevant": 0, "event_type": "", "sentiment": 0.0, "confidence": 0.0}


# --- backends ---------------------------------------------------------------


class TransformersBackend:
    """Portable backend (CUDA/MPS/CPU). Swap to vLLM for production scoring."""

    def __init__(self, model_id: str, adapter: str | None, max_new_tokens: int,
                 base_path: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        src = base_path or model_id
        self.tok = AutoTokenizer.from_pretrained(src)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(self.device).eval()

        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def one(self, user_text: str) -> tuple[str, int, int, float]:
        """Generate a reply. Returns (text, in_tokens, out_tokens, latency_s)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + SCHEMA_HINT},
            {"role": "user", "content": user_text},
        ]
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        in_tokens = int(input_ids.shape[-1])
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        latency = time.perf_counter() - t0
        gen = out[0, input_ids.shape[-1]:]
        out_tokens = int(gen.shape[-1])
        text = self.tok.decode(gen, skip_special_tokens=True)
        return text, in_tokens, out_tokens, latency


class DryRunBackend:
    """Mocked backend for structure checks. Always returns the same valid JSON."""

    def __init__(self, *args, **kwargs):
        self.device = "cpu"

    def one(self, user_text: str) -> tuple[str, int, int, float]:
        reply = ('{"relevant": true, "event_type": "earnings", '
                 '"sentiment": 0.25, "confidence": 0.6}')
        return reply, len(user_text) // 4, 20, 0.001


# --- main scoring loop ------------------------------------------------------


def score(model_tag: str, limit: int | None, out_path: Path, adapter: str | None,
          max_new_tokens: int, dry_run: bool, base_path: str | None) -> None:
    df = _load_news(IN_PATH, limit)
    if df.empty:
        print("No rows to score.")
        return

    done = _done_keys(out_path, model_tag)
    if done:
        keys = list(zip(df["news_id"].astype("int64"), df["ticker"].astype(str)))
        mask = [k not in done for k in keys]
        skipped = len(df) - sum(mask)
        if skipped:
            print(f"Resuming: {skipped:,} rows already scored, {sum(mask):,} to go.")
        df = df[mask].reset_index(drop=True)
    if df.empty:
        print("Everything already scored.")
        return

    if dry_run:
        backend = DryRunBackend()
    else:
        backend = TransformersBackend(
            MODEL_IDS[model_tag], adapter=adapter,
            max_new_tokens=max_new_tokens, base_path=base_path,
        )
    print(f"Model={model_tag}  device={backend.device}  rows={len(df):,}")

    buf: list[dict] = []
    tot_in = tot_out = 0
    bad = 0
    t0 = time.perf_counter()
    for i, row in enumerate(df.itertuples(index=False), start=1):
        user_text = _user_message(row)
        text, n_in, n_out, lat = backend.one(user_text)
        labels, ok = _extract_labels(text)
        if not ok and MAX_RETRIES:
            # One retry with a sharper nudge appended
            text2, n_in2, n_out2, lat2 = backend.one(
                user_text + "\n\nReturn ONLY the JSON object, no prose.")
            labels, ok = _extract_labels(text2)
            n_in += n_in2
            n_out += n_out2
            lat += lat2
        if not ok:
            bad += 1
        buf.append({
            "news_id": int(getattr(row, "news_id")),
            "ticker": str(getattr(row, "ticker")),
            "model": model_tag,
            "relevant": int(labels["relevant"]),
            "event_type": labels["event_type"],
            "sentiment": float(labels["sentiment"]),
            "confidence": float(labels["confidence"]),
            "valid_json": int(ok),
            "latency_ms": int(lat * 1000),
            "in_tokens": int(n_in),
            "out_tokens": int(n_out),
        })
        tot_in += n_in
        tot_out += n_out
        if len(buf) >= CHECKPOINT_EVERY:
            _append_chunk(out_path, buf)
            buf = []
        if i % 25 == 0 or i == len(df):
            rate = i / max(time.perf_counter() - t0, 1e-6)
            eta = (len(df) - i) / rate if rate else 0
            print(f"  [{i:>6d}/{len(df)}]  {rate:4.1f} rows/s  eta {eta/60:5.1f} min  "
                  f"bad_json={bad}", flush=True)
    if buf:
        _append_chunk(out_path, buf)
    wall = time.perf_counter() - t0

    _append_cost({
        "run_id": uuid.uuid4().hex[:8],
        "run_date": date.today().isoformat(),
        "owner": os.environ.get("USER", getpass.getuser()),
        "model": model_tag,
        "stage": "score",
        "n_rows": len(df),
        "n_stock_days": df["ticker"].nunique(),
        "wall_seconds": round(wall, 2),
        "gpu_type": backend.device,
        "gpu_hourly_usd": 0.0,
        "api_cost_usd": 0.0,
        "in_tokens": int(tot_in),
        "out_tokens": int(tot_out),
    })
    print(f"\nScored {len(df):,} rows in {wall:.1f}s ({len(df)/wall:.1f}/s). "
          f"bad_json={bad}  ({bad/len(df):.1%})")


# --- cli --------------------------------------------------------------------


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM news scoring harness (Task 4).")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    ap.add_argument("--adapter", help="Local path or HF id of a LoRA adapter (llama32_3b_ft)")
    ap.add_argument("--base-path", help="Local path to the base model weights (skip HF download)")
    ap.add_argument("--limit", type=int, help="Score only this many rows (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output", type=Path, default=OUT_PATH,
                    help="Scores CSV (default: data/interim/news_scores.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Mock backend: proves structure without loading a model")
    a = ap.parse_args()
    if a.model == "llama32_3b_ft" and not a.adapter:
        sys.exit("--adapter is required for llama32_3b_ft")
    score(a.model, a.limit, a.output, a.adapter, a.max_new_tokens, a.dry_run, a.base_path)
