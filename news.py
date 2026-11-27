"""
LeanTrader news pipeline (Nati) -- spec section "News", three sources.

Which source covers which dates (the team agreed to mix sources):
  FNSPID   2018-01-01 .. 2023-12-31   free Hugging Face dataset: Benzinga file (to Jun 2020)
                                       + Nasdaq file (to 2023)
  (gap)    2024-01-01 .. ~Nov 2024     no free source reaches this
  Massive  ~Nov 2024   .. day before the first Alpaca article   (free plan: ~2 years back)
  Alpaca   first Alpaca article (~Apr 29 2026) .. now           (free plan: recent news only)
  External 2018 .. Oct 2024, gap ticker-years mostly  (added by MJ, Nov 27 2026; see below)

External gap-fill (any source other than fnspid / massive / alpaca):
  data/raw/external/*.csv brings the 179 ticker-years (2018 .. Oct 2024) that had fewer than
  10 articles up to at least 10 kept each. addon_cumulative_delta.csv = the 8,333 rows an
  external backfill added: public datasets (drinkall_yahoo, dolphin_polygon,
  finmultitime_nasdaq, investopedia) plus publisher pages found in public archives, sitemaps
  and RSS (public_<publisher>, issuer_release). Its source names are kept as given. Timestamps
  are the publisher's or dataset's own UTC times; date-only records were left out.
  news_id is re-derived from the URL with id_from_url, and filters are rerun here.
  data/raw/external/provenance/ holds the audit files (publisher, date evidence, method).
  Most new rows are headlines only (summary empty). Existing rows are never dropped or edited.

Commands (run from the repo root, next to dating.py and configs/):
  python news.py pull-fnspid               # Benzinga file All_external.csv (5.7 GB), keep our rows
  python news.py pull-fnspid --file nasdaq-lite   # Nasdaq file, 1.8 GB JSONL copy (to 2023)
  python news.py pull-fnspid --file nasdaq        # fallback: original 23 GB Nasdaq CSV
  python news.py pull-massive [--tickers]  # ~2 years of Massive news   (needs MASSIVE_API_KEY)
  python news.py pull-alpaca  [--tickers]  # recent Alpaca news         (needs APCA_API_KEY_ID / APCA_API_SECRET_KEY)
  python news.py build                     # all caches -> data/interim/news_clean.csv + news_coverage.csv
  python news.py add-external              # merge the external file into an existing news_clean.csv
                                           # (no raw caches needed; reruns the same filters)

Every pull is cached under data/raw/<source>/ and skipped next time, so nothing is downloaded twice.

API keys: copy .env.example to .env and fill in your own keys (.env is gitignored).
Keys already exported in the shell win over the .env file.
"""

import argparse
import collections
import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------- settings
UNIVERSE_CSV = Path("configs/universe.csv")
RAW = Path("data/raw")
ALPACA_DIR, MASSIVE_DIR, FNSPID_DIR = RAW / "alpaca", RAW / "massive", RAW / "fnspid"
EXTERNAL_DIR = RAW / "external"   # hand-sourced / backfilled gap-fill CSVs, see docstring
# FNSPID ships two news files. All_external = Benzinga headlines, which END IN JUNE 2020.
# The Nasdaq file covers to 2023. "nasdaq-lite" is a community JSONL copy (1.8 GB) of the
# 23 GB Nasdaq CSV; try it first, and fall back to "nasdaq" if it lacks the columns.
FNSPID_FILES = {  # name: (Hugging Face repo, file in repo, our extract)
    "external":    ("Zihan1004/FNSPID", "Stock_news/All_external.csv",
                    FNSPID_DIR / "fnspid_universe_2018_2023.csv"),
    "nasdaq-lite": ("khaihernlow/fnspid", "Stock_news/nasdaq_exteral_data.jsonl",
                    FNSPID_DIR / "fnspid_nasdaq_2018_2023.csv"),
    "nasdaq":      ("Zihan1004/FNSPID", "Stock_news/nasdaq_exteral_data.csv",
                    FNSPID_DIR / "fnspid_nasdaq_2018_2023.csv"),
}
OUT_DIR = Path("data/interim")

START = "2018-01-01T00:00:00Z"
FNSPID_FIRST_DAY, FNSPID_LAST_DAY = "2018-01-01", "2023-12-31"

ALPACA_URL = "https://data.alpaca.markets/v1beta1/news"
ALPACA_PAGE, ALPACA_GAP = 50, 0.32        # 200 requests/min free plan
MASSIVE_URL = "https://api.massive.com/v2/reference/news"
MASSIVE_PAGE, MASSIVE_GAP = 1000, 12.5    # 5 requests/min free plan
MASSIVE_DAYS_BACK = 729                   # free plan reaches ~2 years

MAX_TICKERS_FOR_RELEVANCE = 3
DEDUP_WINDOW = pd.Timedelta(hours=48)
DAILY_CAP = 5

COLUMNS = [  # exact order from the spec -- do not change
    "news_id", "ticker", "company", "published_utc", "updated_utc",
    "decision_date", "title", "summary", "author", "source", "url",
    "n_tickers", "in_title", "dup_group", "keep",
]

ALIASES = {
    "GOOG": ["Google", "Alphabet"], "GOOGL": ["Google", "Alphabet"],
    "META": ["Meta", "Facebook"], "BRK-B": ["Berkshire"],
    "JPM": ["JPMorgan", "JP Morgan"], "BAC": ["Bank of America", "BofA"],
    "GS": ["Goldman"], "MS": ["Morgan Stanley"], "WFC": ["Wells Fargo"],
    "KO": ["Coca-Cola", "Coke"], "PG": ["Procter", "P&G"],
    "JNJ": ["Johnson & Johnson", "J&J"], "UNH": ["UnitedHealth"],
    "T": ["AT&T"], "VZ": ["Verizon"], "DIS": ["Disney"],
}


# ---------------------------------------------------------------- helpers
def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_CSV, dtype=str)[["ticker", "company"]]


def to_dot(ticker: str) -> str:
    """Universe spelling BRK-B -> API spelling BRK.B (Alpaca and Massive)."""
    return ticker.replace("-", ".")


# Companies that changed ticker: old symbol in the sources -> today's universe ticker.
OLD_TICKERS = {"FB": "META", "UTX": "RTX", "BK": "BNY"}
# Universe tickers whose news Massive files under a different symbol.
MASSIVE_SYMBOL = {"BNY": "BK"}


def to_universe(symbol: str) -> str:
    s = str(symbol).strip().upper().replace(".", "-")
    return OLD_TICKERS.get(s, s)

BENZINGA_ID = re.compile(r"benzinga\.com/.*?/(\d{5,})/")


def id_from_url(url: str) -> int:
    """Stable integer news_id for sources without Alpaca ids.

    Benzinga URLs carry Benzinga's own article number -- the same number Alpaca
    uses as its id -- so one Benzinga story gets one id in every source.
    Any other URL gets a hash, offset by 10**15 so it can never equal a Benzinga id.
    """
    m = BENZINGA_ID.search(url or "")
    if m:
        return int(m.group(1))
    return 10**15 + int(hashlib.sha1((url or "").encode()).hexdigest()[:12], 16)


def iso_z(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


TAG_RE, SPACE_RE = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def strip_html(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", str(text)))).strip()


def normalize_title(title: str) -> str:
    return SPACE_RE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", title.lower())).strip()


SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|plc|ltd|holdings?|group|"
    r"class [a-c]|the)\b\.?", re.IGNORECASE)


def name_patterns(ticker: str, company: str) -> list:
    clean = SPACE_RE.sub(" ", SUFFIX_RE.sub(" ", company.replace(",", " "))).strip(" &.")
    names = {clean} | set(ALIASES.get(ticker, []))
    pats = [re.compile(r"(?<![A-Za-z])" + re.escape(n) + r"(?![A-Za-z])", re.IGNORECASE)
            for n in names if len(n) >= 3]
    sym = re.escape(to_dot(ticker))
    if len(ticker) <= 2:
        pats.append(re.compile(r"(\$" + sym + r"\b|\(" + sym + r"\)|:\s?" + sym + r"\b)"))
    else:
        pats.append(re.compile(r"(?<![A-Za-z])\$?" + sym + r"(?![A-Za-z])"))
    return pats


class Throttle:
    """Waits so calls are at least `gap` seconds apart."""
    def __init__(self, gap):
        self.gap, self.last = gap, 0.0

    def wait(self):
        pause = self.gap - (time.time() - self.last)
        if pause > 0:
            time.sleep(pause)
        self.last = time.time()


def pick_tickers(arg):
    uni = load_universe()
    tickers = arg.split(",") if arg else uni["ticker"].tolist()
    unknown = set(tickers) - set(uni["ticker"])
    if unknown:
        sys.exit(f"Not in universe.csv: {sorted(unknown)} (use BRK-B, not BRK.B)")
    return tickers


def cached_write(final: Path, fetch):
    """Run fetch(file) into a .part file; rename only when it finished."""
    if final.exists():
        print(f"  {final.stem}: cached, skipping")
        return
    part = final.with_suffix(".jsonl.part")
    with part.open("w", encoding="utf-8") as f:
        n = fetch(f)
    part.rename(final)
    print(f"  {final.stem}: {n} articles")


# ---------------------------------------------------------------- pull: Alpaca
def pull_alpaca_ticker(session, ticker, end, throttle):
    def fetch(f):
        params = {"symbols": to_dot(ticker), "start": START, "end": end,
                  "limit": ALPACA_PAGE, "include_content": "false"}
        n = 0
        while True:
            throttle.wait()
            r = session.get(ALPACA_URL, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
                continue
            r.raise_for_status()
            body = r.json()
            for art in body.get("news") or []:
                f.write(json.dumps(art) + "\n")
                n += 1
            if not body.get("next_page_token"):
                return n
            params["page_token"] = body["next_page_token"]
    cached_write(ALPACA_DIR / f"{ticker}.jsonl", fetch)


def cmd_pull_alpaca(args):
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY first (in .env, see .env.example).")
    ALPACA_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    end, th = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), Throttle(ALPACA_GAP)
    for t in pick_tickers(args.tickers):
        pull_alpaca_ticker(s, t, end, th)


# ---------------------------------------------------------------- pull: Massive
def pull_massive_ticker(session, ticker, start, end, key, throttle):
    def fetch(f):
        url = MASSIVE_URL
        params = {"ticker": MASSIVE_SYMBOL.get(ticker, to_dot(ticker)),
                  "published_utc.gte": start, "published_utc.lte": end,
                  "limit": MASSIVE_PAGE, "sort": "published_utc", "order": "asc", "apiKey": key}
        n = 0
        while url:
            throttle.wait()
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429:           # over 5/min: wait a full minute, retry same page
                time.sleep(60)
                continue
            r.raise_for_status()
            body = r.json()
            for art in body.get("results") or []:
                f.write(json.dumps(art) + "\n")
                n += 1
            url = body.get("next_url")          # already holds the cursor and filters...
            params = {"apiKey": key}            # ...but not the key
        return n
    cached_write(MASSIVE_DIR / f"{ticker}.jsonl", fetch)


def cmd_pull_massive(args):
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        sys.exit("Set MASSIVE_API_KEY first (in .env, see .env.example).")
    MASSIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=MASSIVE_DAYS_BACK)).strftime("%Y-%m-%dT00:00:00Z")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    tickers = pick_tickers(args.tickers)
    print(f"Massive {start} -> {end}, {len(tickers)} tickers, ~{MASSIVE_GAP:.0f}s per call")
    s, th = requests.Session(), Throttle(MASSIVE_GAP)
    for t in tickers:
        pull_massive_ticker(s, t, start, end, key, th)


# ---------------------------------------------------------------- pull: FNSPID
def cmd_pull_fnspid(args):
    """Download FNSPID's All_external.csv once, then keep only our tickers in 2018-2023.

    Two passes over the big file, in chunks so it never has to fit in memory:
      1. keep rows whose symbol is in our universe and whose date is in the window
      2. for those articles' URLs, count how many symbols (any company) each URL is
         tagged with -> n_tickers, which the relevance filter needs
    --file picks which FNSPID file (see FNSPID_FILES); each gets its own extract,
    and `build` reads every extract it finds.
    """
    repo, path_in_repo, extract = FNSPID_FILES[args.file]
    FNSPID_DIR.mkdir(parents=True, exist_ok=True)
    if extract.exists():
        print(f"{extract} exists, skipping")
        return
    if args.local:
        src = args.local
    else:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {repo}/{path_in_repo} -- resumes if interrupted...")
        src = hf_hub_download(repo, path_in_repo, repo_type="dataset", local_dir=FNSPID_DIR / args.file)
    uni = set(load_universe()["ticker"])

    parts, pairs, seen = [], set(), 0
    for chunk in fnspid_chunks(src):
        seen += len(chunk)
        day = chunk["Date"].str[:10]
        chunk["ticker"] = chunk["Stock_symbol"].map(to_universe)
        in_window = chunk[(day >= FNSPID_FIRST_DAY) & (day <= FNSPID_LAST_DAY)]
        # remember every (article, symbol) pair in the window, any company -> n_tickers
        pairs.update(zip(in_window["Url"], in_window["ticker"]))
        parts.append(in_window[in_window["ticker"].isin(uni)])
        print(f"  {seen:,} rows read", end="\r")
    kept = pd.concat(parts, ignore_index=True)
    print(f"\n  kept {len(kept):,} rows for {kept['ticker'].nunique()} of our tickers "
          f"({kept['Date'].min()[:10]} -> {kept['Date'].max()[:10]})")

    counts = collections.Counter(u for u, _ in pairs)
    kept["n_tickers"] = kept["Url"].map(counts).fillna(1).astype(int)
    kept[FNSPID_COLS + ["ticker", "n_tickers"]].to_csv(extract, index=False)
    print(f"  wrote {extract}")


FNSPID_COLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Lsa_summary"]
FNSPID_NEEDED = ["Date", "Article_title", "Stock_symbol", "Url"]

def jsonl_chunks(src, size=200_000):
    """Read a JSONL file line by line, skipping lines that aren't valid JSON.

    pandas' own JSONL reader fails a whole chunk on one broken line; this keeps
    every good line, keeps only the columns we use (less memory), and reports
    how many lines it had to skip.
    """
    rows, good, bad = [], 0, 0
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows.append({k: obj.get(k) for k in FNSPID_COLS if k in obj})
            good += 1
            if len(rows) >= size:
                yield pd.DataFrame(rows)
                rows = []
    if rows:
        yield pd.DataFrame(rows)
    print(f"\n  read {good:,} lines, skipped {bad:,} broken lines ({bad / max(good + bad, 1):.4%})")

def fnspid_chunks(src):
    """Yield chunks of the big file as all-string DataFrames with FNSPID_COLS.

    Works for the original CSVs and for JSONL copies. Columns the file lacks
    (e.g. a copy without summaries) come back as empty strings; the four
    columns in FNSPID_NEEDED must exist.
    """
    src = str(src)
    if src.endswith(".jsonl"):
        reader = jsonl_chunks(src)
    else:
        reader = pd.read_csv(src, dtype=str, on_bad_lines="skip", chunksize=500_000,
                             usecols=lambda c: c in FNSPID_COLS)
    for chunk in reader:
        missing = [c for c in FNSPID_NEEDED if c not in chunk.columns]
        if missing:
            sys.exit(f"{src} lacks columns {missing}; it has {list(chunk.columns)}")
        for c in FNSPID_COLS:
            chunk[c] = chunk[c].fillna("").astype(str) if c in chunk.columns else ""
        # one date format whatever the file used: "YYYY-MM-DD HH:MM:SS UTC"
        ts = pd.to_datetime(chunk["Date"], utc=True, errors="coerce", format="mixed")
        chunk["Date"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S UTC").fillna("")
        yield chunk[FNSPID_COLS].copy()


# ---------------------------------------------------------------- read caches -> common rows
COMMON = ["news_id", "ticker", "published", "updated", "title", "summary",
          "author", "source", "url", "n_tickers", "priority"]


def read_jsonl_dir(folder: Path):
    for path in sorted(folder.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def read_alpaca() -> pd.DataFrame:
    rows = []
    for a in read_jsonl_dir(ALPACA_DIR):
        syms = [to_universe(s) for s in (a.get("symbols") or [])]
        for t in syms:
            rows.append({"news_id": int(a["id"]), "ticker": t,
                         "published": a.get("created_at"), "updated": a.get("updated_at"),
                         "title": a.get("headline") or "", "summary": a.get("summary"),
                         "author": a.get("author") or "", "source": "alpaca",
                         "url": a.get("url") or "", "n_tickers": len(syms), "priority": 0})
    return pd.DataFrame(rows, columns=COMMON)


def read_massive() -> pd.DataFrame:
    rows = []
    for a in read_jsonl_dir(MASSIVE_DIR):
        syms = [to_universe(s) for s in (a.get("tickers") or [])]
        url = a.get("article_url") or ""
        for t in syms:
            rows.append({"news_id": id_from_url(url), "ticker": t,
                         "published": a.get("published_utc"), "updated": a.get("published_utc"),
                         "title": a.get("title") or "", "summary": a.get("description"),
                         "author": a.get("author") or "", "source": "massive",
                         "url": url, "n_tickers": len(syms), "priority": 1})
    return pd.DataFrame(rows, columns=COMMON)


def read_fnspid() -> pd.DataFrame:
    """FNSPID times can't be trusted (date-only before ~2020, shifted after), so every
    article is dated to the END of its listed day (23:59:59 New York time). The shared
    dating function then gives the next trading day -- never earlier than the truth.
    The original listed time is kept in updated_utc for audit."""
    extracts = sorted(FNSPID_DIR.glob("fnspid_*_2018_2023.csv"))
    if not extracts:
        return pd.DataFrame(columns=COMMON)
    df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in extracts],
                   ignore_index=True)
    listed_day = pd.to_datetime(df["Date"].str[:10], errors="coerce")
    end_of_day = (listed_day + pd.Timedelta(hours=23, minutes=59, seconds=59)) \
        .dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return pd.DataFrame({
        "news_id": df["Url"].map(id_from_url), "ticker": df["ticker"],
        "published": end_of_day,
        "updated": pd.to_datetime(df["Date"].str.replace(" UTC", "", regex=False),
                                  utc=True, errors="coerce"),
        "title": df["Article_title"], "summary": df["Lsa_summary"],
        "author": df["Author"].where(df["Author"] != "", df["Publisher"]),
        "source": "fnspid", "url": df["Url"],
        "n_tickers": pd.to_numeric(df["n_tickers"]).astype(int), "priority": 2,
    })[COMMON]


EXTERNAL_COLS = ["ticker", "range_start", "range_end", "title", "publisher",
                 "published_utc", "time_status", "url", "author", "news_id"]


CORE_SOURCES = ("alpaca", "massive", "fnspid")   # every other source value is external gap-fill
ISO_Z_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


def is_external(source: pd.Series) -> pd.Series:
    return ~source.isin(CORE_SOURCES)


def read_external(folder: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Every CSV directly in data/raw/external/ (sub-folders hold audit files only). Two layouts:
    the hand-sourced article list (range_start/range_end/publisher...) or rows already in the
    news_clean.csv layout. Only text, URL, timestamp and ticker are trusted: news_id is re-derived
    from the URL with our own id rule, and in_title / dup_group / keep / decision_date are rerun."""
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    parts = [read_external_clean(p) if "dup_group" in pd.read_csv(p, nrows=0).columns
             else read_external_articles(p) for p in files]
    if not parts:
        return pd.DataFrame(columns=COMMON)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["news_id", "ticker"])
    n = df.groupby("news_id")["ticker"].transform("nunique")
    df["n_tickers"] = df["n_tickers"].combine(n, max).astype(int)
    return df


def read_external_clean(path: Path) -> pd.DataFrame:
    """Rows in the news_clean.csv layout, e.g. the delta of a cumulative backfill."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)}")
    bad = ~df["published_utc"].str.match(ISO_Z_RE) | (df["url"] == "") | (df["title"] == "")
    if bad.any():
        sys.exit(f"{path}: {int(bad.sum())} rows lack a UTC ...Z timestamp, url or title")
    if not is_external(df["source"]).all():
        sys.exit(f"{path}: external rows may not use a core source name {CORE_SOURCES}")
    upd = df["updated_utc"].where(df["updated_utc"].str.match(ISO_Z_RE), df["published_utc"])
    return pd.DataFrame({
        "news_id": df["url"].map(id_from_url).astype("int64"), "ticker": df["ticker"].map(to_universe),
        "published": df["published_utc"], "updated": upd,
        "title": df["title"].map(strip_html), "summary": df["summary"].map(strip_html),
        "author": df["author"], "source": df["source"], "url": df["url"],
        "n_tickers": pd.to_numeric(df["n_tickers"], errors="coerce").fillna(1).astype(int),
        "priority": 3,
    })[COMMON]


def read_external_articles(path: Path) -> pd.DataFrame:
    """The hand-sourced article list. Checked hard on the way in: each row must carry a UTC
    timestamp ending in Z, fall inside the ticker-year range it was sourced for, and have
    the news_id our own id rule gives its URL."""
    df = pd.read_csv(path, dtype={"ticker": str, "author": str})
    missing = set(EXTERNAL_COLS) - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)}")
    bad_ts = ~df["published_utc"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    day = df["published_utc"].astype(str).str[:10]
    outside = (day < df["range_start"]) | (day > df["range_end"])
    bad_id = df["url"].map(id_from_url) != df["news_id"]
    for name, bad in (("timestamp not UTC ...Z", bad_ts), ("outside its range", outside),
                      ("news_id != id_from_url(url)", bad_id)):
        if bad.any():
            sys.exit(f"{path}: {int(bad.sum())} rows {name}, e.g. {df[bad].iloc[0].to_dict()}")
    df["ticker"] = df["ticker"].map(to_universe)
    n_tickers = df.groupby("news_id")["ticker"].transform("nunique")
    return pd.DataFrame({
        "news_id": df["news_id"].astype("int64"), "ticker": df["ticker"],
        "published": df["published_utc"], "updated": df["published_utc"],
        "title": df["title"].map(strip_html), "summary": "",
        "author": df["author"].fillna("").where(df["author"].fillna("") != "", df["publisher"]),
        "source": "external", "url": df["url"], "n_tickers": n_tickers.astype(int), "priority": 3,
    })[COMMON]


def fix_external_n_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """An external article can also be in another source under a different ticker; count
    every ticker the combined corpus tags it with, so the relevance filter sees the truth."""
    ext = is_external(df["source"])
    if ext.any():
        seen = df.groupby("news_id")["ticker"].transform("nunique")
        df.loc[ext, "n_tickers"] = df.loc[ext, "n_tickers"].combine(seen[ext], max).astype(int)
    return df


def combine_sources(universe: pd.DataFrame) -> pd.DataFrame:
    al, ms, fn, ex = read_alpaca(), read_massive(), read_fnspid(), read_external()
    for d in (al, ms, fn, ex):
        d["published"] = pd.to_datetime(d["published"], utc=True)
        d["updated"] = pd.to_datetime(d["updated"], utc=True)

    # Each source covers its own window, so the sources never overlap:
    fn = fn[(fn["published"] >= pd.Timestamp(FNSPID_FIRST_DAY, tz="America/New_York"))
            & (fn["published"] < pd.Timestamp("2024-01-01", tz="America/New_York"))]
    if len(al):
        alpaca_first_day = al["published"].min().tz_convert("America/New_York").normalize()
        ms = ms[ms["published"] < alpaca_first_day]
        al = al[al["published"] >= alpaca_first_day]
    # External rows are not clipped: they were sourced for specific gap ticker-years.
    for name, d in (("fnspid", fn), ("massive", ms), ("alpaca", al), ("external", ex)):
        if len(d):
            print(f"  {name:8s} {len(d):>9,} rows  {d['published'].min():%Y-%m-%d} -> "
                  f"{d['published'].max():%Y-%m-%d}")
        else:
            print(f"  {name:8s}         0 rows")

    df = pd.concat([x for x in (al, ms, fn, ex) if len(x)], ignore_index=True)
    if df.empty:
        sys.exit("No cached articles -- run a pull first.")
    return tidy(df, universe)


def tidy(df: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Universe tickers only, from START on, one row per (news_id, ticker), company attached."""
    df = df[df["ticker"].isin(universe["ticker"]) & (df["published"] >= pd.Timestamp(START))]
    df = df.sort_values("priority", kind="stable").drop_duplicates(["news_id", "ticker"])  # same story twice
    df = fix_external_n_tickers(df.reset_index(drop=True))
    return df.merge(universe, on="ticker", how="left").reset_index(drop=True)


# ---------------------------------------------------------------- build
def add_dup_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "norm_title", "published", "news_id"]).copy()
    groups, key, anchor_t, anchor_id = [], None, None, None
    for t, nt, pub, nid in zip(df["ticker"], df["norm_title"], df["published"], df["news_id"]):
        if (t, nt) != key or pub - anchor_t > DEDUP_WINDOW:
            key, anchor_t, anchor_id = (t, nt), pub, nid
        groups.append(f"{t}_{anchor_id}")
    df["dup_group"] = groups
    df["is_first"] = ~df.duplicated("dup_group", keep="first")
    return df


def cmd_build(args):
    uni = load_universe()
    df = combine_sources(uni)
    df["summary"] = df["summary"].map(strip_html)
    df["title"] = df["title"].map(strip_html)
    finalize(df, uni)


SOURCE_PRIORITY = {"alpaca": 0, "massive": 1, "fnspid": 2, "external": 3}


def cmd_add_external(args):
    """Merge the external file into the news_clean.csv already on disk, for machines that
    don't hold the raw FNSPID/Massive/Alpaca caches. Existing rows keep their text and
    timestamps as written; only the filters (in_title, dup_group, keep) are rerun."""
    uni = load_universe()
    path = OUT_DIR / "news_clean.csv"
    old = pd.read_csv(path, dtype={"ticker": str, "summary": str, "author": str},
                      keep_default_na=False)
    old = old[~is_external(old["source"])]          # rerunnable: replace, never stack
    base = pd.DataFrame({
        "news_id": old["news_id"].astype("int64"), "ticker": old["ticker"],
        "published": pd.to_datetime(old["published_utc"], utc=True),
        "updated": pd.to_datetime(old["updated_utc"], utc=True),
        "title": old["title"], "summary": old["summary"], "author": old["author"],
        "source": old["source"], "url": old["url"], "n_tickers": old["n_tickers"].astype(int),
        "priority": old["source"].map(SOURCE_PRIORITY).fillna(3).astype(int),
    })[COMMON]
    ex = read_external()
    if ex.empty:
        sys.exit(f"no CSV files in {EXTERNAL_DIR}")
    ex["published"] = pd.to_datetime(ex["published"], utc=True)
    ex["updated"] = pd.to_datetime(ex["updated"], utc=True)
    print(f"  existing {len(base):>9,} rows   external {len(ex):,} rows")
    finalize(tidy(pd.concat([base, ex], ignore_index=True), uni), uni)


def finalize(df: pd.DataFrame, uni: pd.DataFrame) -> None:
    """Dating, relevance, de-dup and daily cap; writes news_clean.csv and news_coverage.csv."""
    from dating import decision_date  # MJ's shared function -- never write our own

    df["summary"] = df["summary"].fillna("")
    dd =pd.to_datetime(pd.Series(list(decision_date(df["published"])), index=df.index))
    check = df["published"].sample(min(200, len(df)), random_state=0)   # safety check
    assert (pd.to_datetime(check.map(decision_date)) == dd[check.index]).all(), \
        "decision_date gave different answers for a column vs single rows -- tell Nati/Claude"
    df["decision_date"] = dd.dt.strftime("%Y-%m-%d")

    pats = {t: name_patterns(t, c) for t, c in zip(uni["ticker"], uni["company"])}
    df["in_title"] = [int(any(p.search(ti) for p in pats[t])) for t, ti in zip(df["ticker"], df["title"])]

    df["relevant"] = (df["n_tickers"] <= MAX_TICKERS_FOR_RELEVANCE) | (df["in_title"] == 1)
    df["norm_title"] = df["title"].map(normalize_title)
    df = add_dup_groups(df)

    cand = df[df["relevant"] & df["is_first"]].sort_values(
        ["ticker", "decision_date", "in_title", "published", "news_id"],
        ascending=[True, True, False, True, True])
    df["keep"] = 0
    df.loc[cand.groupby(["ticker", "decision_date"]).head(DAILY_CAP).index, "keep"] = 1

    df = df.sort_values(["published", "news_id", "ticker"])
    df["published_utc"], df["updated_utc"] = iso_z(df["published"]), iso_z(df["updated"])
    out = df[COLUMNS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "news_clean.csv", index=False)

    years = range(int(START[:4]), pd.Timestamp.now(tz="UTC").year + 1)
    grid = pd.MultiIndex.from_product([uni["ticker"], years], names=["ticker", "year"])
    cov = (df.assign(year=df["published"].dt.year).groupby(["ticker", "year"])
             .agg(n_articles=("news_id", "size"), n_kept=("keep", "sum"))
             .reindex(grid, fill_value=0).reset_index())
    cov.to_csv(OUT_DIR / "news_coverage.csv", index=False)

    print(f"news_clean.csv: {len(out):,} rows, {int(out['keep'].sum()):,} kept, "
          f"{out['news_id'].nunique():,} articles, {out['ticker'].nunique()} tickers")
    print(f"dropped by relevance: {int((~df['relevant']).sum()):,}  "
          f"duplicates: {int((~df['is_first']).sum()):,}")
    print("kept rows per source:", df[df["keep"] == 1]["source"].value_counts().to_dict())


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    load_dotenv()  # reads .env from the repo root; never overrides keys already in the shell
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pull-alpaca", "pull-massive"):
        p = sub.add_parser(name)
        p.add_argument("--tickers", help="comma list, e.g. AAPL,BRK-B,DUK (default: all 100)")
    p = sub.add_parser("pull-fnspid")
    p.add_argument("--file", choices=list(FNSPID_FILES), default="external",
                   help="which FNSPID file: external (default), nasdaq-lite, nasdaq")
    p.add_argument("--local", help="use an already-downloaded copy of that file instead of downloading")
    sub.add_parser("build")
    sub.add_parser("add-external")
    a = ap.parse_args()
    {"pull-alpaca": cmd_pull_alpaca, "pull-massive": cmd_pull_massive,
     "pull-fnspid": cmd_pull_fnspid, "build": cmd_build,
     "add-external": cmd_add_external}[a.cmd](a)
