"""Pull SEC EDGAR filing events for the LeanTrader universe and build daily features.

Owner: MJ. Spec: product spec, "SEC filings — Malith".

Output:
    data/interim/sec_events.csv   one row per (filing, ticker): 8-K, 8-K/A, 10-Q, 10-K
    data/interim/sec_daily.csv    one row per (trading day, ticker) from 2018-01-01, zeros on quiet days
    data/raw/sec/*.json           raw SEC responses, cached so nobody pulls twice

Rules (from the spec):
    Source is data.sec.gov/submissions (metadata only, no filing text, no XBRL).
    Send a User-Agent with a name and email; stay under 10 requests per second.
    Load the extra JSON pages listed in filings.files, or older years go missing.
    Date every filing by acceptanceDateTime through dating.decision_date().
    acceptanceDateTime is real UTC: Apple's 8-K 0000320193-26-000024 shows
    2026-11-02T21:30:23.000Z in the JSON and "Accepted 2026-11-02 16:30:23" (New York)
    on its EDGAR index page. tests/test_sec.py locks this in.
    Share classes with one CIK (GOOG/GOOGL) get a copy of each filing.
    Item 9.01 (exhibits) is ignored: it's on almost every 8-K.

Usage:
    python sec_edgar.py --email you@fordham.edu                         # all 100 tickers
    python sec_edgar.py --email you@fordham.edu --tickers AAPL MSFT KO  # quick sample first
    python sec_edgar.py --email you@fordham.edu --refresh               # re-download latest filings

    pytest tests/test_sec.py  # run the unit tests, which check the dating logic and column shapes
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from dating import MARKET_TZ, decision_date, trading_days
from universe import load_universe

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "sec"
OUT_DIR = ROOT / "data" / "interim"
BASE_URL = "https://data.sec.gov/submissions/"

FORMS = ["8-K", "8-K/A", "10-Q", "10-K"]
EIGHT_K = {"8-K", "8-K/A"}
PERIODIC = {"10-Q", "10-K"}
ITEMS = {  # column -> 8-K item code
    "i_202": "2.02",  # earnings results
    "i_502": "5.02",  # executive or director change
    "i_101": "1.01",  # material agreement
    "i_201": "2.01",  # acquisition or sale completed
    "i_701": "7.01",  # Reg FD disclosure
    "i_801": "8.01",  # other events
}
EVENTS_FROM = "2017-01-01"  # one spare year so day-one counters (days since, last 20 days) are right
DAILY_FROM = "2018-01-01"
CAP = 90  # days_since_* are capped at 90 trading days

EVENT_COLS = ["ticker", "cik", "accession", "form", "items", "filing_date",
              "accepted_utc", "decision_date", "report_date"]
DAILY_COLS = ["date", "ticker", "has_8k", *ITEMS, "has_periodic",
              "days_since_202", "days_since_8k", "n_8k_20d", "post_earn_5d"]

MIN_GAP_S = 0.15  # seconds between live requests: about 7 per second, under SEC's 10
_last_request = [0.0]


# ---------- Step 1: fetch politely, cache everything ----------

def fetch_json(name: str, email: str, refresh: bool = False) -> dict:
    """GET data.sec.gov/submissions/<name>, cached in data/raw/sec/<name>."""
    cache = RAW_DIR / name
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    req = urllib.request.Request(
        BASE_URL + name,
        headers={"User-Agent": f"LeanTrader research project {email}"},
    )
    for attempt in range(5):
        wait = MIN_GAP_S - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(2 ** attempt)  # SEC asked us to slow down: back off 1, 2, 4, 8 s
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
        finally:
            _last_request[0] = time.monotonic()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


# ---------- Step 2: one company's filings as a table ----------

def _columns(page: dict) -> dict:
    """The main file nests the columns under filings.recent; extra pages hold them at the top."""
    return page["filings"]["recent"] if "filings" in page else page


def company_filings(cik: str, email: str, refresh: bool = False) -> pd.DataFrame:
    """Every filing for one company since EVENTS_FROM, newest pages plus older extra pages."""
    main = fetch_json(f"CIK{cik}.json", email, refresh)
    frames = [pd.DataFrame(_columns(main))]
    for page in main["filings"].get("files", []):
        if page["filingTo"] >= EVENTS_FROM:  # skip pages that end before our window
            frames.append(pd.DataFrame(_columns(fetch_json(page["name"], email))))  # old pages never change
    return pd.concat(frames, ignore_index=True)


# ---------- Step 3: sec_events ----------

def build_events(universe: pd.DataFrame, fetch) -> pd.DataFrame:
    """One row per (filing, ticker). `fetch(cik)` returns that company's filings table."""
    parts = []
    for cik, group in universe.groupby("cik", sort=False):
        df = fetch(cik)
        df = df[df["form"].isin(FORMS) & (df["filingDate"] >= EVENTS_FROM)]
        for ticker in group["ticker"]:  # one CIK, several share classes -> a copy each
            parts.append(df.assign(ticker=ticker, cik=cik))
    ev = pd.concat(parts, ignore_index=True)

    is_8k = ev["form"].isin(EIGHT_K)
    items = ev["items"].fillna("").astype(str).str.replace(" ", "").str.replace(",", ";")
    accepted = pd.to_datetime(ev["acceptanceDateTime"], utc=True)

    out = pd.DataFrame({
        "ticker": ev["ticker"],
        "cik": ev["cik"],
        "accession": ev["accessionNumber"],
        "form": ev["form"],
        "items": items.where(is_8k, ""),
        "filing_date": ev["filingDate"],
        "accepted_utc": accepted,
        "decision_date": decision_date(accepted),
        "report_date": ev["reportDate"].fillna("").astype(str),
    })
    out = out.drop_duplicates(["accession", "ticker"]).sort_values(["ticker", "accepted_utc", "accession"])
    out["accepted_utc"] = out["accepted_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["decision_date"] = out["decision_date"].dt.strftime("%Y-%m-%d")
    return out[EVENT_COLS].reset_index(drop=True)


# ---------- Step 4: sec_daily ----------

def _days_since(d: pd.DataFrame, flag: str) -> pd.Series:
    """Trading days since the flag was last 1 for this ticker (0 = today), capped at CAP."""
    pos = d.groupby("ticker").cumcount()
    last = pos.where(d[flag] == 1).groupby(d["ticker"]).ffill()
    return (pos - last).fillna(CAP).clip(upper=CAP).astype(int)


def last_trading_day() -> pd.Timestamp:
    today = pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None).normalize()
    return trading_days(EVENTS_FROM, today)[-1]


def build_daily(events: pd.DataFrame, tickers: list[str], end=None) -> pd.DataFrame:
    """One row per (trading day, ticker) from DAILY_FROM to `end`, zeros on days with no filings."""
    end = pd.Timestamp(end) if end is not None else last_trading_day()
    days = trading_days(EVENTS_FROM, end)  # include the spare year so counters start right

    ev = events.assign(date=pd.to_datetime(events["decision_date"]))
    ev = ev[ev["date"].isin(days)]  # a filing dated after `end` waits for a later run
    is_8k = ev["form"].isin(EIGHT_K)
    item_sets = ev["items"].fillna("").str.split(";")

    flags = pd.DataFrame({
        "ticker": ev["ticker"],
        "date": ev["date"],
        "n_8k": is_8k.astype(int),
        "has_periodic": ev["form"].isin(PERIODIC).astype(int),
    })
    for col, code in ITEMS.items():
        flags[col] = (is_8k & item_sets.apply(lambda xs, c=code: c in xs)).astype(int)

    per_day = flags.groupby(["ticker", "date"]).agg(
        n_8k=("n_8k", "sum"),
        has_periodic=("has_periodic", "max"),
        **{col: (col, "max") for col in ITEMS},
    )
    grid = pd.MultiIndex.from_product([sorted(tickers), days], names=["ticker", "date"])
    d = per_day.reindex(grid, fill_value=0).reset_index()  # quiet days become zeros

    d["has_8k"] = (d["n_8k"] > 0).astype(int)
    d["n_8k_20d"] = d.groupby("ticker")["n_8k"].transform(lambda s: s.rolling(20, min_periods=1).sum()).astype(int)
    d["days_since_202"] = _days_since(d, "i_202")
    d["days_since_8k"] = _days_since(d, "has_8k")
    d["post_earn_5d"] = (d["days_since_202"] <= 4).astype(int)  # today or the 4 trading days before

    d = d[d["date"] >= DAILY_FROM].copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d[DAILY_COLS].sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------- Run it ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pull SEC EDGAR filings and build sec_events / sec_daily.")
    parser.add_argument("--email", required=True, help="Your email, for SEC's User-Agent rule")
    parser.add_argument("--tickers", nargs="*", help="Only these tickers (a quick sample run)")
    parser.add_argument("--refresh", action="store_true", help="Re-download the latest filings")
    args = parser.parse_args()

    uni = load_universe()
    sample = bool(args.tickers)
    if sample:
        uni = uni[uni["ticker"].isin(args.tickers)]

    t0 = time.time()
    events = build_events(uni, fetch=lambda cik: company_filings(cik, args.email, args.refresh))
    daily = build_daily(events, uni["ticker"].tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".sample" if sample else ""
    events.to_csv(OUT_DIR / f"sec_events{suffix}.csv", index=False)
    daily.to_csv(OUT_DIR / f"sec_daily{suffix}.csv", index=False)

    print(f"Done in {time.time() - t0:.0f}s: {len(events):,} filings, {len(daily):,} daily rows")
    print("\nFilings per year (eyeball for empty years):")
    per_year = events.assign(year=events["filing_date"].str[:4]).pivot_table(
        index="year", columns="form", values="accession", aggfunc="count", fill_value=0)
    print(per_year.to_string())


if __name__ == "__main__":
    main()