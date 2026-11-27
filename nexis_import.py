"""
Nexis Uni exports -> an external news batch for `python news.py add-external` (MJ, Nov 2026).

Nexis Uni (via Fordham) bulk downloads come as DOCX: one file per article, a folder of them,
a zip of them, or one combined "Files (N).DOCX". This script reads any of those, keeps only
English news about the company, and writes the news_clean.csv layout with source = NexisUni.

  python nexis_import.py --batch nexisuni_batch02 --plan ../NexisUni/plan/nexis_download_plan.csv \
      LIN="../NexisUni/Files (500).DOCX" DUK="../NexisUni/Files (149)"
  python news.py add-external

Writes data/raw/external/<batch>.csv (merged by add-external) and
data/raw/external/provenance/<batch>_meta.csv (every article, kept or excluded, with the reason).
Only the headline and a lead of at most 500 characters are stored, never the full text.

Filters, in order (an article must pass all of them):
  1. same Nexis document (urn:contentItem) seen before -> duplicate
  2. date line parsed; language English (field or stop-word ratio)
  3. not a press release / regulatory / government feed (Business Wire, PR Newswire, GlobeNewswire,
     ENP, M2, AWP OTS, Targeted News Service, Impact News Service, US Fed News, ...)
  4. not an auto-generated holdings site or title (Zolmax, Ticker Report, DailyPolitical, ...)
  5. company named in the headline or lead (news.name_patterns), not a look-alike entity
     (Linde India, van der Linde, ...), not sport / politics, and for ambiguous names a company cue
  6. inside the date ranges the download plan asked for, one copy per headline and day

Timestamps: the article's own date line (not Load-Date). With a time and zone, the zone's
standard offset is used (EST for "EST" even in summer), which can only make the time later than
the truth. "12:00 AM" or no time -> 23:59:59 New York time that day, so the article counts from
the next trading day, as FNSPID rows do.
"""
import argparse
import io
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd

from news import COLUMNS, name_patterns, normalize_title

NY = ZoneInfo("America/New_York")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
URN = re.compile(r"urn:contentItem:[0-9A-Z-]+")
DATE_LINE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|"
                       r"December) \d{1,2}, \d{4}\b")
DATE_RE = re.compile(r"^(\w+ \d{1,2}, \d{4})(?: \w+day)?(?: (\d{1,2}:\d{2}) ?([AP]M)\s+([A-Z]{2,5}))?\s*$")
FIELD = re.compile(r"^(Byline|Length|Section|Dateline|Highlight|Language|Load-Date|Publication-Type|"
                   r"Subject|Company|Ticker|Industry|Geographic|Graphic|Correction-Date|Notes|Series|"
                   r"Classification|Document-Type|Journal Code|Copyright|Edition):\s*(.*)$")
END_OF_BODY = {"Load-Date", "Classification", "Graphic", "Subject", "Company", "Ticker", "Industry",
               "Geographic", "Document-Type", "Publication-Type", "Language", "Correction-Date", "Notes",
               "Journal Code"}
TZ_STD = {"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "PT": -8,
          "ET": -5, "GMT": 0, "UTC": 0, "BST": 1, "CET": 1, "CEST": 2, "EET": 2, "EEST": 3, "IST": 5.5,
          "SGT": 8, "HKT": 8, "JST": 9, "AEST": 10, "AEDT": 11}

PR_SOURCES = re.compile(r"Business Wire|PR Newswire|GlobeNewswire|Globe Newswire|Access ?wire|Access Newswire|"
                        r"ENP Newswire|M2 PressWIRE|AWP OTS|Original text service|Cision|TheNewswire|InPR|"
                        r"Regulatory News Service|FinancialWire|Targeted News Service|Impact News Service|"
                        r"US Fed News|States News Service|Financial Deals Tracker|Contify|News Bites|"
                        r"Dion Global|Africa Centres for Disease|Mena Report|Chemicals Monitor|"
                        r"National News Agency \(UzA\)", re.I)
PR_TITLES = re.compile(r"^(BUSINESSWIRE|EQS|PRNEWSWIRE|GLOBENEWSWIRE)\b")
ROBO_SOURCES = re.compile(r"Zolmax|Ticker Report|DailyPolitical|Transcript Daily|Watchlist News|MarketBeat|"
                          r"ETF Daily News|Cerbat Gem|Modern Readers|Enterprise Leader|Markets Daily|"
                          r"American Banking News|Community Financial News|Stock Observer|Lincolnian", re.I)
ROBO_TITLES = re.compile(r"\bShares? (?:of|in)\b.*\b(?:Purchased|Bought|Sold|Acquired)\b|"
                         r"\b(?:Stake|Position|Holdings?) in\b|\bversus Its Rivals\b|\band The Competition\b|"
                         r"\bCritical Survey\b|\bHead to Head\b|\bHead-To-Head\b|\bShort Interest\b|"
                         r"Weekly Analysts' Ratings", re.I)
NOT_THE_COMPANY = {  # same word, different entity
    "LIN": re.compile(r"Linde India|LINDEINDIA|Linde Pakistan|Linde Bangladesh|van der Linde|Linde Material "
                      r"Handling|Linde MH\b|Hoppstaedter|Linde Hydraulic|Linde Hydraulik|Linde Engineering India",
                      re.I),
    "DUK": re.compile(r"Duke University|Duke Health|Duke Realty|Blue Devils", re.I),
}
COMPANY_CUE = {  # names that are also people's names: headline/lead must carry a company cue too
    "LIN": re.compile(r"Linde (?:plc|PLC|AG|Inc|Gas|Gases|Hydrogen|Engineering|Group)|\(LIN\)|"
                      r"(?:NYSE|NASDAQ|Nasdaq)\s*:\s*LIN\b|\bLIN\b|Linde'?s\b|industrial gas|hydrogen|"
                      r"\bshares?\b|\bstock\b|analyst|price target|earnings|dividend|quarter|outlook|"
                      r"guidance|revenue|acquisition|acquire|plant|supply agreement|contract|projects?|"
                      r"investors?|takeaways|\$\s?\d|million|billion", re.I),
}
NOT_TOPIC = re.compile(r"\bT20I?\b|\bODI\b|wicket|\bSA20\b|Proteas|cricket|George Linde|Ann Linde|"
                       r"Foreign Minister|ambassador|embassy|bilateral|National Day|Grand Prix|Bathurst", re.I)
EN_STOP = re.compile(r"\b(the|and|of|to|in|for|is|on|with|that|by|as|at|from)\b", re.I)


# ---------------------------------------------------------------- DOCX -> articles
def doc_paragraphs(data: bytes) -> list:
    """Paragraph texts; a paragraph holding a Nexis document link also yields an '@@URN ...' line."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8", "replace")
    targets = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    targets.update({k: v for v, k in re.findall(r'Target="([^"]+)"[^>]*Id="([^"]+)"', rels)})
    out = []
    for p in root.iter(W + "p"):
        for h in p.iter(W + "hyperlink"):
            m = URN.search(targets.get(h.get(R + "id"), ""))
            if m:
                out.append("@@URN " + m.group(0))
        parts = []
        for node in p.iter():
            if node.tag == W + "t" and node.text:
                parts.append(node.text)
            elif node.tag in (W + "br", W + "cr"):
                parts.append(" ")
        out.append("".join(parts).strip())
    return out


def parse_article(lines: list):
    nonempty = [l for l in lines if l and not l.startswith("@@URN ")]
    idx = [i for i, l in enumerate(nonempty) if DATE_LINE.match(l)]
    if not idx or idx[0] < 2:
        return None              # a combined export's header / document list
    di = idx[0]
    rec = {"title": nonempty[di - 2], "publication": nonempty[di - 1], "date_line": nonempty[di],
           "byline": "", "language": "", "load_date": ""}
    body, in_body = [], False
    for l in nonempty[di + 1:]:
        if l == "Body":
            in_body = True
            continue
        m = FIELD.match(l)
        if m and (not in_body or m.group(1) in END_OF_BODY):
            key, val = m.group(1), m.group(2)
            if key in END_OF_BODY:
                in_body = False
            if key == "Byline":
                rec["byline"] = val
            elif key == "Language":
                rec["language"] = val
            elif key == "Load-Date":
                rec["load_date"] = val
            continue
        if in_body:
            body.append(l)
    rec["body"] = "\n".join(body)
    urns = [l.split(" ", 1)[1] for l in lines if l.startswith("@@URN ")]
    rec["nexis_urn"] = urns[0] if urns else ""
    return rec


def read_input(path: Path):
    """Yield (file name, docx bytes) for a .docx, a folder of them or a .zip of them."""
    if path.is_dir():
        for f in sorted(path.glob("*")):
            if f.suffix.lower() == ".docx":
                yield f.name, f.read_bytes()
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".docx"):
                    yield n, z.read(n)
    else:
        yield path.name, path.read_bytes()


def articles(path: Path):
    for fname, data in read_input(path):
        chunk = []
        for p in doc_paragraphs(data) + ["End of Document"]:
            if p == "End of Document":
                rec = parse_article(chunk)
                if rec:
                    rec.update({"input": path.name, "file": fname})
                    yield rec
                chunk = []
            else:
                chunk.append(p)


# ---------------------------------------------------------------- filters
def english(text: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ɏ]+", text)
    if len(words) < 30:
        return len(EN_STOP.findall(text)) >= 1 or len(words) < 12
    return len(EN_STOP.findall(text)) / len(words) >= 0.06


def utc_time(date_line: str):
    m = DATE_RE.match(date_line.strip())
    if not m:
        return None, "unparsed"
    day = datetime.strptime(m.group(1), "%B %d, %Y")
    if m.group(2) and m.group(4) in TZ_STD and f"{m.group(2)} {m.group(3)}" != "12:00 AM":
        t = datetime.strptime(f"{m.group(2)} {m.group(3)}", "%I:%M %p")
        local = day.replace(hour=t.hour, minute=t.minute)
        utc = (local - timedelta(hours=TZ_STD[m.group(4)])).replace(tzinfo=timezone.utc)
        return utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "exact"
    eod = day.replace(hour=23, minute=59, second=59, tzinfo=NY).astimezone(timezone.utc)
    return eod.strftime("%Y-%m-%dT%H:%M:%SZ"), "date_only"


def lead(body: str) -> str:
    first = next((p for p in body.split("\n") if len(p.strip()) > 40), body)
    return re.sub(r"\s+", " ", first).strip()[:500]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--batch", required=True, help="output name, e.g. nexisuni_batch02")
    ap.add_argument("--plan", required=True, help="nexis_download_plan.csv (date ranges per ticker)")
    ap.add_argument("inputs", nargs="+", help="TICKER=PATH (a .docx, a folder of them or a .zip)")
    a = ap.parse_args()

    uni = pd.read_csv("configs/universe.csv", dtype=str, keep_default_na=False)
    plan = pd.read_csv(a.plan, dtype=str)
    pats = {t: name_patterns(t, c) for t, c in zip(uni.ticker, uni.company)}
    company = dict(zip(uni.ticker, uni.company))

    def in_plan(ticker, day):
        for rng in plan[plan.ticker == ticker].date_ranges:
            for part in rng.split(";"):
                lo, hi = [x.strip() for x in part.split("→")]
                if lo <= day <= hi:
                    return True
        return False

    rows, meta, seen_urn, seen_story = [], [], set(), set()
    for spec in a.inputs:
        t, path = spec.split("=", 1)
        if t not in company:
            sys.exit(f"{t} is not in configs/universe.csv")
        for r in articles(Path(path)):
            pub_utc, status = utc_time(r["date_line"])
            lead_text = lead(r["body"])
            head = f"{r['title']} {lead_text}"
            reason, story = "", None
            if r["nexis_urn"] and r["nexis_urn"] in seen_urn:
                reason = "duplicate (same Nexis document)"
            elif pub_utc is None:
                reason = "unparsed date line"
            elif r["language"] and r["language"].lower() != "english":
                reason = f"language {r['language']}"
            elif not english(f"{r['title']} {r['body'][:1500]}"):
                reason = "not English (stop-word check)"
            elif PR_SOURCES.search(r["publication"]) or PR_TITLES.match(r["title"]):
                reason = "press release / regulatory / government feed"
            elif ROBO_SOURCES.search(r["publication"]):
                reason = "auto-generated holdings site"
            elif ROBO_TITLES.search(r["title"]):
                reason = "auto-generated holdings/comparison title"
            elif t in NOT_THE_COMPANY and NOT_THE_COMPANY[t].search(head):
                reason = "different entity with the same name"
            elif not any(p.search(head) for p in pats[t]):
                reason = "company not in headline or lead"
            elif NOT_TOPIC.search(head):
                reason = "not about the company (sport / politics / person with the same name)"
            elif t in COMPANY_CUE and not COMPANY_CUE[t].search(head):
                reason = "no company context for an ambiguous name"
            elif len(r["body"].split()) < 25:
                reason = "no article text"
            else:
                day = (datetime.strptime(pub_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                       .astimezone(NY).strftime("%Y-%m-%d"))
                story = (t, normalize_title(r["title"]), day)
                if not in_plan(t, day):
                    reason = "month already has enough news (outside the plan's ranges)"
                elif story in seen_story:
                    reason = "duplicate story (same headline, same day)"
            seen_urn.add(r["nexis_urn"])
            url = (f"https://advance.lexis.com/api/document?collection=news&id={r['nexis_urn']}"
                   if r["nexis_urn"] else f"nexis://{r['publication']}/{r['date_line']}/{r['title']}")
            meta.append({"ticker": t, "input": r["input"], "file": r["file"], "nexis_urn": r["nexis_urn"],
                         "url": url, "title": r["title"], "publication": r["publication"],
                         "date_line": r["date_line"], "time_status": status, "published_utc": pub_utc or "",
                         "load_date": r["load_date"], "language": r["language"] or "English (checked)",
                         "decision": "excluded" if reason else "kept", "reason": reason})
            if reason:
                continue
            seen_story.add(story)
            n_t = sum(1 for ps in pats.values() if any(p.search(head) for p in ps)) or 1
            rows.append({"news_id": "", "ticker": t, "company": company[t], "published_utc": pub_utc,
                         "updated_utc": pub_utc, "decision_date": "", "title": r["title"], "summary": lead_text,
                         "author": r["byline"] or r["publication"], "source": "NexisUni", "url": url,
                         "n_tickers": n_t, "in_title": "", "dup_group": "", "keep": ""})

    out = Path("data/raw/external") / f"{a.batch}.csv"
    prov = Path("data/raw/external/provenance") / f"{a.batch}_meta.csv"
    prov.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    m = pd.DataFrame(meta)
    m.to_csv(prov, index=False)
    print(f"{out}: {len(rows)} kept of {len(m)} articles   (log: {prov})")
    print(m[m.decision == "excluded"].reason.value_counts().to_string())


if __name__ == "__main__":
    main()
