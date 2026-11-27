# LeanTrader
Cost efficient and faster framework for Market Research and Stock Recommendation

## Architecture

![LeanTrader architecture: news, SEC filings and prices become one daily feature table; a ranker (LightGBM, GRU, LSTM comparison) orders the 100 stocks and a rule-based portfolio trades the top 10](docs/leantrader-architecture.png)

A fine-tuned Llama 3.2 3B reads the news. Plain code turns SEC filings and prices into features.
LightGBM (main), a GRU over a 60-day window and an LSTM (comparison) rank the stocks, and fixed rules
make the trade. Offline, a Gemini Flash teacher labels news once to fine-tune the 3B model.

To edit the diagram, open `docs/leantrader-architecture.excalidraw` in [excalidraw.com](https://excalidraw.com),
then export it over `docs/leantrader-architecture.png`.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ml.txt # only if you run model scripts (FinBERT, Llama)
pytest                             # should pass before you start
```

**Adding a package?** Put it in `requirements.txt` (core) or `requirements-ml.txt` (GPU/model)
with a minimum version, `package>=X.Y`, where X.Y comes from `pip show <package>`.
A test blocks the merge if you forget, and Claude's review flags it (rule in `CLAUDE.md`).

## Shared pieces (use these, don't rewrite them)

**Universe.** `configs/universe.csv` holds the 100 tickers (S&P 100 as of the Sept 2026 rebalance, minus GOOG).
Columns: `ticker`, `company`, `cik`, `sector`, `aliases`, `past_tickers`.

```python
from universe import load_universe, tickers
uni = load_universe()      # DataFrame, every column as text
syms = tickers()           # ['AAPL', 'ABBV', ...]
```

- Use `ticker` spelling everywhere (`BRK-B`, not `BRK.B`).
- `aliases` are the names headlines use, `;`-joined. Use them for the news `in_title` check.
- `past_tickers` are older tickers of the same company (BNY was `BK` until May 21, 2026; META was `FB`).
  Query them too when pulling old news or prices, then map rows back to `ticker`.

**Dating rule.** Every table sets `decision_date` with this function and nothing else.

```python
from dating import decision_date, trading_days
news["decision_date"] = decision_date(news["published_utc"])   # UTC strings ending in Z
```

Before 4:00 pm New York time on a trading day means that day; 4:00 pm or later means the next
trading day; weekends, holidays and special closures roll forward. Timestamps without a timezone
are refused on purpose.

## Folders

```
configs/        universe.csv (committed)
docs/           architecture diagram (.excalidraw source + .png for the README)
data/raw/       raw API responses, one folder per source (gitignored)
data/interim/   our output CSVs (gitignored; share through the team folder)
tests/          one test file per table
```

## News (Nati)

`news.py` pulls news from three sources and merges them. Each pull is cached under
`data/raw/<source>/`, so a rerun skips what you already have.

| Source  | Dates                          | Needs                           |
|---------|--------------------------------|---------------------------------|
| FNSPID  | 2018-01-01 to 2023-12-31       | nothing (public Hugging Face dataset), `pip install -r requirements-ml.txt` |
| Massive | about the last 2 years         | `MASSIVE_API_KEY`               |
| Alpaca  | recent news (~Apr 2026 onward) | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` |

No free source covers 2024-01-01 to about Nov 2024.

**External gap-fill (`source = external`, added Nov 27 by MJ).** 179 ticker-years (2018 to Oct 2024)
had fewer than 10 articles, not counting years before a company listed. We added 454 hand-sourced
articles for them from public publisher archives, sitemaps and RSS feeds. No news API or Nexis
was used. The largest publishers are Zacks, TheStreet, GovCon Wire, gasworld, Benzinga and
Fox Business. Each `published_utc` is the publisher's own timestamp. The file is
`data/raw/external/added_articles.csv` (team folder), with audit columns `publisher`,
`time_status`, `date_evidence` and `method`. The rows are headlines only (`summary` is empty) and
go through the same relevance, de-dup and daily-cap rules.

- `python news.py build` includes the file automatically.
- `python news.py add-external` merges it into an existing `news_clean.csv` when you don't have
  the raw caches. Existing rows are left unchanged.
- **It only partly closes the gaps.** 72 of the 179 ticker-years gained articles, and only 11 now
  have 10 or more. 77 still have none, including AAPL, AMZN, MSFT, META, GOOGL, JPM, BAC, MA and
  NFLX in 2021–2023. A filled year has a median of about 7 articles, against about 400 for a
  normal ticker-year. Keep treating `has_news = 0` in those years as missing data.
- **Possible selection bias.** These articles were searched for in 2026, so they may lean
  toward stories that turned out to matter. Report external rows separately where it counts.
- The Jan–Oct 2024 gap is still open for every ticker that wasn't on the gap list.

**API keys.** Copy the template and fill in your own keys. `.env` is gitignored, so never commit it.

```bash
cp .env.example .env
```

- Massive: sign up at [massive.com](https://massive.com); the free plan's key works.
- Alpaca: sign up at [alpaca.markets](https://alpaca.markets), then Dashboard → API Keys
  (paper-trading keys work for news).

`news.py` reads `.env` when it starts. Keys you export in the shell take priority.

```bash
python news.py pull-fnspid                       # Benzinga file, to Jun 2020 (5.7 GB download)
python news.py pull-fnspid --file nasdaq-lite    # Nasdaq file, to 2023 (1.8 GB)
python news.py pull-massive                      # add --tickers AAPL,MSFT to pull a few
python news.py pull-alpaca
python news.py build                             # -> data/interim/news_clean.csv, news_coverage.csv
```

You only need keys if you pull news yourself. The finished CSVs are shared through the team folder.

## Building the universe (MJ, once)

```bash
python universe.py --email you@fordham.edu   # adds CIKs from SEC, writes configs/universe.csv
pytest tests/test_universe.py                # the CIK test runs once the CSV exists
```
