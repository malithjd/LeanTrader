# LeanTrader
Cost efficient and faster framework for Market Research and Stock Recommendation

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # should pass before you start
```

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
data/raw/       raw API responses, one folder per source (gitignored)
data/interim/   our output CSVs (gitignored; share through the team folder)
tests/          one test file per table
```

## Building the universe (MJ, once)

```bash
python universe.py --email you@fordham.edu   # adds CIKs from SEC, writes configs/universe.csv
pytest tests/test_universe.py                # the CIK test runs once the CSV exists
```
