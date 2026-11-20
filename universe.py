"""The LeanTrader universe: the S&P 100 as of the Sept 2026 rebalance, minus GOOG.

That is 100 names. GOOG is dropped because it is the same company as GOOGL
(two share classes, near-identical returns), and keeping both would double
Alphabet's weight and make the table 101 rows per day.

The list is frozen. Later index changes do not change it, so backtests and the
forward test always use the same names. Survivorship bias from using today's
list is a known limitation and goes in the report.

Columns of configs/universe.csv:
    ticker        Yahoo/SEC spelling (BRK-B, not BRK.B). The key everywhere.
    company       Short common name. Goes into the LLM prompt.
    cik           10-digit SEC company id, zero-padded. Filled from SEC by build().
    sector        GICS sector. Used for the 30% sector cap.
    aliases       Names headlines use, joined with ';' (case-insensitive match).
                  Used for news `in_title`. Includes the short name itself.
    past_tickers  Earlier tickers for the same company, joined with ';'.
                  Use them when pulling old news or prices (e.g. BNY was BK
                  until May 21, 2026), then map rows back to `ticker`.

Usage:
    python universe.py --email you@fordham.edu    # MJ: builds configs/universe.csv once
    from universe import load_universe, tickers   # everyone else reads the CSV
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

UNIVERSE_AS_OF = "2026-09-21"  # S&P 100 after the Sept 2026 rebalance
CSV_PATH = Path(__file__).resolve().parent / "configs" / "universe.csv"
COLUMNS = ["ticker", "company", "cik", "sector", "aliases", "past_tickers"]
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

GICS_SECTORS = {
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
}

IT = "Information Technology"
HC = "Health Care"
FIN = "Financials"
IND = "Industrials"
CD = "Consumer Discretionary"
CS = "Consumer Staples"
COM = "Communication Services"

# (ticker, company, sector, aliases, past_tickers)
_ROWS: list[tuple[str, str, str, str, str]] = [
    ("AAPL", "Apple", IT, "Apple", ""),
    ("ABBV", "AbbVie", HC, "AbbVie", ""),
    ("ABT", "Abbott Laboratories", HC, "Abbott", ""),
    ("ACN", "Accenture", IT, "Accenture", ""),
    ("ADBE", "Adobe", IT, "Adobe", ""),
    ("AMAT", "Applied Materials", IT, "Applied Materials", ""),
    ("AMD", "AMD", IT, "AMD;Advanced Micro Devices", ""),
    ("AMGN", "Amgen", HC, "Amgen", ""),
    ("AMT", "American Tower", "Real Estate", "American Tower", ""),
    ("AMZN", "Amazon", CD, "Amazon", ""),
    ("ANET", "Arista Networks", IT, "Arista", ""),
    ("AVGO", "Broadcom", IT, "Broadcom", ""),
    ("AXP", "American Express", FIN, "American Express;Amex", ""),
    ("BA", "Boeing", IND, "Boeing", ""),
    ("BAC", "Bank of America", FIN, "Bank of America;BofA", ""),
    ("BKNG", "Booking Holdings", CD, "Booking Holdings;Booking.com;Priceline", "PCLN"),
    ("BLK", "BlackRock", FIN, "BlackRock", ""),
    ("BMY", "Bristol Myers Squibb", HC, "Bristol Myers;Bristol-Myers", ""),
    ("BNY", "BNY", FIN, "BNY;BNY Mellon;Bank of New York", "BK"),
    ("BRK-B", "Berkshire Hathaway", FIN, "Berkshire Hathaway;Berkshire", ""),
    ("C", "Citigroup", FIN, "Citigroup;Citi", ""),
    ("CAT", "Caterpillar", IND, "Caterpillar", ""),
    ("CMCSA", "Comcast", COM, "Comcast", ""),
    ("COF", "Capital One", FIN, "Capital One", ""),
    ("COP", "ConocoPhillips", "Energy", "ConocoPhillips;Conoco", ""),
    ("COST", "Costco", CS, "Costco", ""),
    ("CRM", "Salesforce", IT, "Salesforce", ""),
    ("CSCO", "Cisco", IT, "Cisco", ""),
    ("CVS", "CVS Health", HC, "CVS", ""),
    ("CVX", "Chevron", "Energy", "Chevron", ""),
    ("DE", "Deere", IND, "Deere;John Deere", ""),
    ("DELL", "Dell Technologies", IT, "Dell", ""),
    ("DHR", "Danaher", HC, "Danaher", ""),
    ("DIS", "Disney", COM, "Disney", ""),
    ("DUK", "Duke Energy", "Utilities", "Duke Energy", ""),
    ("EMR", "Emerson Electric", IND, "Emerson", ""),
    ("FDX", "FedEx", IND, "FedEx", ""),
    ("GD", "General Dynamics", IND, "General Dynamics", ""),
    ("GE", "GE Aerospace", IND, "GE Aerospace;General Electric", ""),
    ("GEV", "GE Vernova", IND, "GE Vernova", ""),
    ("GILD", "Gilead Sciences", HC, "Gilead", ""),
    ("GM", "General Motors", CD, "General Motors", ""),
    ("GOOGL", "Alphabet", COM, "Alphabet;Google", ""),
    ("GS", "Goldman Sachs", FIN, "Goldman Sachs;Goldman", ""),
    ("HD", "Home Depot", CD, "Home Depot", ""),
    ("IBM", "IBM", IT, "IBM", ""),
    ("INTC", "Intel", IT, "Intel", ""),
    ("INTU", "Intuit", IT, "Intuit", ""),
    ("ISRG", "Intuitive Surgical", HC, "Intuitive Surgical", ""),
    ("JNJ", "Johnson & Johnson", HC, "Johnson & Johnson;J&J", ""),
    ("JPM", "JPMorgan Chase", FIN, "JPMorgan;JPMorgan Chase;JP Morgan", ""),
    ("KO", "Coca-Cola", CS, "Coca-Cola;Coke", ""),
    ("LIN", "Linde", "Materials", "Linde;Praxair", ""),
    ("LLY", "Eli Lilly", HC, "Eli Lilly;Lilly", ""),
    ("LMT", "Lockheed Martin", IND, "Lockheed Martin;Lockheed", ""),
    ("LOW", "Lowe's", CD, "Lowe's;Lowes", ""),
    ("LRCX", "Lam Research", IT, "Lam Research", ""),
    ("MA", "Mastercard", FIN, "Mastercard", ""),
    ("MCD", "McDonald's", CD, "McDonald's;McDonalds", ""),
    ("MDLZ", "Mondelez", CS, "Mondelez;Mondelēz", ""),
    ("MDT", "Medtronic", HC, "Medtronic", ""),
    ("META", "Meta Platforms", COM, "Meta Platforms;Meta;Facebook", "FB"),
    ("MMM", "3M", IND, "3M", ""),
    ("MO", "Altria", CS, "Altria", ""),
    ("MRK", "Merck", HC, "Merck", ""),
    ("MS", "Morgan Stanley", FIN, "Morgan Stanley", ""),
    ("MSFT", "Microsoft", IT, "Microsoft", ""),
    ("MU", "Micron Technology", IT, "Micron", ""),
    ("NEE", "NextEra Energy", "Utilities", "NextEra", ""),
    ("NFLX", "Netflix", COM, "Netflix", ""),
    ("NOW", "ServiceNow", IT, "ServiceNow", ""),
    ("NVDA", "Nvidia", IT, "Nvidia", ""),
    ("ORCL", "Oracle", IT, "Oracle", ""),
    ("PANW", "Palo Alto Networks", IT, "Palo Alto Networks", ""),
    ("PEP", "PepsiCo", CS, "PepsiCo;Pepsi", ""),
    ("PFE", "Pfizer", HC, "Pfizer", ""),
    ("PG", "Procter & Gamble", CS, "Procter & Gamble;P&G", ""),
    ("PLTR", "Palantir", IT, "Palantir", ""),
    ("PM", "Philip Morris International", CS, "Philip Morris", ""),
    ("QCOM", "Qualcomm", IT, "Qualcomm", ""),
    ("RTX", "RTX", IND, "RTX;Raytheon;United Technologies", "UTX"),
    ("SBUX", "Starbucks", CD, "Starbucks", ""),
    ("SCHW", "Charles Schwab", FIN, "Schwab", ""),
    ("SNDK", "Sandisk", IT, "Sandisk", ""),
    ("SO", "Southern Company", "Utilities", "Southern Company;Southern Co", ""),
    ("T", "AT&T", COM, "AT&T", ""),
    ("TMO", "Thermo Fisher Scientific", HC, "Thermo Fisher", ""),
    ("TMUS", "T-Mobile", COM, "T-Mobile", ""),
    ("TSLA", "Tesla", CD, "Tesla", ""),
    ("TXN", "Texas Instruments", IT, "Texas Instruments", ""),
    ("UBER", "Uber", IND, "Uber", ""),
    ("UNH", "UnitedHealth Group", HC, "UnitedHealth", ""),
    ("UNP", "Union Pacific", IND, "Union Pacific", ""),
    ("UPS", "UPS", IND, "UPS;United Parcel Service", ""),
    ("USB", "U.S. Bancorp", FIN, "U.S. Bancorp;US Bancorp;U.S. Bank", ""),
    ("V", "Visa", FIN, "Visa", ""),
    ("VZ", "Verizon", COM, "Verizon", ""),
    ("WFC", "Wells Fargo", FIN, "Wells Fargo", ""),
    ("WMT", "Walmart", CS, "Walmart;Wal-Mart", ""),
    ("XOM", "ExxonMobil", "Energy", "ExxonMobil;Exxon", ""),
]


def base_table() -> pd.DataFrame:
    """The frozen list without CIKs (no network needed)."""
    df = pd.DataFrame(_ROWS, columns=["ticker", "company", "sector", "aliases", "past_tickers"])
    df.insert(2, "cik", "")
    return df[COLUMNS]


def fetch_ciks(email: str) -> dict[str, str]:
    """Ticker -> 10-digit CIK from SEC's public ticker file. SEC requires a contact email."""
    req = urllib.request.Request(
        SEC_TICKERS_URL,
        headers={"User-Agent": f"LeanTrader research project {email}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in data.values()}


def build(email: str, out: Path = CSV_PATH) -> pd.DataFrame:
    """Fill CIKs from SEC and write configs/universe.csv. Fails loudly on any unknown ticker."""
    df = base_table()
    ciks = fetch_ciks(email)
    missing = [t for t in df["ticker"] if t not in ciks]
    if missing:
        raise SystemExit(f"SEC has no CIK for: {missing}. Check the spelling before committing.")
    df["cik"] = df["ticker"].map(ciks)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    return df


def load_universe(path: Path = CSV_PATH) -> pd.DataFrame:
    """Read configs/universe.csv with every column as text (keeps CIK zeros)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Pull the latest repo; MJ builds it with `python universe.py`.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def tickers(path: Path = CSV_PATH) -> list[str]:
    """The 100 tickers in file order."""
    return load_universe(path)["ticker"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build configs/universe.csv with CIKs from SEC.")
    parser.add_argument("--email", required=True, help="Contact email for SEC's User-Agent rule")
    build(parser.parse_args().email)
