"""tests/test_news.py -- spec 'Done means' checks for news_clean.csv and news_coverage.csv.
Run from the repo root:  pytest tests/test_news.py -q
"""
from pathlib import Path

import pandas as pd
import pytest

NEWS = Path("data/interim/news_clean.csv")
COVERAGE = Path("data/interim/news_coverage.csv")
UNIVERSE = Path("configs/universe.csv")

COLUMNS = [
    "news_id", "ticker", "company", "published_utc", "updated_utc",
    "decision_date", "title", "summary", "author", "source", "url",
    "n_tickers", "in_title", "dup_group", "keep",
]
ISO_Z = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


@pytest.fixture(scope="module")
def news():
    return pd.read_csv(NEWS, dtype={"ticker": str, "summary": str, "author": str},
                       keep_default_na=False)


@pytest.fixture(scope="module")
def universe():
    return pd.read_csv(UNIVERSE, dtype=str)


def test_exact_columns_in_order(news):
    assert list(news.columns) == COLUMNS


def test_no_duplicate_keys(news):
    assert not news.duplicated(["news_id", "ticker"]).any()


def test_no_nulls_in_key_columns(news):
    for col in ["news_id", "ticker", "company", "published_utc", "decision_date",
                "title", "dup_group", "keep"]:
        assert (news[col].astype(str).str.len() > 0).all(), col


def test_types_and_formats(news):
    assert pd.api.types.is_integer_dtype(news["news_id"])
    assert pd.api.types.is_integer_dtype(news["n_tickers"])
    assert news["published_utc"].str.match(ISO_Z).all()
    assert news["updated_utc"].str.match(ISO_Z).all()
    assert news["decision_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert set(news["in_title"].unique()) <= {0, 1}
    assert set(news["keep"].unique()) <= {0, 1}


def test_tickers_and_companies_match_universe(news, universe):
    assert set(news["ticker"]) <= set(universe["ticker"])
    m = news.merge(universe[["ticker", "company"]], on="ticker", suffixes=("", "_u"))
    assert (m["company"] == m["company_u"]).all()


def test_date_range(news):
    pub = pd.to_datetime(news["published_utc"], utc=True)
    assert pub.min() >= pd.Timestamp("2018-01-01", tz="UTC")
    assert pub.max() <= pd.Timestamp.now(tz="UTC")


def test_decision_date_never_before_publication(news):
    # compare in New York time: an article can't count for a day before it existed
    pub_et = pd.to_datetime(news["published_utc"], utc=True).dt.tz_convert("America/New_York")
    assert (pd.to_datetime(news["decision_date"]) >= pub_et.dt.tz_localize(None).dt.normalize()).all()


def test_after_4pm_rolls_forward(news):
    pub_et = pd.to_datetime(news["published_utc"], utc=True).dt.tz_convert("America/New_York")
    late = pub_et.dt.hour >= 16
    same_day = pd.to_datetime(news["decision_date"]) == pub_et.dt.tz_localize(None).dt.normalize()
    assert not (late & same_day).any()


def test_kept_rows_pass_relevance(news):
    kept = news[news["keep"] == 1]
    assert ((kept["n_tickers"] <= 3) | (kept["in_title"] == 1)).all()


def test_one_kept_row_per_dup_group(news):
    assert news[news["keep"] == 1]["dup_group"].value_counts().max() == 1


def test_daily_cap(news):
    kept = news[news["keep"] == 1]
    assert kept.groupby(["ticker", "decision_date"]).size().max() <= 5


def test_failing_rows_are_kept_not_dropped(news):
    # the file must still contain rejected rows (keep = 0) so filters can be retuned
    assert (news["keep"] == 0).any()


def test_coverage_file(news):
    cov = pd.read_csv(COVERAGE)
    assert list(cov.columns) == ["ticker", "year", "n_articles", "n_kept"]
    assert cov["n_articles"].sum() == len(news)
    assert cov["n_kept"].sum() == news["keep"].sum()
