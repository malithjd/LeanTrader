"""Tests for the frozen universe (universe.py and configs/universe.csv)."""

import re

import pytest

from universe import COLUMNS, CSV_PATH, GICS_SECTORS, base_table, load_universe


@pytest.fixture
def base():
    return base_table()


def test_exactly_100_unique_tickers(base):
    assert len(base) == 100
    assert base["ticker"].is_unique


def test_goog_dropped_googl_kept(base):
    assert "GOOG" not in set(base["ticker"])
    assert "GOOGL" in set(base["ticker"])


def test_ticker_spelling(base):
    assert "BRK-B" in set(base["ticker"])
    assert not base["ticker"].str.contains(r"\.").any(), "use BRK-B, not BRK.B"
    assert base["ticker"].str.fullmatch(r"[A-Z]{1,5}(-[A-Z])?").all()


def test_sectors_are_gics(base):
    assert set(base["sector"]) <= GICS_SECTORS


def test_text_fields_filled_and_csv_safe(base):
    for col in ["company", "aliases", "sector"]:
        assert (base[col].str.len() > 0).all(), col
    for col in ["company", "aliases", "past_tickers"]:
        assert not base[col].str.contains(",").any(), f"no commas in {col}"


def test_aliases_include_company_short_name_or_first_word(base):
    for _, row in base.iterrows():
        aliases = [a.lower() for a in row["aliases"].split(";")]
        assert any(a in row["company"].lower() or row["company"].lower() in a for a in aliases), row["ticker"]


def test_past_tickers_not_in_universe(base):
    current = set(base["ticker"])
    past = {t for cell in base["past_tickers"] if cell for t in cell.split(";")}
    assert not past & current


@pytest.mark.skipif(not CSV_PATH.exists(), reason="configs/universe.csv not built yet (python universe.py --email ...)")
def test_built_csv_matches_list(base):
    df = load_universe()
    assert list(df.columns) == COLUMNS
    assert df["ticker"].tolist() == base["ticker"].tolist()
    assert df["cik"].map(lambda c: bool(re.fullmatch(r"\d{10}", c))).all(), "CIK must be 10 digits, zero-padded"
    assert df["cik"].is_unique, "two tickers share a CIK; GOOG should not be in the list"
