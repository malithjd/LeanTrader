"""tests/test_teacher_labels.py -- spec 'Done means' checks for teacher_labels.csv.
Run from the repo root:  pytest tests/test_teacher_labels.py -q

Skipped until data/interim/teacher_labels.csv exists on the machine running the tests
(build it with `python teacher.py finalize`); the data is shared via the team folder, not committed.
"""
from pathlib import Path

import pandas as pd
import pytest

from news_prompt import EVENT_TYPES

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/interim/teacher_labels.csv"
NEWS = ROOT / "data/interim/news_clean.csv"
UNIVERSE = ROOT / "configs/universe.csv"

COLUMNS = ["news_id", "ticker", "split", "relevant", "event_type", "sentiment",
           "confidence", "rationale", "teacher_model", "cost_usd"]
SPLITS = {"train": (2018, 2022, 4000), "check": (2023, 2023, 1000), "post": (2025, 2025, 1000)}


@pytest.fixture(scope="module")
def labels():
    if not LABELS.exists():
        pytest.skip(f"{LABELS.relative_to(ROOT)} not present yet")
    return pd.read_csv(LABELS, dtype={"ticker": str}, keep_default_na=False)


def test_exact_columns_in_order(labels):
    assert list(labels.columns) == COLUMNS


def test_types(labels):
    assert pd.api.types.is_integer_dtype(labels["news_id"])
    assert pd.api.types.is_integer_dtype(labels["relevant"])
    for col in ["sentiment", "confidence", "cost_usd"]:
        assert pd.api.types.is_float_dtype(labels[col]), col


def test_no_nulls_in_key_or_label_columns(labels):
    for col in COLUMNS:
        assert (labels[col].astype(str).str.strip() != "").all(), f"empty values in {col}"


def test_no_duplicate_keys(labels):
    assert not labels.duplicated(["news_id", "ticker"]).any()


def test_split_sizes(labels):
    counts = labels["split"].value_counts().to_dict()
    assert counts == {name: n for name, (_, _, n) in SPLITS.items()}


def test_value_ranges(labels):
    assert set(labels["relevant"].unique()) <= {0, 1}
    assert set(labels["event_type"].unique()) <= set(EVENT_TYPES)
    assert labels["sentiment"].between(-1, 1).all()
    assert labels["confidence"].between(0, 1).all()
    assert (labels["cost_usd"] >= 0).all()


def test_one_teacher_model(labels):
    assert labels["teacher_model"].nunique() == 1


def test_tickers_in_universe(labels):
    universe = pd.read_csv(UNIVERSE, dtype=str)
    assert set(labels["ticker"]) <= set(universe["ticker"])


def test_rows_are_kept_news_in_their_split_years(labels):
    """Every label joins to a keep = 1 row of news_clean.csv, dated inside its split's years."""
    if not NEWS.exists():
        pytest.skip(f"{NEWS.relative_to(ROOT)} not present yet")
    news = pd.read_csv(NEWS, usecols=["news_id", "ticker", "decision_date", "keep"],
                       dtype={"ticker": str})
    m = labels.merge(news, on=["news_id", "ticker"], how="left")
    assert m["keep"].notna().all(), "labels not found in news_clean.csv"
    assert (m["keep"] == 1).all()
    year = m["decision_date"].str[:4].astype(int)
    for name, (y0, y1, _) in SPLITS.items():
        assert year[m["split"] == name].between(y0, y1).all(), name
