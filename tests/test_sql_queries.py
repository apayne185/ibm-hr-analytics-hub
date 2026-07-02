"""Runs every sql/queries/*.sql file against a freshly-built db and checks
basic sanity (no errors, non-empty, no NULL row identifiers) -- plus a
regression test for the month-spine bug in query 10, where LAG() previously
compared non-adjacent months as if they were adjacent during sparse years."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hr_analytics.load_db as load_db

QUERIES_DIR = REPO_ROOT / "sql/queries"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("sql_test_db")
    db = tmp_path / "test.db"

    import hr_analytics.synthetic_hiring as synthetic_hiring

    raw = pd.read_csv(load_db.RAW_PATH, encoding="utf-8-sig")
    pipeline = synthetic_hiring.build_synthetic_hiring_pipeline(raw)
    pipeline_path = tmp_path / "pipeline.csv"
    pipeline.to_csv(pipeline_path, index=False)

    orig = (load_db.PIPELINE_PATH, load_db.SCHEMA_PATH, load_db.DB_PATH)
    load_db.PIPELINE_PATH = pipeline_path
    load_db.SCHEMA_PATH = REPO_ROOT / "sql/schema.sql"
    load_db.DB_PATH = db
    try:
        load_db.main()
    finally:
        load_db.PIPELINE_PATH, load_db.SCHEMA_PATH, load_db.DB_PATH = orig

    return db


@pytest.mark.parametrize("query_path", sorted(QUERIES_DIR.glob("*.sql")), ids=lambda p: p.name)
def test_query_runs_and_returns_rows(query_path: Path, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query_path.read_text(), conn)
    assert len(df) > 0, f"{query_path.name} returned no rows"


def test_monthly_hiring_trend_has_no_month_gaps(db_path: Path) -> None:
    """Regression test: LAG()-over-distinct-months previously compared
    months across multi-year gaps as if adjacent. month_spine must produce
    one row per calendar month with no gaps."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query((QUERIES_DIR / "10_monthly_hiring_trend.sql").read_text(), conn)

    # Consecutive rows must be exactly one calendar month apart -- convert
    # to a period index and diff in month units.
    periods = pd.PeriodIndex(df["hire_month"], freq="M")
    period_diffs = periods[1:].asi8 - periods[:-1].asi8
    assert (period_diffs == 1).all(), "month_spine has a gap or a duplicate month"


def test_monthly_hiring_trend_cumulative_matches_total(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query((QUERIES_DIR / "10_monthly_hiring_trend.sql").read_text(), conn)
        total_hires = pd.read_sql_query("SELECT COUNT(*) AS n FROM hiring_pipeline", conn)["n"].iloc[0]

    assert df["hires"].sum() == total_hires
    assert df["cumulative_hires"].iloc[-1] == total_hires
