from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hr_analytics.load_db as load_db
from hr_analytics.sql_tool import UnsafeQueryError, run_read_only_query, validate_query


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Same fresh-temp-db pattern as test_sql_queries.py."""
    tmp_path = tmp_path_factory.mktemp("sql_tool_test_db")
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


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO employees (employee_number) VALUES (9999)",
        "UPDATE employees SET age = 1",
        "DELETE FROM employees",
        "DROP TABLE employees",
        "ALTER TABLE employees ADD COLUMN x TEXT",
        "PRAGMA table_info(employees)",
        "ATTACH DATABASE 'x' AS y",
        "SELECT * FROM employees; DROP TABLE employees",
        "not sql at all",
    ],
)
def test_validate_query_rejects_unsafe_input(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_query(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM employees",
        "select department, count(*) from employees group by department",
        "WITH t AS (SELECT * FROM employees) SELECT * FROM t",
    ],
)
def test_validate_query_accepts_safe_select(sql: str) -> None:
    validated = validate_query(sql)
    assert "LIMIT 50" in validated.upper()


def test_validate_query_overrides_larger_user_supplied_limit() -> None:
    validated = validate_query("SELECT * FROM employees LIMIT 10000")
    assert validated.count("LIMIT") == 1
    assert "LIMIT 50" in validated.upper()


def test_run_read_only_query_returns_rows(db_path: Path) -> None:
    rows = run_read_only_query("SELECT employee_number, department FROM employees", db_path)
    assert len(rows) == 50  # MAX_ROWS cap, not the full 1470-row table
    assert set(rows[0].keys()) == {"employee_number", "department"}


def test_run_read_only_query_respects_aggregate_queries(db_path: Path) -> None:
    rows = run_read_only_query("SELECT COUNT(*) AS n FROM employees", db_path)
    assert rows == [{"n": 1470}]


def test_connection_is_physically_read_only(db_path: Path) -> None:
    """The real security boundary: even bypassing validate_query entirely and
    connecting the way run_read_only_query does, a write must fail at the
    SQLite/OS level, not just be rejected by the regex allowlist."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO employees (employee_number) VALUES (99999)")
    finally:
        conn.close()


def test_run_read_only_query_rejects_unsafe_sql_before_touching_db(db_path: Path) -> None:
    with pytest.raises(UnsafeQueryError):
        run_read_only_query("DELETE FROM employees", db_path)
