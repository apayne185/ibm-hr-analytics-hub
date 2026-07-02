from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hr_analytics.load_db as load_db


def _write_minimal_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    raw = pd.DataFrame(
        [
            {col: 0 for col in load_db.EMPLOYEES_COLUMN_MAP}
            | {"EmployeeNumber": 1, "Attrition": "No", "OverTime": "No"},
            {col: 0 for col in load_db.EMPLOYEES_COLUMN_MAP}
            | {"EmployeeNumber": 2, "Attrition": "Yes", "OverTime": "Yes"},
        ]
    )
    pipeline = pd.DataFrame(
        [
            {col: 0 for col in load_db.PIPELINE_COLUMN_MAP} | {"EmployeeNumber": 1},
            {col: 0 for col in load_db.PIPELINE_COLUMN_MAP} | {"EmployeeNumber": 2},
        ]
    )
    raw_path = tmp_path / "raw.csv"
    pipeline_path = tmp_path / "pipeline.csv"
    raw.to_csv(raw_path, index=False)
    pipeline.to_csv(pipeline_path, index=False)
    return raw_path, pipeline_path


def test_loads_both_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_path, pipeline_path = _write_minimal_fixtures(tmp_path)
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(load_db, "RAW_PATH", raw_path)
    monkeypatch.setattr(load_db, "PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(load_db, "SCHEMA_PATH", REPO_ROOT / "sql/schema.sql")
    monkeypatch.setattr(load_db, "DB_PATH", db_path)

    load_db.main()

    with sqlite3.connect(db_path) as conn:
        employees = pd.read_sql_query("SELECT * FROM employees", conn)
        pipeline = pd.read_sql_query("SELECT * FROM hiring_pipeline", conn)
    assert len(employees) == 2
    assert len(pipeline) == 2
    assert set(employees["employee_number"]) == {1, 2}


def test_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure partway through must not leave a partially-written db at DB_PATH."""
    raw_path, pipeline_path = _write_minimal_fixtures(tmp_path)
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(load_db, "RAW_PATH", raw_path)
    monkeypatch.setattr(load_db, "PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(load_db, "SCHEMA_PATH", REPO_ROOT / "sql/schema.sql")
    monkeypatch.setattr(load_db, "DB_PATH", db_path)

    # First, a real successful load establishes a "good" db to protect.
    load_db.main()
    before = db_path.read_bytes()

    _orig_to_sql = pd.DataFrame.to_sql

    def flaky_to_sql(self: pd.DataFrame, name: str, *args, **kwargs):
        if name == "hiring_pipeline":
            raise RuntimeError("simulated failure")
        return _orig_to_sql(self, name, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_sql", flaky_to_sql)

    with pytest.raises(RuntimeError, match="simulated failure"):
        load_db.main()

    monkeypatch.setattr(pd.DataFrame, "to_sql", _orig_to_sql)

    after = db_path.read_bytes()
    assert before == after, "a failed load must not modify the existing good database"
