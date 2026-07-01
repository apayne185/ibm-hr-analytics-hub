"""Run every .sql file in sql/queries/ against the SQLite db and save results as CSV.

Doubles as a smoke test that all the analysis queries still execute cleanly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/processed/hr_analytics.db")
QUERIES_DIR = Path("sql/queries")
RESULTS_DIR = Path("sql/results")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for query_path in sorted(QUERIES_DIR.glob("*.sql")):
            df = pd.read_sql_query(query_path.read_text(), conn)
            out_path = RESULTS_DIR / f"{query_path.stem}.csv"
            df.to_csv(out_path, index=False)
            print(f"{query_path.name}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
