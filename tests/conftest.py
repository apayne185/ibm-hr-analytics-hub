from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RAW_PATH = Path(__file__).resolve().parent.parent / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, encoding="utf-8-sig")
