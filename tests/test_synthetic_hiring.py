from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hr_analytics.synthetic_hiring import SNAPSHOT_DATE, build_synthetic_hiring_pipeline


def test_one_row_per_employee(raw_df: pd.DataFrame) -> None:
    pipeline = build_synthetic_hiring_pipeline(raw_df)
    assert len(pipeline) == len(raw_df)
    assert pipeline["EmployeeNumber"].is_unique


def test_no_dates_after_snapshot(raw_df: pd.DataFrame) -> None:
    pipeline = build_synthetic_hiring_pipeline(raw_df)
    assert (pipeline["start_date"] <= SNAPSHOT_DATE).all()


def test_milestone_ordering(raw_df: pd.DataFrame) -> None:
    """requisition_open -> offer_accepted -> start_date must never invert."""
    pipeline = build_synthetic_hiring_pipeline(raw_df)
    assert (pipeline["requisition_open_date"] < pipeline["offer_accepted_date"]).all()
    assert (pipeline["offer_accepted_date"] < pipeline["start_date"]).all()


def test_time_to_hire_is_positive_and_consistent(raw_df: pd.DataFrame) -> None:
    pipeline = build_synthetic_hiring_pipeline(raw_df)
    assert (pipeline["time_to_hire_days"] > 0).all()
    expected = (pipeline["start_date"] - pipeline["requisition_open_date"]).dt.days
    assert (pipeline["time_to_hire_days"] == expected).all()


def test_reproducible_with_same_seed(raw_df: pd.DataFrame) -> None:
    first = build_synthetic_hiring_pipeline(raw_df, seed=42)
    second = build_synthetic_hiring_pipeline(raw_df, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_output(raw_df: pd.DataFrame) -> None:
    first = build_synthetic_hiring_pipeline(raw_df, seed=42)
    second = build_synthetic_hiring_pipeline(raw_df, seed=7)
    assert not first["start_date"].equals(second["start_date"])
