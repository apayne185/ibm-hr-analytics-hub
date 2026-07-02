from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hr_analytics.survival_model import (
    DURATION_COL,
    EVENT_COL,
    build_model_frame,
    feature_columns,
    fit_model,
)


def test_duration_is_always_positive(raw_df: pd.DataFrame) -> None:
    """CoxPHFitter requires strictly positive durations -- this is the
    invariant the YearsAtCompany.clip(lower=0.5) floor exists to guarantee."""
    model_df = build_model_frame(raw_df)
    assert (model_df[DURATION_COL] > 0).all()


def test_event_is_binary(raw_df: pd.DataFrame) -> None:
    model_df = build_model_frame(raw_df)
    assert set(model_df[EVENT_COL].unique()) <= {0, 1}


def test_no_missing_values(raw_df: pd.DataFrame) -> None:
    model_df = build_model_frame(raw_df)
    assert not model_df.isna().any().any()


def test_row_count_preserved(raw_df: pd.DataFrame) -> None:
    model_df = build_model_frame(raw_df)
    assert len(model_df) == len(raw_df)


def test_feature_columns_excludes_id_and_optionally_target(raw_df: pd.DataFrame) -> None:
    model_df = build_model_frame(raw_df)

    fit_cols = feature_columns(model_df, include_target=True)
    assert "EmployeeNumber" not in fit_cols
    assert DURATION_COL in fit_cols
    assert EVENT_COL in fit_cols

    predict_cols = feature_columns(model_df, include_target=False)
    assert "EmployeeNumber" not in predict_cols
    assert DURATION_COL not in predict_cols
    assert EVENT_COL not in predict_cols


def test_model_fits_and_scores_every_row(raw_df: pd.DataFrame) -> None:
    model_df = build_model_frame(raw_df)
    cph = fit_model(model_df)
    assert 0.5 < cph.concordance_index_ <= 1.0

    partial_hazard = cph.predict_partial_hazard(model_df[feature_columns(model_df, include_target=False)])
    assert len(partial_hazard) == len(model_df)
    assert partial_hazard.index.equals(model_df.index)
