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


def test_overtime_is_the_dominant_risk_factor(raw_df: pd.DataFrame) -> None:
    """Exact coefficient values aren't bit-reproducible across platforms
    (CoxPHFitter's optimizer can converge to meaningfully different values
    depending on the BLAS backend -- see DECISIONS.md), so this checks the
    headline finding stays true within a wide tolerance band instead of
    pinning an exact number: OverTime should come out as a large,
    statistically significant risk factor every time this is fit."""
    model_df = build_model_frame(raw_df)
    cph = fit_model(model_df)

    overtime_row = cph.summary.loc["OverTime"]
    assert overtime_row["p"] < 0.01, "OverTime should be a statistically significant hazard factor"
    assert 1.5 < overtime_row["exp(coef)"] < 3.0, "OverTime's hazard ratio drifted outside the expected range"
