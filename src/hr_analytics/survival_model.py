"""Survival-model attrition prediction.

The dataset has no hire/termination dates, but it does give every
employee's tenure (YearsAtCompany) and whether they've left (Attrition).
That's exactly the shape a survival model wants: duration + event
indicator, with current employees right-censored at their tenure so far.
This fits a Cox Proportional Hazards model instead of a plain
classifier so the output is a hazard ratio per risk factor ("overtime
roughly doubles your hazard of leaving") rather than a single point
prediction, and produces per-employee risk scores for employees who are
still on staff. See DECISIONS.md for the duration/censoring assumptions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

ENRICHED_PATH = Path("data/processed/hr_employee_attrition_enriched.csv")
COEFFICIENTS_PATH = Path("data/processed/survival_model_coefficients.csv")
RISK_SCORES_PATH = Path("data/processed/predicted_attrition_risk.csv")
FIGURES_DIR = Path("docs/figures")

CATEGORICAL_COVARIATES = ["Department", "BusinessTravel", "MaritalStatus"]
NUMERIC_COVARIATES = [
    "OverTime",  # recoded to 0/1 below
    "JobLevel",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "DistanceFromHome",
    "NumCompaniesWorked",
    "YearsSinceLastPromotion",
    "StockOptionLevel",
    "log_monthly_income",
]

DURATION_COL = "duration_years"
EVENT_COL = "event_left"


def build_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)
    df["log_monthly_income"] = np.log(df["MonthlyIncome"])
    df[EVENT_COL] = (df["Attrition"] == "Yes").astype(int)

    # CoxPHFitter needs strictly positive durations; a handful of employees
    # have YearsAtCompany == 0 (hired within the last year), so treat them
    # as having survived half a year rather than dropping them.
    df[DURATION_COL] = df["YearsAtCompany"].clip(lower=0.5)

    model_df = df[[DURATION_COL, EVENT_COL, "EmployeeNumber"] + NUMERIC_COVARIATES + CATEGORICAL_COVARIATES]
    model_df = pd.get_dummies(model_df, columns=CATEGORICAL_COVARIATES, drop_first=True)
    return model_df


def fit_model(model_df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=0.1)
    feature_cols = [c for c in model_df.columns if c not in ("EmployeeNumber",)]
    cph.fit(model_df[feature_cols], duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph


def plot_kaplan_meier(raw: pd.DataFrame, duration: pd.Series, event: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf.fit(duration, event, label="All employees")
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Overall retention curve (Kaplan-Meier)")
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Proportion remaining")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "km_overall.png", dpi=150)
    plt.close(fig)

    overtime = (raw["OverTime"] == "Yes")
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, mask in [("OverTime = Yes", overtime), ("OverTime = No", ~overtime)]:
        kmf = KaplanMeierFitter()
        kmf.fit(duration[mask], event[mask], label=label)
        kmf.plot_survival_function(ax=ax)
    ax.set_title("Retention curve by overtime status")
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Proportion remaining")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "km_by_overtime.png", dpi=150)
    plt.close(fig)

    result = logrank_test(
        duration[overtime], duration[~overtime],
        event_observed_A=event[overtime], event_observed_B=event[~overtime],
    )
    print(f"Log-rank test, overtime vs. no overtime: p = {result.p_value:.2e}")


def main() -> None:
    raw = pd.read_csv(ENRICHED_PATH, encoding="utf-8-sig")
    model_df = build_model_frame(raw)

    cph = fit_model(model_df)
    print(cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False))
    print(f"\nConcordance index: {cph.concordance_index_:.3f}")

    COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cph.summary.to_csv(COEFFICIENTS_PATH)

    plot_kaplan_meier(raw, model_df[DURATION_COL], model_df[EVENT_COL])

    feature_cols = [c for c in model_df.columns if c not in ("EmployeeNumber", DURATION_COL, EVENT_COL)]
    partial_hazard = cph.predict_partial_hazard(model_df[feature_cols])

    risk = pd.DataFrame(
        {
            "EmployeeNumber": model_df["EmployeeNumber"],
            "Department": raw["Department"],
            "JobRole": raw["JobRole"],
            "Attrition": raw["Attrition"],
            "predicted_hazard_score": partial_hazard.to_numpy(),
        }
    )
    risk["risk_percentile"] = risk["predicted_hazard_score"].rank(pct=True).round(3)
    risk = risk.sort_values("predicted_hazard_score", ascending=False)
    risk.to_csv(RISK_SCORES_PATH, index=False)

    still_here = risk[risk["Attrition"] == "No"]
    print(f"\nTop 10 current employees by predicted attrition hazard:")
    print(still_here.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
