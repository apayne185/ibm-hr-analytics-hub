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

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import k_fold_cross_validation

ENRICHED_PATH = Path("data/processed/hr_employee_attrition_enriched.csv")
COEFFICIENTS_PATH = Path("data/processed/survival_model_coefficients.csv")
RISK_SCORES_PATH = Path("data/processed/predicted_attrition_risk.csv")
METRICS_PATH = Path("data/processed/survival_model_metrics.json")
PH_ASSUMPTIONS_PATH = Path("docs/ph_assumptions_check.txt")
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
PENALIZER = 0.1
CV_FOLDS = 5
CV_SEED = 42


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


def feature_columns(model_df: pd.DataFrame, include_target: bool) -> list[str]:
    """Covariate columns, optionally including duration/event for fitting."""
    exclude = {"EmployeeNumber"}
    if not include_target:
        exclude |= {DURATION_COL, EVENT_COL}
    return [c for c in model_df.columns if c not in exclude]


def fit_model(model_df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(model_df[feature_columns(model_df, include_target=True)], duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph


def cross_validated_concordance(model_df: pd.DataFrame, k: int = CV_FOLDS) -> np.ndarray:
    """Out-of-sample concordance via k-fold CV, since cph.concordance_index_ after
    fit_model() is an in-sample (optimistic) estimate."""
    cph = CoxPHFitter(penalizer=PENALIZER)
    fit_cols = feature_columns(model_df, include_target=True)
    scores = k_fold_cross_validation(
        cph, model_df[fit_cols], duration_col=DURATION_COL, event_col=EVENT_COL,
        k=k, scoring_method="concordance_index", seed=CV_SEED,
    )
    return np.array(scores)


def check_ph_assumptions(cph: CoxPHFitter, model_df: pd.DataFrame, p_value_threshold: float = 0.01) -> None:
    """Write the proportional-hazards diagnostic to a text file.

    Calls proportional_hazard_test() directly rather than
    CoxPHFitter.check_assumptions(): that method's own printing detects
    whether IPython is importable and, if so, renders its summary table via
    IPython's rich display instead of plain text -- which produces an opaque
    "<IPython.core.display.HTML object>" placeholder when its stdout is
    captured outside a real notebook. This project has IPython as a
    transitive dev dependency (for notebooks/), so relying on
    check_assumptions()'s environment-sensitive printing broke this output
    the moment that dependency was added. Building the table ourselves from
    the same underlying statistical test sidesteps that fragility entirely.
    """
    training_df = model_df[feature_columns(model_df, include_target=True)]
    residuals = cph.compute_residuals(training_df, kind="scaled_schoenfeld")
    test_results = proportional_hazard_test(
        cph, training_df, time_transform=["rank", "km"], precomputed_residuals=residuals
    )

    lines = [
        f"Proportional-hazards assumption check (p_value_threshold = {p_value_threshold}).",
        "Two time-transform tests (rank, km) per covariate; see docs/survival_model_findings.md for interpretation.",
        "",
        test_results.summary.round(6).to_string(),  # see coefficients rounding note in main()
        "",
    ]

    min_p_by_variable = test_results.summary["p"].groupby(level=0).min()
    failing = min_p_by_variable[min_p_by_variable < p_value_threshold]
    if failing.empty:
        lines.append("No covariate failed the proportional-hazards test at this threshold.")
    else:
        for variable, p in failing.items():
            lines.append(
                f"- '{variable}' failed the non-proportional test: p-value is {p:.4f}. "
                "Its hazard ratio is a time-averaged effect, not a constant one -- "
                "see docs/survival_model_findings.md."
            )

    PH_ASSUMPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PH_ASSUMPTIONS_PATH.write_text("\n".join(lines))
    print(f"Proportional-hazards assumption check written to {PH_ASSUMPTIONS_PATH}")


def _save_km_plot(groups: list[tuple[str, pd.Series, pd.Series]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, duration, event in groups:
        kmf = KaplanMeierFitter()
        kmf.fit(duration, event, label=label)
        kmf.plot_survival_function(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Proportion remaining")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kaplan_meier(raw: pd.DataFrame, duration: pd.Series, event: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _save_km_plot(
        [("All employees", duration, event)],
        "Overall retention curve (Kaplan-Meier)",
        FIGURES_DIR / "km_overall.png",
    )

    overtime = raw["OverTime"] == "Yes"
    _save_km_plot(
        [("OverTime = Yes", duration[overtime], event[overtime]), ("OverTime = No", duration[~overtime], event[~overtime])],
        "Retention curve by overtime status",
        FIGURES_DIR / "km_by_overtime.png",
    )

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
    print(f"\nIn-sample concordance index: {cph.concordance_index_:.3f}")

    cv_scores = cross_validated_concordance(model_df)
    print(f"{CV_FOLDS}-fold cross-validated concordance index: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    check_ph_assumptions(cph, model_df)

    COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Rounded before writing: CoxPHFitter's Newton-Raphson optimizer can
    # converge to slightly different values in the noise-floor digits across
    # platforms/BLAS backends, which showed up as a spurious CI diff on
    # full float64 precision. 6 decimals is far more precision than a hazard
    # ratio is ever interpreted at, and absorbs that cross-platform noise.
    cph.summary.round(6).to_csv(COEFFICIENTS_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {
                "concordance_in_sample": round(cph.concordance_index_, 3),
                "concordance_cv_mean": round(cv_scores.mean(), 3),
                "concordance_cv_std": round(cv_scores.std(), 3),
                "cv_folds": CV_FOLDS,
                "n_employees": len(model_df),
                "n_events": int(model_df[EVENT_COL].sum()),
            },
            indent=2,
        )
    )

    plot_kaplan_meier(raw, model_df[DURATION_COL], model_df[EVENT_COL])

    partial_hazard = cph.predict_partial_hazard(model_df[feature_columns(model_df, include_target=False)])

    # partial_hazard is a Series indexed like model_df, so assigning it directly
    # (rather than .to_numpy()) keeps every column aligned by index, not row
    # position -- safe even if a future change filters/reorders model_df's rows.
    risk = pd.DataFrame(
        {
            "EmployeeNumber": model_df["EmployeeNumber"],
            "Department": raw["Department"],
            "JobRole": raw["JobRole"],
            "Attrition": raw["Attrition"],
            "predicted_hazard_score": partial_hazard,
        }
    )
    risk["risk_percentile"] = risk["predicted_hazard_score"].rank(pct=True).round(3)
    risk["predicted_hazard_score"] = risk["predicted_hazard_score"].round(6)  # see coefficients rounding note above
    risk = risk.sort_values("predicted_hazard_score", ascending=False)
    risk.to_csv(RISK_SCORES_PATH, index=False)

    still_here = risk[risk["Attrition"] == "No"]
    print(f"\nTop 10 current employees by predicted attrition hazard:")
    print(still_here.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
