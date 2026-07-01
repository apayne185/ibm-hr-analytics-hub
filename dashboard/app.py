"""IBM HR Analytics Hub — live dashboard.

Reads the artifacts produced by the SQL and survival-model phases
(data/processed/, sql/results/, docs/figures/). Run `uv run streamlit run
dashboard/app.py` from the repo root; if those artifacts are missing, this
regenerates them on first load so the app works from a fresh clone.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DB_PATH = REPO_ROOT / "data/processed/hr_analytics.db"
RISK_PATH = REPO_ROOT / "data/processed/predicted_attrition_risk.csv"
COEF_PATH = REPO_ROOT / "data/processed/survival_model_coefficients.csv"
METRICS_PATH = REPO_ROOT / "data/processed/survival_model_metrics.json"
KM_OVERALL_PATH = REPO_ROOT / "docs/figures/km_overall.png"
KM_OVERTIME_PATH = REPO_ROOT / "docs/figures/km_by_overtime.png"

SYNTHETIC_NOTE = (
    "Dates and durations on this tab are **simulated**, not observed hiring "
    "data — generated from each employee's real tenure so the shape is "
    "plausible, but they should be read as illustrative, not factual. "
    "See DECISIONS.md."
)


@st.cache_resource
def ensure_pipeline_artifacts() -> None:
    """Regenerate data/processed/* and docs/figures/* if this is a fresh clone."""
    import hr_analytics.load_db as load_db
    import hr_analytics.survival_model as survival_model
    import hr_analytics.synthetic_hiring as synthetic_hiring

    if not (REPO_ROOT / "data/processed/synthetic_hiring_pipeline.csv").exists():
        synthetic_hiring.main()
    if not DB_PATH.exists():
        load_db.main()
    if not RISK_PATH.exists() or not METRICS_PATH.exists():
        survival_model.main()


@st.cache_data
def load_employees() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM employees", conn)


@st.cache_data
def load_hiring_pipeline() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT h.*, e.department, e.job_level FROM hiring_pipeline h "
            "JOIN employees e ON e.employee_number = h.employee_number",
            conn,
        )


@st.cache_data
def load_risk() -> pd.DataFrame:
    return pd.read_csv(RISK_PATH)


@st.cache_data
def load_coefficients() -> pd.DataFrame:
    return pd.read_csv(COEF_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def render_overview(employees: pd.DataFrame) -> None:
    st.header("Overview")

    headcount = len(employees)
    leavers = (employees["attrition"] == "Yes").sum()
    attrition_rate = 100 * leavers / headcount
    avg_tenure = employees["years_at_company"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headcount", f"{headcount:,}")
    c2.metric("Attrition rate", f"{attrition_rate:.1f}%")
    c3.metric("Leavers", f"{leavers:,}")
    c4.metric("Avg. tenure", f"{avg_tenure:.1f} yrs")

    by_dept = (
        employees.groupby("department")["attrition"]
        .apply(lambda s: 100 * (s == "Yes").mean())
        .reset_index(name="attrition_rate_pct")
        .sort_values("attrition_rate_pct", ascending=False)
    )
    fig = px.bar(by_dept, x="department", y="attrition_rate_pct", title="Attrition rate by department")
    fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="")
    st.plotly_chart(fig, width='stretch')


def render_attrition_drivers(employees: pd.DataFrame) -> None:
    st.header("Attrition Drivers")
    st.caption("Recreated from sql/queries/ — see docs/sql_findings.md for the full write-up.")

    col1, col2 = st.columns(2)

    with col1:
        by_overtime = (
            employees.groupby("over_time")["attrition"]
            .apply(lambda s: 100 * (s == "Yes").mean())
            .reset_index(name="attrition_rate_pct")
        )
        fig = px.bar(by_overtime, x="over_time", y="attrition_rate_pct", title="Attrition by overtime status", color="over_time")
        fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="Works overtime", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        bins = [-1, 0, 2, 4, 9, 100]
        labels = ["<1 year", "1-2 years", "3-4 years", "5-9 years", "10+ years"]
        tenure = employees.copy()
        tenure["tenure_bucket"] = pd.cut(tenure["years_at_company"], bins=bins, labels=labels)
        by_tenure = (
            tenure.groupby("tenure_bucket", observed=True)["attrition"]
            .apply(lambda s: 100 * (s == "Yes").mean())
            .reset_index(name="attrition_rate_pct")
        )
        fig = px.bar(by_tenure, x="tenure_bucket", y="attrition_rate_pct", title="Attrition by tenure bucket")
        fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="")
        st.plotly_chart(fig, width='stretch')

    by_role = (
        employees.groupby(["department", "job_role"])
        .agg(headcount=("employee_number", "count"), attrition_rate_pct=("attrition", lambda s: 100 * (s == "Yes").mean()))
        .reset_index()
    )
    by_role = by_role[by_role["headcount"] >= 10].sort_values("attrition_rate_pct", ascending=False)
    fig = px.bar(by_role, x="job_role", y="attrition_rate_pct", color="department", title="Attrition by role (roles with 10+ headcount)")
    fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="")
    st.plotly_chart(fig, width='stretch')


def render_survival_model(coefficients: pd.DataFrame, metrics: dict) -> None:
    st.header("Survival Model (Cox Proportional Hazards)")
    st.caption(
        "Duration = YearsAtCompany, event = Attrition, current employees right-censored. "
        "See docs/survival_model_findings.md and DECISIONS.md for the full framing and caveats."
    )

    c1, c2 = st.columns(2)
    c1.metric("Concordance (in-sample)", f"{metrics['concordance_in_sample']:.3f}")
    c2.metric(
        f"Concordance ({metrics['cv_folds']}-fold CV)",
        f"{metrics['concordance_cv_mean']:.3f}",
        help="Out-of-sample estimate — the one to trust for generalization.",
    )

    coef = coefficients.sort_values("exp(coef)", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coef["exp(coef)"],
            y=coef["covariate"],
            error_x=dict(
                type="data",
                symmetric=False,
                array=coef["exp(coef) upper 95%"] - coef["exp(coef)"],
                arrayminus=coef["exp(coef)"] - coef["exp(coef) lower 95%"],
            ),
            mode="markers",
            marker=dict(size=8),
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Hazard ratios (95% CI) — right of the line raises attrition risk, left lowers it",
        xaxis_title="Hazard ratio",
        height=500,
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Retention curves (Kaplan-Meier)")
    img_col1, img_col2 = st.columns(2)
    if KM_OVERALL_PATH.exists():
        img_col1.image(str(KM_OVERALL_PATH))
    if KM_OVERTIME_PATH.exists():
        img_col2.image(str(KM_OVERTIME_PATH))


def render_flight_risk(risk: pd.DataFrame) -> None:
    st.header("Flight Risk Watchlist")
    st.caption(
        "Current employees (Attrition = No), ranked by predicted hazard score from the survival "
        "model. Scored in-sample — a retention-prioritization tool, not a validated forecast."
    )

    current = risk[risk["Attrition"] == "No"].copy()

    departments = ["All"] + sorted(current["Department"].unique().tolist())
    selected_dept = st.selectbox("Department", departments)
    if selected_dept != "All":
        current = current[current["Department"] == selected_dept]

    top_n = st.slider("Show top N by risk", min_value=10, max_value=100, value=20, step=10)
    st.dataframe(
        current.sort_values("predicted_hazard_score", ascending=False).head(top_n)[
            ["EmployeeNumber", "Department", "JobRole", "predicted_hazard_score", "risk_percentile"]
        ],
        width='stretch',
        hide_index=True,
    )


def render_hiring_pipeline(hiring: pd.DataFrame) -> None:
    st.header("Hiring Pipeline")
    st.warning(SYNTHETIC_NOTE)

    by_level = (
        hiring.groupby(["department", "job_level"])["time_to_fill_days"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        by_level, x="job_level", y="time_to_fill_days", color="department", barmode="group",
        title="Avg. time-to-fill by job level and department (synthetic)",
    )
    fig.update_layout(yaxis_title="Avg. days to fill", xaxis_title="Job level")
    st.plotly_chart(fig, width='stretch')

    trend = hiring.copy()
    trend["hire_month"] = pd.to_datetime(trend["start_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = trend.groupby("hire_month").size().reset_index(name="hires")
    fig2 = px.line(monthly, x="hire_month", y="hires", title="Simulated hires per month over time (synthetic)")
    st.plotly_chart(fig2, width='stretch')


def main() -> None:
    st.set_page_config(page_title="IBM HR Analytics Hub", layout="wide")
    st.title("IBM HR Analytics Hub")
    st.caption("Raw HR data -> SQL analysis -> survival-model attrition prediction -> this dashboard.")

    ensure_pipeline_artifacts()

    employees = load_employees()
    hiring = load_hiring_pipeline()
    risk = load_risk()
    coefficients = load_coefficients()
    metrics = load_metrics()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Attrition Drivers", "Survival Model", "Flight Risk Watchlist", "Hiring Pipeline"]
    )
    with tab1:
        render_overview(employees)
    with tab2:
        render_attrition_drivers(employees)
    with tab3:
        render_survival_model(coefficients, metrics)
    with tab4:
        render_flight_risk(risk)
    with tab5:
        render_hiring_pipeline(hiring)


if __name__ == "__main__":
    main()
