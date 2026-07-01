"""Load the raw + synthetic HR datasets into a SQLite database for SQL analysis.

Run after synthetic_hiring.py has produced data/processed/synthetic_hiring_pipeline.csv.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
PIPELINE_PATH = Path("data/processed/synthetic_hiring_pipeline.csv")
SCHEMA_PATH = Path("sql/schema.sql")
DB_PATH = Path("data/processed/hr_analytics.db")

# Raw CSV columns (CamelCase) -> employees table columns (snake_case).
EMPLOYEES_COLUMN_MAP = {
    "EmployeeNumber": "employee_number",
    "Age": "age",
    "Attrition": "attrition",
    "BusinessTravel": "business_travel",
    "DailyRate": "daily_rate",
    "Department": "department",
    "DistanceFromHome": "distance_from_home",
    "Education": "education",
    "EducationField": "education_field",
    "EnvironmentSatisfaction": "environment_satisfaction",
    "Gender": "gender",
    "HourlyRate": "hourly_rate",
    "JobInvolvement": "job_involvement",
    "JobLevel": "job_level",
    "JobRole": "job_role",
    "JobSatisfaction": "job_satisfaction",
    "MaritalStatus": "marital_status",
    "MonthlyIncome": "monthly_income",
    "MonthlyRate": "monthly_rate",
    "NumCompaniesWorked": "num_companies_worked",
    "Over18": "over_18",
    "OverTime": "over_time",
    "PercentSalaryHike": "percent_salary_hike",
    "PerformanceRating": "performance_rating",
    "RelationshipSatisfaction": "relationship_satisfaction",
    "StandardHours": "standard_hours",
    "StockOptionLevel": "stock_option_level",
    "TotalWorkingYears": "total_working_years",
    "TrainingTimesLastYear": "training_times_last_year",
    "WorkLifeBalance": "work_life_balance",
    "YearsAtCompany": "years_at_company",
    "YearsInCurrentRole": "years_in_current_role",
    "YearsSinceLastPromotion": "years_since_last_promotion",
    "YearsWithCurrManager": "years_with_curr_manager",
}

PIPELINE_COLUMN_MAP = {
    "EmployeeNumber": "employee_number",
    "requisition_open_date": "requisition_open_date",
    "offer_accepted_date": "offer_accepted_date",
    "start_date": "start_date",
    "time_to_fill_days": "time_to_fill_days",
    "offer_to_start_lag_days": "offer_to_start_lag_days",
    "time_to_hire_days": "time_to_hire_days",
}


def main() -> None:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    pipeline = pd.read_csv(PIPELINE_PATH)

    employees = raw.rename(columns=EMPLOYEES_COLUMN_MAP)[list(EMPLOYEES_COLUMN_MAP.values())]
    hiring_pipeline = pipeline.rename(columns=PIPELINE_COLUMN_MAP)[list(PIPELINE_COLUMN_MAP.values())]

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        employees.to_sql("employees", conn, if_exists="append", index=False)
        hiring_pipeline.to_sql("hiring_pipeline", conn, if_exists="append", index=False)

    print(f"Loaded {len(employees)} employees and {len(hiring_pipeline)} pipeline rows into {DB_PATH}")


if __name__ == "__main__":
    main()
