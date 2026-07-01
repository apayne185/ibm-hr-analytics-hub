"""Simulate a requisition-to-hire pipeline for the IBM HR Attrition dataset.

The Kaggle dataset has no recruiting/time-to-hire fields, only YearsAtCompany.
This module backs out a plausible hire_date per employee from a fixed
snapshot date, then simulates the recruiting milestones that would precede
it (requisition opened -> offer accepted -> start date). All generated
fields are synthetic and are prefixed/labelled accordingly; see
DECISIONS.md for the modelling assumptions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUT_ENRICHED_PATH = Path("data/processed/hr_employee_attrition_enriched.csv")
OUT_PIPELINE_PATH = Path("data/processed/synthetic_hiring_pipeline.csv")

SNAPSHOT_DATE = pd.Timestamp("2024-01-15")
SEED = 42

# Median days-to-fill by job level (1 = entry, 5 = executive), before
# department adjustment. Roughly follows real-world recruiting benchmarks:
# junior roles fill faster than senior/specialist roles.
BASE_MEDIAN_DAYS_BY_LEVEL = {1: 20, 2: 28, 3: 38, 4: 48, 5: 60}

# Multiplier on top of the job-level baseline, reflecting that technical
# R&D roles typically run more interview loops than HR/Sales.
DEPARTMENT_MULTIPLIER = {
    "Research & Development": 1.15,
    "Sales": 1.0,
    "Human Resources": 0.9,
}


def _simulate_time_to_fill(job_level: pd.Series, department: pd.Series, rng: np.random.Generator) -> np.ndarray:
    base_median = job_level.map(BASE_MEDIAN_DAYS_BY_LEVEL).astype(float)
    multiplier = department.map(DEPARTMENT_MULTIPLIER).fillna(1.0)
    median_days = base_median * multiplier

    # Lognormal gives the right-skewed shape real time-to-fill data has
    # (most reqs close quickly, a long tail of hard-to-fill roles).
    mu = np.log(median_days)
    sigma = 0.4
    days = rng.lognormal(mean=mu, sigma=sigma)
    return np.clip(days, 7, 180).round().astype(int)


def _simulate_offer_to_start_lag(job_level: pd.Series, rng: np.random.Generator) -> np.ndarray:
    # Higher-level hires tend to serve longer notice periods.
    low = 7 + job_level.to_numpy() * 2
    high = 21 + job_level.to_numpy() * 4
    return rng.integers(low, high + 1).astype(int)


def build_synthetic_hiring_pipeline(raw: pd.DataFrame, seed: int = SEED, snapshot_date: pd.Timestamp = SNAPSHOT_DATE) -> pd.DataFrame:
    """Return a DataFrame of synthetic requisition->hire dates keyed by EmployeeNumber."""
    rng = np.random.default_rng(seed)
    n = len(raw)

    # Spread hire anniversaries across the year instead of all landing on
    # the same day-of-year as the snapshot date. Jitter is subtracted (not
    # added) so a hire_date never lands after the snapshot date.
    day_jitter = pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
    years_at_company = pd.to_timedelta(raw["YearsAtCompany"].to_numpy() * 365.25, unit="D")
    hire_date = snapshot_date - years_at_company - day_jitter
    hire_date = hire_date.normalize()

    time_to_fill_days = _simulate_time_to_fill(raw["JobLevel"], raw["Department"], rng)
    offer_to_start_lag_days = _simulate_offer_to_start_lag(raw["JobLevel"], rng)

    offer_accepted_date = hire_date - pd.to_timedelta(offer_to_start_lag_days, unit="D")
    requisition_open_date = offer_accepted_date - pd.to_timedelta(time_to_fill_days, unit="D")

    pipeline = pd.DataFrame(
        {
            "EmployeeNumber": raw["EmployeeNumber"],
            "requisition_open_date": requisition_open_date,
            "offer_accepted_date": offer_accepted_date,
            "start_date": hire_date,
            "time_to_fill_days": time_to_fill_days,
            "offer_to_start_lag_days": offer_to_start_lag_days,
        }
    )
    # time_to_hire spans requisition open through the employee's actual first day.
    pipeline["time_to_hire_days"] = (pipeline["start_date"] - pipeline["requisition_open_date"]).dt.days
    return pipeline


def main() -> None:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    pipeline = build_synthetic_hiring_pipeline(raw)

    OUT_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_csv(OUT_PIPELINE_PATH, index=False)

    enriched = raw.merge(pipeline, on="EmployeeNumber", how="left")
    enriched.to_csv(OUT_ENRICHED_PATH, index=False)

    print(f"Wrote {len(pipeline)} rows to {OUT_PIPELINE_PATH}")
    print(f"Wrote {len(enriched)} rows to {OUT_ENRICHED_PATH}")
    print(pipeline["time_to_hire_days"].describe())


if __name__ == "__main__":
    main()
