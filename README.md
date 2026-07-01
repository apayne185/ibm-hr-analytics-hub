# IBM HR Analytics Hub

An end-to-end people analytics pipeline: raw HR data → SQL analysis →
survival-model attrition prediction → a live dashboard. Built as a
hands-on project covering the SQL, Python, BI, and GenAI-assisted
analysis workflow used in modern people-analytics roles.

## Data

- **Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/)
  dataset (also distributed on Kaggle), 1,470 employees with department,
  tenure, satisfaction, income, overtime, and attrition fields.
  `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` — unmodified.
- **Synthetic extension:** the raw dataset has no recruiting/time-to-hire
  fields, so a simulated requisition-to-hire pipeline
  (`src/hr_analytics/synthetic_hiring.py`) is joined on `EmployeeNumber`
  to add `requisition_open_date`, `offer_accepted_date`, `start_date`,
  `time_to_fill_days`, and `time_to_hire_days`. **This pipeline data is
  synthetic**, generated to be internally consistent with each employee's
  real tenure — see [DECISIONS.md](DECISIONS.md) for exactly how and why.

Outputs land in `data/processed/`:
- `synthetic_hiring_pipeline.csv` — the simulated pipeline table alone.
- `hr_employee_attrition_enriched.csv` — raw dataset + pipeline table joined.

## Repo structure

```
data/
  raw/            unmodified source data
  processed/      derived/enriched datasets
src/
  hr_analytics/   Python package (data generation, analysis code)
sql/              SQL analysis queries
notebooks/        exploratory analysis
dashboard/        BI dashboard
docs/             supporting documentation
DECISIONS.md      log of non-obvious modelling/design decisions
```

## SQL analysis

The enriched dataset is loaded into a SQLite database
(`data/processed/hr_analytics.db`) with two tables — `employees` (HRIS
export) and `hiring_pipeline` (synthetic ATS export) — joined on
`employee_number`, mirroring how this data would actually land in a
warehouse. `sql/schema.sql` defines the tables; `sql/queries/` holds ten
business-question queries (attrition drivers, income-percentile risk,
flight-risk ranking, time-to-hire trends) covering aggregates, CTEs,
window functions, and joins. `sql/results/` holds their output as CSV, and
[docs/sql_findings.md](docs/sql_findings.md) summarizes what they show.

```bash
uv run python -m src.hr_analytics.load_db      # build data/processed/hr_analytics.db
uv run python -m src.hr_analytics.run_queries   # run sql/queries/*.sql -> sql/results/*.csv
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m src.hr_analytics.synthetic_hiring   # regenerate data/processed/*
uv run python -m src.hr_analytics.load_db            # build the SQLite database
uv run python -m src.hr_analytics.run_queries        # run the SQL analysis queries
```

## Status

- [x] Phase 0 — raw data in place, synthetic hiring pipeline extension generated
- [x] Phase 1 — SQL analysis (SQLite db + 10 business-question queries)
- [ ] Survival-model attrition prediction
- [ ] Dashboard
