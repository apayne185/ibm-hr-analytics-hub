# IBM HR Analytics Hub

An end-to-end people analytics pipeline: raw HR data → SQL analysis →
survival-model attrition prediction → a live dashboard. Built as a
hands-on project covering the SQL, Python, and BI workflow used in
modern people-analytics roles, developed with AI-assisted coding
tooling throughout — every non-obvious design or modeling decision along
the way is reviewed and recorded in [DECISIONS.md](DECISIONS.md).

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
tests/            pytest suite
.github/workflows/  CI
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

## Survival-model attrition prediction

A Cox Proportional Hazards model (`src/hr_analytics/survival_model.py`,
via `lifelines`) predicts attrition risk using each employee's real
tenure (`YearsAtCompany`) as duration and `Attrition` as the event,
right-censoring employees still on staff — see
[DECISIONS.md](DECISIONS.md) for why this framing fits the data better
than a plain classifier. Concordance index: **0.870 in-sample / 0.860
5-fold cross-validated**; the proportional-hazards assumption is checked
for every covariate (`docs/ph_assumptions_check.txt`). Outputs:
per-factor hazard ratios (`data/processed/survival_model_coefficients.csv`),
a per-employee risk score (`data/processed/predicted_attrition_risk.csv`),
Kaplan-Meier retention curves (`docs/figures/`), and a findings write-up
at [docs/survival_model_findings.md](docs/survival_model_findings.md).

## Dashboard

**Live demo:** *not yet deployed — see deployment steps below.*

A Streamlit app (`dashboard/app.py`) ties everything together: Overview
(headline KPIs), Attrition Drivers (interactive versions of the Phase 1
SQL cuts), Survival Model (hazard-ratio forest plot + Kaplan-Meier
curves, sourced from `data/processed/survival_model_metrics.json` rather
than hardcoded), Flight Risk Watchlist (filterable per-employee risk
table), and Hiring Pipeline (clearly labelled as synthetic). It rebuilds
`data/processed/*` and `docs/figures/*` on first run if they're missing,
so it works from a fresh clone with no other setup.

```bash
uv run streamlit run dashboard/app.py
```

### Deploying to Streamlit Community Cloud

`requirements.txt` (pip-installable, runtime deps only — no jupyter/pytest)
is committed for this; deploying is a one-time manual step since it
requires connecting your own GitHub account:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. "Create app" → pick this repo, branch `main`, main file path `dashboard/app.py`.
   Open **Advanced settings** and confirm/set the Python version to **3.12**
   — this repo requires it (`pyproject.toml`'s `requires-python`), and
   `.python-version` is only a suggested default, not guaranteed to be
   picked up automatically. Skipping this can fail the install outright.
3. Deploy. First boot takes ~15-30s (installs dependencies, then runs the
   self-bootstrap check — which only rebuilds `hr_analytics.db` from the
   already-committed CSVs, not a full model refit, so it's fast).
4. Update the live demo link above once you have the app's URL.

## Testing

`tests/` (pytest) covers the data-generation invariants (date ordering,
reproducibility under a fixed seed), the survival model (duration/event
validity, feature-column correctness), the SQLite loader (atomic-write
behavior under a simulated mid-load failure), and every SQL query
(runs cleanly + a regression test for the month-spine fix in query 10).
CI (`.github/workflows/ci.yml`) runs the suite on every push/PR, then
rebuilds the full pipeline end-to-end and fails the build if regenerating
it produces any diff against what's committed — the same class of
staleness bug caught twice during development (see DECISIONS.md).

```bash
uv run pytest tests/ -v
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

**Just want the dashboard?** `uv sync && uv run streamlit run dashboard/app.py`
is enough — every input file it needs is already committed, and it
self-bootstraps anything missing (see Dashboard section above). The full
sequence below is for regenerating or inspecting each phase's output
directly (e.g. after changing a SQL query or a model covariate) rather
than a required setup step.

```bash
uv sync
uv run python -m src.hr_analytics.synthetic_hiring   # regenerate data/processed/*
uv run python -m src.hr_analytics.load_db            # build the SQLite database
uv run python -m src.hr_analytics.run_queries        # run the SQL analysis queries
uv run python -m src.hr_analytics.survival_model     # fit the Cox model, generate risk scores + plots
uv run streamlit run dashboard/app.py                # launch the dashboard
```

## Status

- [x] Phase 0 — raw data in place, synthetic hiring pipeline extension generated
- [x] Phase 1 — SQL analysis (SQLite db + 10 business-question queries)
- [x] Phase 2 — survival-model attrition prediction (Cox PH, per-employee risk scores)
- [x] Phase 3 — dashboard (Streamlit, 5 tabs, self-bootstrapping)
- [x] Tests + CI (pytest suite, GitHub Actions pipeline-reproducibility check)

## License

[MIT](LICENSE)
