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
reports/          generated Excel executive report
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
table), Hiring Pipeline (clearly labelled as synthetic), and Ask the
Data (a RAG + tool-calling chat agent — see below). It rebuilds
`data/processed/*` and `docs/figures/*` on first run if they're missing,
so it works from a fresh clone with no other setup.

```bash
uv run streamlit run dashboard/app.py
```

### Ask the Data (RAG + agent tool-use)

The 6th tab is a chat agent, not a scripted view: it retrieves relevant
context from this project's own docs (`docs/sql_findings.md`,
`docs/survival_model_findings.md`, `docs/ph_assumptions_check.txt`,
`DECISIONS.md` — TF-IDF + cosine similarity, chunked on each doc's own
section headers) and calls into the SQL database as tools
(`src/hr_analytics/sql_tool.py` — a constrained, read-only query tool,
plus two parameterized tools for the most common questions) to answer
questions like *"what's the attrition rate in Sales?"* or *"why does
overtime matter so much?"*.

- **Provider-agnostic:** `src/hr_analytics/llm_providers.py` defines a
  vendor-neutral interface (`AnthropicProvider`, `OpenAIProvider`, and a
  `FakeProvider` used throughout the test suite) — nothing else in the
  app codes against a specific vendor's API shape.
- **Zero API key, zero crash:** every one of the other 5 tabs works
  identically with no LLM configured. Without a key, this tab shows a
  short setup message instead of an error.
- **To enable it:**
  ```bash
  uv sync --extra llm   # installs anthropic + openai + python-dotenv (opt-in, not a hard dependency)
  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env   # gitignored; or OPENAI_API_KEY, or export it directly
  uv run streamlit run dashboard/app.py
  ```
  `HR_CHAT_PROVIDER` picks between them if both keys are set. `.env` is
  loaded automatically (via `python-dotenv`, part of the `llm` extra) if
  present — no need to `export` it manually every session.
- **Design writeup:** the chunking strategy, why TF-IDF over dense
  embeddings/a vector DB, the SQL tool's two-layer safety design (an
  allowlist plus a genuinely read-only SQLite connection — not just the
  allowlist), and the explicit context-window budgets are all documented
  in [DECISIONS.md](DECISIONS.md). An in-app "What the model actually
  saw" panel shows the assembled prompt, retrieved chunks, and an
  approximate token count for every turn.

### Deploying to Streamlit Community Cloud

`requirements.txt` (pip-installable — includes `anthropic`/`openai`/
`python-dotenv` so the deployed app can run Ask the Data too, not just
tabs 1-5) is committed for this; deploying is a one-time manual step
since it requires connecting your own GitHub account:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. "Create app" → pick this repo, branch `main`, main file path `dashboard/app.py`.
   Open **Advanced settings** and confirm/set the Python version to **3.12**
   — this repo requires it (`pyproject.toml`'s `requires-python`), and
   `.python-version` is only a suggested default, not guaranteed to be
   picked up automatically. Skipping this can fail the install outright.
3. In the same Advanced settings screen (or later, in the deployed app's
   Settings → **Secrets**), add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Streamlit Cloud's Secrets panel populates `st.secrets`, not `os.environ`
   — unlike a local `.env` file. `dashboard/app.py` bridges the two
   automatically (`_bridge_streamlit_secrets_to_env()`), so no other
   config is needed, but the secret has to actually be set here — an
   env var exported locally on your own machine has no effect on the
   deployed app.
4. Deploy. First boot takes ~15-30s (installs dependencies, then runs the
   self-bootstrap check — which only rebuilds `hr_analytics.db` from the
   already-committed CSVs, not a full model refit, so it's fast).
5. Update the live demo link above once you have the app's URL.

## Automated Excel executive report

`src/hr_analytics/excel_report.py` generates a formatted, 6-sheet Excel
workbook (`reports/HR_Executive_Report.xlsx`) from the same pipeline
outputs the dashboard reads — the kind of stakeholder-ready deliverable
that gets emailed around, not just viewed live:

- **Executive Summary** — KPI scorecard, including an estimated total
  turnover cost. **That figure is an illustrative estimate** (income x
  12 x a job-level-scaled industry rule-of-thumb multiplier), not
  observed cost data — labelled as such on the sheet itself. See
  DECISIONS.md for the exact methodology.
- **Attrition by Dept & Role**, **Flight Risk Watchlist**, **Survival
  Model Hazard Ratios** — direct reuse of existing pipeline outputs,
  with conditional color-scale formatting.
- **Department x Tenure Pivot** — a `pandas.pivot_table()` cross-tab
  written as a static formatted table. This is *not* a genuine
  interactive native Excel PivotTable — neither `openpyxl` nor
  `xlsxwriter` can reliably create those from scratch. See DECISIONS.md
  for why that's the honest scope.
- **Hiring Pipeline (Synthetic)** — clearly labelled synthetic, same as
  everywhere else this data appears.

The output is byte-reproducible (verified, not assumed — see
DECISIONS.md for two non-obvious `openpyxl` determinism gotchas found
and fixed while building this), so it's committed and covered by CI's
deterministic-diff check like the SQL results are. Also downloadable
directly from the dashboard's Overview tab.

```bash
uv run python -m src.hr_analytics.excel_report   # writes reports/HR_Executive_Report.xlsx
```

## Testing

`tests/` (pytest) covers the data-generation invariants (date ordering,
reproducibility under a fixed seed), the survival model (duration/event
validity, feature-column correctness), the SQLite loader (atomic-write
behavior under a simulated mid-load failure), every SQL query (runs
cleanly + a regression test for the month-spine fix in query 10), and
the chat agent — provider abstraction, SQL tool safety (including a real
injection-payload test), RAG chunking/retrieval, context-window bounding,
and the full multi-turn tool-calling loop via a scripted `FakeProvider` —
and the Excel report generator (turnover-cost calculation, reloaded cell
values, conditional formatting actually attached, and byte-reproducibility
across repeated regenerations). None of this requires an API key or
network access — that's a deliberate
design constraint of the chat agent's test suite, not just this repo's
general testing philosophy. CI (`.github/workflows/ci.yml`) runs the
suite on every push/PR, then rebuilds the full pipeline end-to-end and
fails the build if regenerating it produces any diff against what's
committed — the same class of staleness bug caught twice during
development (see DECISIONS.md).

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
uv run python -m src.hr_analytics.excel_report       # generate reports/HR_Executive_Report.xlsx
uv run streamlit run dashboard/app.py                # launch the dashboard
```

## Status

- [x] Phase 0 — raw data in place, synthetic hiring pipeline extension generated
- [x] Phase 1 — SQL analysis (SQLite db + 10 business-question queries)
- [x] Phase 2 — survival-model attrition prediction (Cox PH, per-employee risk scores)
- [x] Phase 3 — dashboard (Streamlit, 6 tabs, self-bootstrapping)
- [x] Tests + CI (pytest suite, GitHub Actions pipeline-reproducibility check)
- [x] Ask the Data — RAG + tool-calling chat agent, provider-agnostic, zero-network test suite
- [x] Automated Excel executive report — byte-reproducible, downloadable from the dashboard

## License

[MIT](LICENSE)
