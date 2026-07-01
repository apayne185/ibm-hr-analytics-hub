# Decisions Log

Running record of non-obvious modelling and design choices, in the order
they were made. Anything in `data/processed/` that isn't a straight
transformation of the raw Kaggle file should be explained here.

## 2026-07-01 — Synthetic hiring pipeline extension

**Decision:** Extend the IBM HR Analytics Employee Attrition dataset
(`data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`, 1,470 employees) with a
simulated requisition-to-hire pipeline, since the source dataset only has
`YearsAtCompany` and no recruiting/time-to-hire fields.

**Why:** the role this project is built for explicitly calls out
time-to-hire as a metric of interest, and no public version of this
dataset includes it. Rather than fabricate a disconnected dataset, the
pipeline dates are derived from each employee's actual `YearsAtCompany`,
so the synthetic fields stay internally consistent with the real data.

**How it works** (`src/hr_analytics/synthetic_hiring.py`):
1. Fix a snapshot/"today" date of **2024-01-15** for the whole dataset.
2. For each employee, back out an approximate `start_date` from
   `snapshot_date - YearsAtCompany` years, minus a random 0-364 day jitter
   (subtracted, never added, so no `start_date` can fall after the
   snapshot date) — this avoids every hire anniversary landing on the same
   calendar day.
3. Simulate `time_to_fill_days` (requisition opened -> offer accepted)
   from a lognormal distribution, right-skewed like real recruiting data,
   with a median that scales with `JobLevel` (junior roles fill faster)
   and a department multiplier (R&D roles assumed to run more interview
   loops than Sales or HR).
4. Simulate `offer_to_start_lag_days` (offer accepted -> first day) as a
   uniform draw that grows with `JobLevel`, approximating longer notice
   periods for more senior hires.
5. `requisition_open_date` and `offer_accepted_date` are derived by
   walking backwards from `start_date` through those two lags.

**Output:**
- `data/processed/synthetic_hiring_pipeline.csv` — one row per
  `EmployeeNumber` with the four pipeline dates plus
  `time_to_fill_days`, `offer_to_start_lag_days`, `time_to_hire_days`.
- `data/processed/hr_employee_attrition_enriched.csv` — the raw dataset
  left-joined with the pipeline table on `EmployeeNumber`.

**Labelling:** every column and file this process touches is documented
here and called out as synthetic in the README. Nothing in
`data/raw/` is modified. Reproducible via `numpy.random.default_rng(seed=42)`
— re-running `synthetic_hiring.py` regenerates identical output.

**Known limitation:** this simulates *plausible* recruiting timelines
consistent with tenure, not real historical ATS data. It's useful for
building and demonstrating time-to-hire analysis (SQL queries, dashboard
panels, trend charts) but should never be presented as observed fact
about IBM's actual hiring process.

## 2026-07-01 — SQLite for the SQL analysis layer

**Decision:** load `employees` and `hiring_pipeline` into a local SQLite
database (`data/processed/hr_analytics.db`, gitignored, rebuilt via
`load_db.py`) rather than standing up Postgres/DuckDB.

**Why:** zero setup/infra for anyone cloning the repo, full standard SQL
(CTEs, window functions, joins) so the queries in `sql/queries/`
demonstrate the same skills a warehouse would require, and the two-table
split (HRIS export vs. ATS export, joined on `employee_number`) mirrors
how this data would actually be structured in a real analytics stack.

**How to apply:** if this ever needs to run against a real warehouse, the
`sql/schema.sql` and `sql/queries/*.sql` are close to portable — only
SQLite-specific functions (`strftime`, `NTILE`/`PERCENT_RANK` window
syntax) would need review against the target engine's dialect.

## 2026-07-01 — Survival framing: duration and censoring

**Decision:** model attrition with a Cox Proportional Hazards model
(`src/hr_analytics/survival_model.py`) using `YearsAtCompany` as the
duration and `Attrition == 'Yes'` as the event indicator, with employees
still employed treated as right-censored at their current tenure.

**Why:** the dataset has no actual hire/termination timestamps for the
*real* HR fields (only the synthetic hiring-pipeline dates, which are a
separate concern — see the synthetic hiring pipeline entry above).
`YearsAtCompany` + `Attrition` is the standard, well-documented way this
specific Kaggle dataset gets adapted for survival analysis: it's real
data (not synthetic), and right-censoring current employees at their
tenure-so-far is exactly what a survival model is designed to handle,
unlike a plain classifier which has no way to express "hasn't happened
yet, might still happen."

**Known adjustment:** `CoxPHFitter` requires strictly positive durations.
44 employees have `YearsAtCompany == 0`. Rather than drop them, their
duration is floored at 0.5 years (`duration_years = YearsAtCompany.clip(lower=0.5)`).
This is a modeling necessity, not a data change — `data/raw/` and
`data/processed/hr_employee_attrition_enriched.csv` are untouched; the
floor only applies inside the model-fitting frame.

**How to apply:** treat the model's hazard ratios as relative risk
indicators, not causal claims or precise probabilities — see
`docs/survival_model_findings.md` for the full caveat and results.

## 2026-07-02 — Validate the model, don't just report its in-sample fit

**Decision:** after an initial pass reported only the in-sample
concordance index, added two checks before trusting those numbers: 5-fold
cross-validated concordance (`lifelines.utils.k_fold_cross_validation`,
`seed=42`) and a formal proportional-hazards test
(`cph.check_assumptions()`, output written to `docs/ph_assumptions_check.txt`).

**Why:** an in-sample concordance index is fit and evaluated on the same
rows, so it's an optimistic estimate of how well the model generalizes.
Separately, "Cox Proportional Hazards" is a name with a testable
assumption baked in (hazard ratios constant over time) — reporting hazard
ratios without checking that assumption holds is asserting something
that was never verified.

**Result:** cross-validated concordance (0.860) came in close to
in-sample (0.870) — the model isn't overfitting meaningfully. The PH
check passed for every covariate except `YearsSinceLastPromotion`
(p = 0.0014), which is noted as a caveat on that specific coefficient in
`docs/survival_model_findings.md` rather than treated as invalidating the
whole model.

**How to apply:** any future change to the covariate set should re-run
both checks (`uv run python -m src.hr_analytics.survival_model`) before
updating the findings doc's numbers — don't hand-edit the reported
concordance index or assume the PH assumption still holds.
