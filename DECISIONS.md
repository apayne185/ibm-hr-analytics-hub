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

## 2026-07-02 — Streamlit for the dashboard, metrics read from a file not hardcoded

**Decision:** build the dashboard (`dashboard/app.py`) with Streamlit +
Plotly rather than a static BI tool (Tableau/Power BI/Looker), and have
it self-bootstrap `data/processed/*` and `docs/figures/*` on first run if
they're missing.

**Why:** a static BI tool would need its own proprietary project file
committed to the repo (not diffable, not runnable from a fresh clone
without that tool installed) and can't be demoed from a `git clone` +
one command. Streamlit is pure Python, reads directly from the same
SQLite db and CSVs the SQL/survival-model phases already produce, and
`uv run streamlit run dashboard/app.py` is the whole setup — matching
the "one command per phase" pattern the rest of this repo already uses.

**Known fix during build:** the first draft hardcoded the concordance
index values (0.870 / 0.860) as literal strings in the Survival Model
tab. Fixed by having `survival_model.py` write
`data/processed/survival_model_metrics.json` and having the dashboard
read it — the same class of staleness risk already fixed once in this
project (see the SQL query 10 gap-month bug and the load_db.py
atomicity fix); hardcoded numbers that mirror a pipeline output are a
recurring trap here and worth checking for on every new artifact.

**How to apply:** any new dashboard metric that comes from a Python
computation (not a live SQL query against the db) should be written to a
small file by the script that computes it and read back by the
dashboard — never typed in by hand a second time.

## 2026-07-02 — Adding a dev dependency silently broke an unrelated script's output

**Decision:** replace `CoxPHFitter.check_assumptions()` in
`check_ph_assumptions()` (`survival_model.py`) with a direct call to
`lifelines.statistics.proportional_hazard_test()`, formatting the result
ourselves with `.to_string()` instead of relying on lifelines' own
printing.

**Why:** adding `jupyter`/`ipykernel` as a dev dependency (for the EDA
notebook) made IPython importable in this project's venv for the first
time. `check_assumptions()` detects IPython at runtime and switches from
plain-text printing to IPython's rich `display()` — which, when its
stdout is captured (as `check_ph_assumptions()` does, to write the file),
produces a useless `<IPython.core.display.HTML object>` placeholder
instead of the actual table. This was caught by re-running the full
pipeline end-to-end while building the CI workflow and diffing the
output against what was committed — `docs/ph_assumptions_check.txt` had
silently gone blank of real data.

**Why fixed this way, not by suppressing IPython:** monkeypatching or
disabling IPython detection would be a bandaid tied to lifelines'
internal implementation and could break again on a lifelines upgrade.
Calling `proportional_hazard_test()` directly and formatting the
DataFrame ourselves has zero dependency on what's importable in the
environment — the same statistical test, a strictly more robust caller.

**How to apply:** this is a general lesson for this repo, not just this
function — adding *any* new dependency (even a dev-only one) can change
the runtime behavior of existing code through packages that conditionally
detect what else is installed (rich display, plotting backends, optional
accelerators). After adding a dependency, re-run the full pipeline and
diff tracked output files, not just the code you meant to touch.

## 2026-07-02 — CI failed on GitHub despite passing locally: cross-platform float noise

**Decision:** round `cph.summary` and `predicted_hazard_score` to 6
decimal places before writing to CSV (`survival_model.py`), and round
the proportional-hazards test summary before formatting it to text,
instead of writing full float64 precision.

**Why:** the GitHub Actions CI run (added in the previous commit) failed
its pipeline-reproducibility check even though the exact same pipeline
passed locally multiple times. `CoxPHFitter`'s Newton-Raphson optimizer
and the residual-based proportional-hazards test can converge to
slightly different values in the low-order digits depending on the
platform's BLAS/LAPACK backend and CPU vector instructions — a
well-known limitation of bitwise floating-point reproducibility across
environments, not a bug in this project's logic. Writing full float64
precision to a tracked CSV turned that harmless numerical noise into a
spurious diff every time the pipeline ran on a different machine than
the one that last committed it.

**Also fixed in the same pass:** the CI workflow was checking
`docs/figures/*.png` for byte-for-byte equality, which is not a
meaningful reproducibility signal — matplotlib's PNG output isn't
guaranteed identical across platforms (font rendering/anti-aliasing,
embedded metadata) even with identical input data. Changed that check to
verify the PNGs regenerate and are a plausible size, not byte-equal.

**How to apply:** for any future numeric output written to a tracked
file, ask whether full float precision is actually meaningful (it rarely
is past 6 decimals for a hazard ratio or a percentage) — round it before
writing. For any binary/rendered output (images, plots), don't add it to
a strict byte-diff CI check; check that it exists and is a sane size
instead. Discovered by actually checking the GitHub Actions run result
via the API rather than assuming "passed locally" meant "passed in CI."

## 2026-07-02 — Rounding wasn't enough: stop requiring exact-match CI for model-fitted outputs

**Decision:** correcting the previous entry — rounding to 6 decimals did
*not* fix the CI failure. The GitHub Actions runner's re-fit coefficients
differed from the locally-committed ones by more than 6 decimal places,
confirmed by diffing the rounding-only commit against itself (same
values, just reformatted) before the CI run still failed on the next
commit. Removed `data/processed/survival_model_coefficients.csv`,
`predicted_attrition_risk.csv`, `survival_model_metrics.json`, and
`docs/ph_assumptions_check.txt` from CI's exact-diff check entirely.
The check now only covers `synthetic_hiring_pipeline.csv`,
`hr_employee_attrition_enriched.csv` (pure seeded RNG + arithmetic), and
`sql/results/` (pure SQL against byte-identical data) — genuinely
bit-reproducible outputs.

**Why:** `CoxPHFitter`'s Newton-Raphson optimizer, run against several
highly correlated covariates (`JobLevel`, `MonthlyIncome`,
`TotalWorkingYears`, `YearsAtCompany` — see the correlation heatmap in
`notebooks/01_exploratory_data_analysis.ipynb`), can converge to
meaningfully different coefficient values across BLAS/LAPACK backends,
not just different last-digit noise. This is a known, accepted
limitation of iterative numerical optimization — forcing byte-identical
output across arbitrary machines is not achievable without pinning the
exact BLAS implementation (impractical for a portfolio project), and
rounding only helps with noise below the rounding precision, not
genuine divergence above it.

**How correctness is actually enforced instead:** `tests/test_survival_model.py`
now has `test_overtime_is_the_dominant_risk_factor`, asserting the
headline finding (OverTime is a significant, large hazard factor) holds
within a tolerance band on every fit, rather than pinning an exact
coefficient. The existing invariant tests (positive duration, valid
event coding, index-aligned predictions) cover the rest. The committed
CSVs remain the last-known-good analysis artifacts referenced by the
findings docs — accurate as of when they were generated, not
guaranteed bit-identical to a future re-run.

**How to apply:** don't add new tracked files derived from an iterative
model fit to a strict CI diff check. Test model outputs by asserting
properties/tolerance bounds, not exact values.
