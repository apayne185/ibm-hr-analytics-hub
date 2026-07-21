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

## 2026-07-21 — Provider-agnostic LLM abstraction for the chat agent

**Decision:** built the "Ask the Data" chat tab against a thin `Protocol`
(`LLMProvider.complete()`, `src/hr_analytics/llm_providers.py`) using
vendor-neutral `Message`/`ToolCall`/`ToolResult`/`ToolSpec` dataclasses,
instead of coding directly against Anthropic's or OpenAI's SDK shape.
`AnthropicProvider`/`OpenAIProvider` each translate to/from their own
wire format internally and lazy-import their SDK inside `__init__`, not
at module top level. A `FakeProvider` scripts a sequence of responses
for deterministic, zero-network multi-turn tool-calling tests.

**Why:** no LLM API credentials were available at build time, and the
explicit goal was a feature that's fully testable without any network
call or API key — matching this repo's established test philosophy (see
the CI-reproducibility entries above: don't assert on anything
non-deterministic). Coding against one vendor's SDK would have made that
impossible without a live key. The lazy-import pattern is what lets
`anthropic`/`openai` be an opt-in extra (`uv sync --extra llm`) rather
than a hard dependency — CI never installs either package, so CI itself
enforces that the import discipline doesn't regress: an eager top-level
`import anthropic` would break the entire test suite immediately, not
just a targeted test.

**How to apply:** wiring up a real provider live (and choosing Anthropic
vs. OpenAI as the first one) is a separate, later step once API
credentials exist — this abstraction supports both equally, so that
choice is deferred, not blocked on anything built here.

## 2026-07-21 — TF-IDF over dense embeddings or a vector DB for RAG retrieval

**Decision:** `src/hr_analytics/rag.py` retrieves from this project's
own docs using `scikit-learn`'s `TfidfVectorizer` + in-memory cosine
similarity — not dense/neural embeddings, not FAISS/Chroma/pgvector.

**Why:** the corpus is a handful of markdown files (~30-60 chunks after
header-based chunking) — small enough that a sparse matrix held in a
Python object genuinely *is* the index at this scale; a vector database
here would be the kind of over-engineering this project has deliberately
avoided elsewhere (see the SQLite-not-Postgres and
lognormal-not-a-full-simulator-package decisions above). More
specifically: a dense-embedding model (e.g. `sentence-transformers`)
needs to download model weights the first time it's used, which is a
disguised network dependency that would silently break the "zero network
calls in tests" requirement the whole chat feature was built around.
TF-IDF needs no download, is fully deterministic, and the docs are
technical/keyword-heavy (`"OverTime"`, `"concordance index"`, `"Sales
Representative"`) — exactly the profile where lexical matching works
well and dense embeddings' advantage (capturing semantic similarity
between differently-worded but related concepts) matters least.

**How to apply:** if the doc corpus grows substantially (many more files,
or longer documents), reconsider dense local embeddings (accepting the
one-time model download as a setup step, not a test-time dependency) and,
past a few thousand chunks, a real vector index. Neither is justified at
the current scale.

## 2026-07-21 — SQL tool safety: two independent layers, not one

**Decision:** `src/hr_analytics/sql_tool.py` gives the chat agent
database access through two layers that don't depend on each other:
`validate_query()` (a fast-fail allowlist — single `SELECT`/`WITH`
statement only, rejects DDL/DML/PRAGMA/ATTACH, forces `LIMIT 50`) and a
SQLite connection opened via the `mode=ro` URI, which is physically
incapable of writing regardless of what SQL text reaches it.

**Why:** considered two alternatives first. A purely parameterized tool
set (fixed functions like `get_attrition_rate(department)`) is the
safest option but can't answer genuinely open-ended questions or
follow-ups — it either explodes combinatorially trying to cover every
possible question shape, or just can't answer some of them. Unconstrained
free-form SQL generation handed straight to `sqlite3.connect()` is
flexible but hands a network-facing LLM real query power with only a
system-prompt instruction as a guardrail — not a real safety boundary.
The resolution was to offer **both**: `get_attrition_rate` and
`get_flight_risk_watchlist` as parameterized tools for the highest-
frequency questions (using `execute_parameterized()`'s `?` placeholder
binding, not string interpolation, even though the values involved are
low-risk — see the entry below), and `sql_query` as a constrained
free-form fallback for everything else.

The read-only URI connection, not the regex allowlist, is the actual
security boundary — verified by `test_connection_is_physically_read_only`,
which bypasses `validate_query()` entirely and confirms a write still
fails at the `sqlite3.OperationalError` level. The regex allowlist exists
to give the calling LLM a fast, readable error it can self-correct from,
not as the primary defense.

**How to apply:** any new tool that touches the database should go
through one of `sql_tool.py`'s two entry points
(`run_read_only_query()` for untrusted SQL text,
`execute_parameterized()` for trusted SQL text with untrusted bound
values) — never open a fresh unguarded connection.

## 2026-07-21 — Caught during build: parameterized tools still need placeholders, not interpolation

**Decision:** while implementing `chat_agent.py`'s `get_attrition_rate`
and `get_flight_risk_watchlist` tools, initially planned to
string-interpolate the LLM-supplied `department`/`job_role` arguments
directly into a SQL `WHERE` clause. Caught this before shipping it and
added `sql_tool.execute_parameterized()` (binds values via `?`
placeholders) instead, splitting `sql_tool.py`'s connection logic into a
shared internal helper with two named entry points making the safety
contract explicit at each call site.

**Why:** the read-only connection means an interpolated value can't
cause a write, but it can still corrupt the query's `WHERE`-clause logic
— e.g. a `department` argument of `"Sales' OR '1'='1"` would silently
widen the filter to match every row instead of erroring or matching
nothing. Verified this was a real, working issue with
`test_execute_parameterized_does_not_let_a_value_alter_query_logic`,
which passes a classic injection payload as a bound value and asserts it
matches zero rows (treated as a literal string), then confirmed the fix
by checking the same payload would have broken a naive f-string version.

**How to apply:** this is a general lesson, not just for this tool —
when a value that ultimately came from an LLM (or any untrusted source)
ends up in a query, prefer parameter binding over string interpolation
even in a context that already has another safety layer (here, the
read-only connection). Layers should be independent, not load-bearing on
each other.

## 2026-07-22 — Ship the chat feature to the deployed dashboard, bridge Streamlit secrets

**Decision:** now that a real Anthropic key was live-tested and confirmed
working (see the provider-agnostic abstraction entry above, which
explicitly deferred this exact choice), `requirements.txt` now includes
the `llm` extra (`anthropic`, `openai`, `python-dotenv`) via
`uv export --extra llm`, and `dashboard/app.py` gained
`_bridge_streamlit_secrets_to_env()`.

**Why:** Streamlit Community Cloud's Secrets panel populates `st.secrets`,
not `os.environ` — confirmed by testing both the missing-`secrets.toml`
case (raises `StreamlitSecretNotFoundError`) and the
key-present-but-different-key-requested case (raises `KeyError`) directly,
not assumed from documentation. Without the bridge, a correctly-set
Streamlit Cloud secret would have silently done nothing, since
`llm_providers.get_provider()` only ever reads `os.environ` — the tab
would show the same "set an API key" message whether or not a secret was
configured, which is actively misleading (looks like a key problem when
it's actually a wiring problem).

**How it works:** `_bridge_streamlit_secrets_to_env()` copies
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`HR_CHAT_PROVIDER` from `st.secrets`
into `os.environ` if present and not already set, before calling
`get_provider()` — never overwrites an explicitly-set env var. Kept in
`dashboard/app.py`, not `llm_providers.py`: the provider abstraction
should stay Streamlit-agnostic (it's meant to be usable outside a
Streamlit context, e.g. a future CLI), so the Streamlit-specific bridging
lives at the Streamlit-specific integration point.

**Verified, not assumed:** installed `requirements.txt` into a clean venv
via plain `pip install` (matching Streamlit Cloud's own install
mechanism, not `uv`) and confirmed the dashboard, including Ask the
Data, works from it. Wrote a real `.streamlit/secrets.toml` in a scratch
directory and confirmed `_bridge_streamlit_secrets_to_env()` genuinely
copies the value into `os.environ`, not just that it doesn't crash.

**How to apply:** the CI `requirements.txt` drift-check
(`.github/workflows/ci.yml`) was updated to export with `--extra llm` to
match — if that ever needs regenerating by hand, use
`uv export --no-dev --no-hashes --no-emit-project --extra llm --format requirements-txt -o requirements.txt`,
not the shorter command from the earlier entries in this log.

## 2026-07-22 — Automated Excel executive report: scope and methodology

**Decision:** `src/hr_analytics/excel_report.py` generates a 6-sheet
formatted workbook (`reports/HR_Executive_Report.xlsx`) using `openpyxl`,
covering an executive KPI summary, attrition by department/role, a
department-by-tenure cross-tab, the flight-risk watchlist, survival
model hazard ratios, and the synthetic hiring pipeline.

**"Pivot table" scope, stated plainly:** the Department x Tenure sheet is
a `pandas.pivot_table()` result written as a static formatted table, not
a genuine interactive native Excel PivotTable. Neither `openpyxl` nor
`xlsxwriter` can reliably create a real PivotTable (tied to Excel's own
pivot cache) from scratch — both document this as unsupported. This is
the honest, achievable scope, not a shortcut taken silently.

**Turnover cost estimate, methodology:** `TURNOVER_COST_MULTIPLIER_BY_LEVEL`
(`{1: 0.5, 2: 0.75, 3: 1.0, 4: 1.5, 5: 2.0}`) applied to
`MonthlyIncome * 12` per actual leaver, mirroring
`synthetic_hiring.py`'s existing job-level-scaled pattern
(`BASE_MEDIAN_DAYS_BY_LEVEL`). This is an **illustrative industry
rule-of-thumb** (replacement cost commonly cited as 0.5x-2x annual
salary depending on seniority) applied to this dataset's real income
figures — not observed cost data for this (fictional) company. Labelled
explicitly on the Executive Summary sheet itself, not just here, the
same standard this repo already holds the synthetic hiring pipeline to.

**Library choice — `openpyxl`, not `xlsxwriter`:** `xlsxwriter` is
write-only; verifying generated content in tests would need a second
library just to read it back. `openpyxl` handles both, so the test
suite reloads what it wrote and asserts real cell values and that
conditional-formatting rules were actually attached — one dependency,
symmetric read/write.

**Determinism, found the hard way:** the workbook needed to be
byte-reproducible to commit it and add it to CI's existing
deterministic-diff-check (the same mechanism covering `sql/results/`
etc.). Two independent, non-obvious sources of non-determinism, both
found by actually regenerating twice and diffing bytes rather than
assuming a `wb.properties.modified = <fixed value>` assignment would
hold:
1. `openpyxl.writer.excel.save_workbook()` unconditionally overwrites
   `workbook.properties.modified` with `datetime.now()` immediately
   before writing — confirmed by reading the library source
   (`writer/excel.py` line 292), not guessed. No public parameter
   disables this. Worked around by post-processing the already-saved
   zip's `docProps/core.xml` after the fact.
2. That fix's first version corrupted the file: the replacement regex
   used bare `\1`/`\2` backreferences immediately followed by the fixed
   timestamp string, which starts with digits (`"2026-..."`) — Python's
   `re` module parsed `\1` + `"20"` as the octal escape `\120`
   (`chr(80)` = `"P"`) instead of "backreference, then literal text."
   Reproduced the exact corruption in isolation before fixing it with
   `\g<1>`/`\g<2>`, then confirmed the file reloads correctly afterward.
   The zip container's own per-entry DOS timestamps turned out to be a
   *third*, independent source, invisible to a content-only diff (e.g.
   `diff` after `unzip`) since it's zip metadata, not file content —
   caught by comparing raw file sizes and running `cmp` directly, not
   just `diff -rq` on extracted contents.

**How to apply:** if this workbook's structure changes, re-verify
determinism explicitly (`generate_report()` twice, compare bytes) before
assuming the existing fix still covers whatever changed — don't assume
byte-reproducibility carries over automatically to new content or a
library upgrade.
