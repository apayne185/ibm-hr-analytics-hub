# SQL Analysis: Findings

Synthesis of `sql/queries/01`-`10` run against `data/processed/hr_analytics.db`.
Full output for each query is in `sql/results/`. Queries `08`-`10` run
against the synthetic hiring pipeline (see [DECISIONS.md](../DECISIONS.md))
and should be read as illustrative, not observed fact.

## Attrition drivers (real data)

**Overtime is the single strongest signal.** Employees who regularly work
overtime leave at **30.5%**, nearly 3x the rate of those who don't
(**10.4%**) — `03_attrition_by_overtime.sql`. Company-wide attrition sits
at **16.1%** (237 of 1,470) — `01_overall_attrition_rate.sql`.

**Attrition is heavily front-loaded, and stays high through year two.**
Employees in their first year leave at **36.4%**; year two is still
**28.9%** — nearly as high, not a quick drop-off; only by year 3-4 does
it fall to **16.4%**, then keep falling to **10.4%** past year 10 —
`05_tenure_buckets_vs_attrition.sql`. Whatever is driving attrition, it's
concentrated in the first *two* years, not just the first one — retention
efforts should cover the full 24-month window, not taper off after year one.

**Sales Representatives are the highest-risk role by far**: **39.8%**
attrition on 83 headcount, well above every other role with meaningful
headcount. Laboratory Technicians (23.9% on 259 people) and the HR
generalist role (23.1% on 52 people) are the next-worst —
`02_attrition_by_department_and_role.sql`. Research Director and HR
Manager are near-zero, but both are small populations.

**Satisfaction scores are directionally consistent but not dramatic.**
Leavers self-report slightly lower job satisfaction (2.47 vs 2.78),
environment satisfaction (2.46 vs 2.77), and work-life balance (2.66 vs
2.78) than stayers — `06_satisfaction_scores_vs_attrition.sql`. The gap
exists on every dimension, but it's a few tenths on a 4-point scale, not
a bright line — satisfaction alone is a weak individual predictor,
consistent with why the project moves to a proper survival model next
rather than relying on these univariate cuts.

**Pay-relative-to-peers matters more in some roles than others.**
Laboratory Technicians and Research Scientists show a clean gradient —
bottom income quartile within the role attrites at roughly double the top
quartile (e.g. Lab Techs: 32.3% in Q1 vs 14.1% in Q4). Sales
Representatives are volatile at every quartile (28-62%), suggesting
something role-specific beyond pay is going on there —
`04_income_percentile_by_job_role.sql`.

**A simple composite flight-risk score surfaces a concrete watch-list.**
Ranking current employees (`attrition = 'No'`) by overtime + low
satisfaction/engagement flags + below-median pay for their role produces
a ranked shortlist of 20 people worth a retention conversation, not just
an aggregate rate — `07_current_employee_flight_risk_ranking.sql`. This
is a transparent heuristic, not a model; Phase 2 replaces it with a
proper survival-model risk score.

## Hiring pipeline (synthetic data)

**Time-to-fill scales with seniority, as designed.** Job level 5 hires in
R&D take an average of **102 days** end-to-end vs. **41 days** for level
1 — `08_time_to_hire_by_department_and_level.sql`. This reflects the
simulation's assumptions (see DECISIONS.md) more than a discovered
insight, but it validates the generator is behaving as intended and gives
the dashboard something realistic to visualize.

**No evidence that slow-to-fill roles attrite faster.** If anything, the
fastest-filled roles (<25 days) show the *highest* attrition (22.1%) and
the 40-59 day bucket the lowest (9.8%) — `09_attrition_rate_by_time_to_hire_bucket.sql`.
Worth noting precisely because it's a **synthetic** dataset: time-to-fill
and attrition were generated independently (time-to-fill from job
level/department, hire date from tenure), so this near-absence of a
pattern is the expected null result, not a real-world finding.

## Caveat

Queries 8-10 exist to demonstrate SQL against date/pipeline data and to
feed the dashboard — draw hiring conclusions from them for portfolio
purposes only, never as claims about IBM's actual recruiting operation.
