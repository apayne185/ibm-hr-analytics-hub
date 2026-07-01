# Survival Model: Attrition Prediction

Model: `src/hr_analytics/survival_model.py`, a Cox Proportional Hazards
model (`lifelines.CoxPHFitter`, `penalizer=0.1`). Full coefficient table
in `data/processed/survival_model_coefficients.csv`; per-employee risk
scores in `data/processed/predicted_attrition_risk.csv`. See
[DECISIONS.md](../DECISIONS.md) for how duration/censoring were framed.

**Concordance index: 0.870 in-sample, 0.860 (5-fold CV, +/- 0.012)
out-of-sample.** The two are close, so the model isn't meaningfully
overfitting — but the CV number is the one to trust as a generalization
estimate; the in-sample figure alone would have been optimistic. A model
that correctly orders which of two employees leaves first ~86% of the
time is a strong result for only 16 covariates and no external
labor-market data — largely because `OverTime` and `MonthlyIncome` alone
carry most of the signal already visible in the Phase 1 SQL results.

**Proportional-hazards assumption:** checked with `cph.check_assumptions()`
(full output in `docs/ph_assumptions_check.txt`). Every covariate passes
except `YearsSinceLastPromotion` (p = 0.0014) — its effect isn't constant
over the full tenure range, which is a caveat on that one coefficient
specifically (see below), not on the model as a whole. `OverTime`, the
model's dominant factor, passes cleanly (p = 0.45–0.76).

## What raises the hazard of leaving

| Factor | Hazard ratio | p-value |
|---|---|---|
| OverTime = Yes | **2.16x** | 5.1e-13 |
| Marital status = Single | 1.49x | 0.0016 |
| Travels frequently | 1.46x | 0.0084 |
| Department = Sales | 1.29x | 0.054 (borderline) |
| Each additional prior employer | 1.08x per company | 0.0002 |
| Each mile further from home | 1.02x per mile | 0.014 |

**OverTime dominates everything else in the model**, exactly as it did in
the raw SQL cut (30.5% vs 10.4% attrition) — a hazard ratio of 2.16
means, holding everything else constant, an employee working overtime is
about twice as likely to leave at any given point in their tenure. The
Kaplan-Meier curves make this visible directly:
`docs/figures/km_by_overtime.png` shows the two groups' retention curves
diverging almost immediately and never reconverging (log-rank test,
p = 1.6e-19 — this is not noise).

## What lowers the hazard of leaving

| Factor | Hazard ratio | p-value |
|---|---|---|
| log(Monthly Income) | **0.53x** | 8.9e-11 |
| Each job level up | 0.75x | 1.7e-06 |
| Each year since last promotion | 0.94x | 0.0004 |
| Job satisfaction (per point) | 0.86x | 0.0012 |
| Environment satisfaction (per point) | 0.87x | 0.0026 |
| Stock option level (per level) | 0.86x | 0.025 |

Income and job level are the strongest protective factors — a doubling
of monthly income roughly halves the hazard. Note the `YearsSinceLastPromotion`
direction: *more* time since the last promotion **lowers** hazard here,
which looks backwards until you recall this is a survival model — people
who've been promoted recently are disproportionately people who are
early in a fast-track career (more prone to keep moving, including
elsewhere), while people who've gone a long time without a promotion and
are still here have, by construction, already survived. Read that
coefficient as a proxy for "settled into the role," not as "don't promote
people." This is also the one covariate that fails the proportional-hazards
check above, which is consistent with it being a proxy effect rather than
a stable, constant-over-time hazard — its coefficient describes an average
effect across tenure, not a fixed rate.

`WorkLifeBalance` and both `BusinessTravel_Travel_Rarely` and
`MaritalStatus_Married` were not statistically significant at p < 0.05 —
included for completeness but shouldn't be leaned on individually.

## Per-employee risk scores

`predict_partial_hazard()` gives every employee a relative risk score;
ranking current staff (`Attrition == 'No'`) by this score produces a
prioritized retention watch-list — a modeled version of the ad-hoc SQL
flight-risk query from Phase 1, but now accounting for all 16 factors
jointly instead of four hand-picked flags. The top of that list skews
heavily toward Sales Representatives and Research Scientists, consistent
with the role-level attrition rates already seen in the SQL analysis.

## Caveat

This model explains association, not causation, and duration is measured
in whole years of tenure with heavy ties (many employees share the same
`YearsAtCompany`) — the standard limitation of adapting a single
cross-sectional HR extract into a survival framing. Treat hazard ratios
as relative risk indicators for prioritizing retention conversations, not
as precise probabilities.
