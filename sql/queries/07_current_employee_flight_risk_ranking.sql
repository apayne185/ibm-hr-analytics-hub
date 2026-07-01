-- Q: Among employees who are still here, who looks most like the profile
--    of people who've historically left (frequent overtime, low
--    satisfaction, below-median pay for their role)? A simple, transparent
--    composite score -- not a model -- to prioritize retention conversations.
WITH scored AS (
    SELECT
        employee_number,
        department,
        job_role,
        monthly_income,
        over_time,
        job_satisfaction,
        environment_satisfaction,
        work_life_balance,
        PERCENT_RANK() OVER (PARTITION BY job_role ORDER BY monthly_income) AS income_percentile_in_role,
        (CASE WHEN over_time = 'Yes' THEN 1 ELSE 0 END)
            + (CASE WHEN job_satisfaction <= 2 THEN 1 ELSE 0 END)
            + (CASE WHEN environment_satisfaction <= 2 THEN 1 ELSE 0 END)
            + (CASE WHEN work_life_balance <= 2 THEN 1 ELSE 0 END) AS risk_flags
    FROM employees
    WHERE attrition = 'No'
)
SELECT
    employee_number,
    department,
    job_role,
    monthly_income,
    ROUND(income_percentile_in_role, 2) AS income_percentile_in_role,
    over_time,
    risk_flags,
    RANK() OVER (ORDER BY risk_flags DESC, income_percentile_in_role ASC) AS flight_risk_rank
FROM scored
WHERE risk_flags >= 2
ORDER BY flight_risk_rank
LIMIT 20;
