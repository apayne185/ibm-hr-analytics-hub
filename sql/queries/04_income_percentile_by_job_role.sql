-- Q: For each employee who left, where did their income sit relative to
--    peers in the same job role? Flags whether pay was likely a factor
--    (e.g. leavers clustered in the bottom income quartile of their role).
WITH ranked AS (
    SELECT
        employee_number,
        job_role,
        monthly_income,
        attrition,
        NTILE(4) OVER (PARTITION BY job_role ORDER BY monthly_income) AS income_quartile_in_role
    FROM employees
)
SELECT
    job_role,
    income_quartile_in_role,
    COUNT(*)                                            AS headcount,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END)  AS leavers,
    ROUND(100.0 * SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM ranked
GROUP BY job_role, income_quartile_in_role
ORDER BY job_role, income_quartile_in_role;
