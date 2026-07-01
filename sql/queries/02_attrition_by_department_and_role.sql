-- Q: Which departments/roles have the worst attrition, and are they big
--    enough to matter (avoid over-indexing on a role with 3 people)?
SELECT
    department,
    job_role,
    COUNT(*)                                            AS headcount,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END)  AS leavers,
    ROUND(100.0 * SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY department, job_role
HAVING COUNT(*) >= 10
ORDER BY attrition_rate_pct DESC;
