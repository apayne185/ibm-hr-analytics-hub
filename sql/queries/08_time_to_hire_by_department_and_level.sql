-- Q: Where in the org does it take longest to fill a role? (synthetic
--    hiring_pipeline data, see DECISIONS.md)
SELECT
    e.department,
    e.job_level,
    COUNT(*)                                       AS hires,
    ROUND(AVG(h.time_to_fill_days), 1)             AS avg_time_to_fill_days,
    ROUND(AVG(h.time_to_hire_days), 1)             AS avg_time_to_hire_days
FROM employees e
JOIN hiring_pipeline h ON h.employee_number = e.employee_number
GROUP BY e.department, e.job_level
ORDER BY e.department, e.job_level;
