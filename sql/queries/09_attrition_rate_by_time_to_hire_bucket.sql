-- Q: Do slower hires (long time-to-fill) end up leaving sooner, e.g.
--    because roles that are hard to fill also see rushed/compromise hires?
--    (synthetic hiring_pipeline data, see DECISIONS.md)
SELECT
    CASE
        WHEN h.time_to_fill_days < 25 THEN '0. <25 days'
        WHEN h.time_to_fill_days < 40 THEN '1. 25-39 days'
        WHEN h.time_to_fill_days < 60 THEN '2. 40-59 days'
        ELSE '3. 60+ days'
    END AS time_to_fill_bucket,
    COUNT(*)                                            AS headcount,
    SUM(CASE WHEN e.attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
    ROUND(100.0 * SUM(CASE WHEN e.attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees e
JOIN hiring_pipeline h ON h.employee_number = e.employee_number
GROUP BY time_to_fill_bucket
ORDER BY time_to_fill_bucket;
