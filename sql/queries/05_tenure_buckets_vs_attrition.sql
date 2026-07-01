-- Q: Is attrition concentrated in a particular tenure window (e.g. the
--    classic "leaves in year 1-2" pattern)?
SELECT
    CASE
        WHEN years_at_company < 1 THEN '0. <1 year'
        WHEN years_at_company < 3 THEN '1. 1-2 years'
        WHEN years_at_company < 5 THEN '2. 3-4 years'
        WHEN years_at_company < 10 THEN '3. 5-9 years'
        ELSE '4. 10+ years'
    END AS tenure_bucket,
    COUNT(*)                                            AS headcount,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END)  AS leavers,
    ROUND(100.0 * SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY tenure_bucket
ORDER BY tenure_bucket;
