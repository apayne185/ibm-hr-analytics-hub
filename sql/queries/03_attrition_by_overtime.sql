-- Q: Does regularly working overtime correlate with leaving?
SELECT
    over_time,
    COUNT(*)                                            AS headcount,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END)  AS leavers,
    ROUND(100.0 * SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY over_time
ORDER BY attrition_rate_pct DESC;
