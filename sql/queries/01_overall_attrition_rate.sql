-- Q: What's the company-wide attrition rate?
SELECT
    COUNT(*)                                            AS headcount,
    SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END)  AS leavers,
    ROUND(100.0 * SUM(CASE WHEN attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_rate_pct
FROM employees;
