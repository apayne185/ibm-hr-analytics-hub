-- Q: What does hiring volume look like over time, and how does it trend
--    month over month? (synthetic hiring_pipeline data, see DECISIONS.md)
WITH monthly AS (
    SELECT
        strftime('%Y-%m', start_date) AS hire_month,
        COUNT(*)                      AS hires
    FROM hiring_pipeline
    GROUP BY hire_month
)
SELECT
    hire_month,
    hires,
    SUM(hires) OVER (ORDER BY hire_month) AS cumulative_hires,
    hires - LAG(hires) OVER (ORDER BY hire_month) AS change_vs_prior_month
FROM monthly
ORDER BY hire_month;
