-- Q: What does hiring volume look like over time, and how does it trend
--    month over month? (synthetic hiring_pipeline data, see DECISIONS.md)
--
-- hiring_pipeline's start_date is sparse in the early years (long stretches
-- with zero hires), so a plain LAG() over the distinct hired months would
-- silently treat non-adjacent months as adjacent (e.g. comparing a hire
-- month to one from years earlier with no zero-hire months in between).
-- month_spine fills every calendar month in range so gaps count as 0, not
-- as if they don't exist.
WITH RECURSIVE month_spine AS (
    SELECT strftime('%Y-%m', (SELECT MIN(start_date) FROM hiring_pipeline)) AS hire_month
    UNION ALL
    SELECT strftime('%Y-%m', date(hire_month || '-01', '+1 month'))
    FROM month_spine
    WHERE hire_month < strftime('%Y-%m', (SELECT MAX(start_date) FROM hiring_pipeline))
),
monthly AS (
    SELECT
        strftime('%Y-%m', start_date) AS hire_month,
        COUNT(*)                      AS hires
    FROM hiring_pipeline
    GROUP BY hire_month
)
SELECT
    s.hire_month,
    COALESCE(m.hires, 0) AS hires,
    SUM(COALESCE(m.hires, 0)) OVER (ORDER BY s.hire_month) AS cumulative_hires,
    COALESCE(m.hires, 0) - LAG(COALESCE(m.hires, 0)) OVER (ORDER BY s.hire_month) AS change_vs_prior_month
FROM month_spine s
LEFT JOIN monthly m ON m.hire_month = s.hire_month
ORDER BY s.hire_month;
