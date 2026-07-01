-- Q: Do leavers report meaningfully lower satisfaction/engagement scores
--    than stayers, across every dimension we survey?
SELECT
    attrition,
    ROUND(AVG(job_satisfaction), 2)           AS avg_job_satisfaction,
    ROUND(AVG(environment_satisfaction), 2)   AS avg_environment_satisfaction,
    ROUND(AVG(relationship_satisfaction), 2)  AS avg_relationship_satisfaction,
    ROUND(AVG(work_life_balance), 2)          AS avg_work_life_balance,
    ROUND(AVG(job_involvement), 2)            AS avg_job_involvement,
    COUNT(*)                                  AS headcount
FROM employees
GROUP BY attrition;
