-- Schema for the IBM HR Analytics Hub SQLite database.
-- Two tables, kept separate to mirror how this data would actually arrive
-- in a warehouse: an HRIS export (employees) and an ATS export
-- (hiring_pipeline) that get joined on employee_number.

DROP TABLE IF EXISTS employees;
CREATE TABLE employees (
    employee_number             INTEGER PRIMARY KEY,
    age                         INTEGER,
    attrition                   TEXT,   -- 'Yes' / 'No'
    business_travel             TEXT,
    daily_rate                  INTEGER,
    department                  TEXT,
    distance_from_home          INTEGER,
    education                   INTEGER,
    education_field             TEXT,
    environment_satisfaction    INTEGER,
    gender                      TEXT,
    hourly_rate                 INTEGER,
    job_involvement             INTEGER,
    job_level                   INTEGER,
    job_role                    TEXT,
    job_satisfaction            INTEGER,
    marital_status              TEXT,
    monthly_income              INTEGER,
    monthly_rate                INTEGER,
    num_companies_worked        INTEGER,
    over_18                     TEXT,
    over_time                   TEXT,   -- 'Yes' / 'No'
    percent_salary_hike         INTEGER,
    performance_rating          INTEGER,
    relationship_satisfaction   INTEGER,
    standard_hours              INTEGER,
    stock_option_level          INTEGER,
    total_working_years         INTEGER,
    training_times_last_year    INTEGER,
    work_life_balance           INTEGER,
    years_at_company            INTEGER,
    years_in_current_role       INTEGER,
    years_since_last_promotion  INTEGER,
    years_with_curr_manager     INTEGER
);

DROP TABLE IF EXISTS hiring_pipeline;
CREATE TABLE hiring_pipeline (
    employee_number          INTEGER PRIMARY KEY REFERENCES employees(employee_number),
    requisition_open_date    TEXT,   -- synthetic, see DECISIONS.md
    offer_accepted_date      TEXT,   -- synthetic
    start_date               TEXT,   -- synthetic
    time_to_fill_days        INTEGER,
    offer_to_start_lag_days  INTEGER,
    time_to_hire_days        INTEGER
);
