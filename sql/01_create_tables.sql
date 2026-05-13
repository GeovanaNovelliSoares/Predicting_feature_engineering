CREATE TABLE IF NOT EXISTS credit_transactions (
    id                        INTEGER PRIMARY KEY,
    customer_id               INTEGER NOT NULL,
    credit_limit              REAL,
    age                       INTEGER,
    num_late_30_59_days       INTEGER,
    debt_ratio                REAL,
    monthly_income            REAL,
    num_open_credit_lines     INTEGER,
    num_times_90_days_late    INTEGER,
    num_real_estate_loans     INTEGER,
    num_late_60_89_days       INTEGER,
    num_dependents            INTEGER,
    serious_delinquency_2yrs  INTEGER   -- target: 1 = defaulted, 0 = paid on time
);
