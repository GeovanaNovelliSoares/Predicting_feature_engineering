CREATE VIEW IF NOT EXISTS v_model_features AS
WITH base AS (
    SELECT *,

        -- Weighted late score: recency-aware severity index
        (num_late_30_59_days  * 1 +
         num_late_60_89_days  * 2 +
         num_times_90_days_late * 3)              AS weighted_late_score,

        -- Credit utilisation proxy
        CASE
            WHEN credit_limit > 0
            THEN ROUND(debt_ratio / credit_limit, 6)
            ELSE NULL
        END                                       AS utilisation_ratio,

        -- Income band (categorical bucketing)
        CASE
            WHEN monthly_income IS NULL THEN 0   -- unknown
            WHEN monthly_income < 2000  THEN 1   -- low
            WHEN monthly_income < 6000  THEN 2   -- mid
            ELSE 3                               -- high
        END                                       AS income_band,

        -- Dependents-to-income ratio (financial stress proxy)
        CASE
            WHEN monthly_income > 0
            THEN ROUND(num_dependents * 1.0 / monthly_income, 6)
            ELSE NULL
        END                                       AS dependents_income_ratio

    FROM credit_transactions
)
SELECT
    customer_id,
    credit_limit,
    age,
    debt_ratio,
    monthly_income,
    num_open_credit_lines,
    num_real_estate_loans,
    num_dependents,
    weighted_late_score,
    utilisation_ratio,
    income_band,
    dependents_income_ratio,
    serious_delinquency_2yrs  AS target
FROM base;
