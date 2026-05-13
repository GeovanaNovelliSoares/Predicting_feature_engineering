-- 1. Overall default rate
SELECT
    serious_delinquency_2yrs                                            AS defaulted,
    COUNT(*)                                                            AS total,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)                 AS pct
FROM credit_transactions
GROUP BY serious_delinquency_2yrs;


-- 2. Default rate by age band
WITH age_bands AS (
    SELECT *,
        CASE
            WHEN age < 30              THEN 'Under 30'
            WHEN age BETWEEN 30 AND 44 THEN '30-44'
            WHEN age BETWEEN 45 AND 59 THEN '45-59'
            ELSE '60+'
        END AS age_band
    FROM credit_transactions
)
SELECT
    age_band,
    COUNT(*)                                           AS total_customers,
    SUM(serious_delinquency_2yrs)                      AS defaults,
    ROUND(AVG(serious_delinquency_2yrs) * 100, 2)     AS default_rate_pct,
    ROUND(AVG(debt_ratio), 4)                          AS avg_debt_ratio
FROM age_bands
GROUP BY age_band
ORDER BY default_rate_pct DESC;


-- 3. Late payment frequency vs default rate
WITH late_summary AS (
    SELECT
        customer_id,
        num_late_30_59_days + num_late_60_89_days + num_times_90_days_late AS total_late_events,
        serious_delinquency_2yrs
    FROM credit_transactions
)
SELECT
    total_late_events,
    COUNT(*)                                       AS customers,
    SUM(serious_delinquency_2yrs)                  AS defaults,
    ROUND(AVG(serious_delinquency_2yrs) * 100, 2) AS default_rate_pct
FROM late_summary
GROUP BY total_late_events
ORDER BY total_late_events;


-- 4. High-risk segment: above-median debt ratio + recent late payments
WITH medians AS (
    SELECT AVG(debt_ratio) AS med_debt
    FROM credit_transactions
),
segmented AS (
    SELECT
        t.*,
        CASE
            WHEN t.debt_ratio > m.med_debt AND t.num_late_30_59_days > 0
            THEN 'High Risk'
            ELSE 'Standard'
        END AS risk_segment
    FROM credit_transactions t
    CROSS JOIN medians m
)
SELECT
    risk_segment,
    COUNT(*)                                       AS customers,
    ROUND(AVG(serious_delinquency_2yrs) * 100, 2) AS default_rate_pct,
    ROUND(AVG(monthly_income), 2)                 AS avg_monthly_income,
    ROUND(AVG(debt_ratio), 4)                     AS avg_debt_ratio
FROM segmented
GROUP BY risk_segment;


-- 5. Customers ranked by risk within each income band (window function)
WITH income_bands AS (
    SELECT *,
        CASE
            WHEN monthly_income IS NULL  THEN 'Unknown'
            WHEN monthly_income < 2000   THEN 'Low'
            WHEN monthly_income < 6000   THEN 'Mid'
            ELSE 'High'
        END AS income_band
    FROM credit_transactions
),
ranked AS (
    SELECT
        customer_id,
        income_band,
        debt_ratio,
        num_times_90_days_late,
        serious_delinquency_2yrs,
        ROW_NUMBER() OVER (
            PARTITION BY income_band
            ORDER BY debt_ratio DESC, num_times_90_days_late DESC
        ) AS risk_rank_within_band
    FROM income_bands
)
SELECT *
FROM ranked
WHERE risk_rank_within_band <= 10
ORDER BY income_band, risk_rank_within_band;
