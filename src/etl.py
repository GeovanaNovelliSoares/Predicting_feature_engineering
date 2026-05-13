import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH        = Path("data/processed/ar_risk.db")
PROCESSED_PATH = Path("data/processed/features.csv")

RNG = np.random.default_rng(seed=42)


def generate_dataset(n: int = 150_000) -> pd.DataFrame:
    """
    Generate a synthetic AR dataset with realistic correlations between
    payment behavior, financial stress indicators, and default probability.
    """

    age = RNG.integers(22, 75, size=n)

    monthly_income = RNG.lognormal(mean=8.5, sigma=0.6, size=n)      # median ~$4,900
    monthly_income[RNG.random(n) < 0.08] = np.nan                     # 8% missing

    credit_limit       = RNG.lognormal(mean=10.0, sigma=0.9, size=n).clip(500, 500_000)
    num_open_credit    = RNG.integers(0, 20, size=n)
    num_real_estate    = RNG.integers(0, 4, size=n)
    num_dependents_raw = RNG.choice([0, 1, 2, 3, 4], size=n, p=[0.38, 0.28, 0.20, 0.09, 0.05])

    base_debt = RNG.beta(2, 5, size=n)                                 # most < 0.4
    stress_mask = RNG.random(n) < 0.15
    base_debt[stress_mask] += RNG.uniform(0.3, 0.8, size=stress_mask.sum())
    debt_ratio = base_debt.clip(0, 2.0)

    late_prob_base = (debt_ratio * 0.4 + RNG.random(n) * 0.2).clip(0, 1)
    num_late_30_59 = RNG.binomial(5, late_prob_base * 0.5)
    num_late_60_89 = RNG.binomial(3, late_prob_base * 0.3)
    num_times_90   = RNG.binomial(2, late_prob_base * 0.2)

    log_odds = (
        -4.5
        + 0.8  * (num_late_30_59 > 0).astype(float)
        + 1.2  * (num_late_60_89 > 0).astype(float)
        + 2.0  * (num_times_90   > 0).astype(float)
        + 1.5  * debt_ratio
        - 0.3  * np.log1p(np.nan_to_num(monthly_income, nan=3000) / 1000)
        + 0.02 * (age < 30).astype(float)
        + RNG.normal(0, 0.5, size=n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    target = RNG.binomial(1, prob_default)

    df = pd.DataFrame({
        "customer_id":              np.arange(1, n + 1),
        "age":                      age,
        "monthly_income":           monthly_income.round(2),
        "credit_limit":             credit_limit.round(2),
        "debt_ratio":               debt_ratio.round(6),
        "num_open_credit_lines":    num_open_credit,
        "num_real_estate_loans":    num_real_estate,
        "num_dependents":           num_dependents_raw,
        "num_late_30_59_days":      num_late_30_59,
        "num_late_60_89_days":      num_late_60_89,
        "num_times_90_days_late":   num_times_90,
        "serious_delinquency_2yrs": target,
    })

    print(f"Generated {n:,} synthetic records | default rate: {target.mean():.2%}")
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median()).round(2)

    for col in ["num_late_30_59_days", "num_late_60_89_days", "num_times_90_days_late"]:
        df[col] = df[col].clip(upper=df[col].quantile(0.99)).astype(int)

    df = df[(df["age"] >= 18) & (df["age"] <= 100)].reset_index(drop=True)
    print(f"After cleaning: {len(df):,} rows")
    return df

def to_sqlite(df: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("credit_transactions", conn, if_exists="replace", index=False)
        sql_view = Path("sql/03_feature_engineering.sql").read_text()
        conn.executescript(sql_view)
        conn.commit()
    print(f"Data written to SQLite → {DB_PATH}")


def export_features() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM v_model_features", conn)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Features exported → {PROCESSED_PATH}  ({len(df):,} rows × {df.shape[1]} cols)")
    return df


def run() -> pd.DataFrame:
    df_raw      = generate_dataset(n=150_000)
    df_clean    = clean(df_raw)
    to_sqlite(df_clean)
    df_features = export_features()
    return df_features


if __name__ == "__main__":
    run()
