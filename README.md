# AR Risk Scoring Model Finance Analytics

Predicting the probability of payment delinquency using machine learning, SQL-based feature engineering, and interactive dashboards.

> **Context:** This project simulates a real Accounts Receivable risk scoring use case in a Finance environment. The dataset is **fully synthetic and self-contained** — generated programmatically with realistic statistical distributions (150,000 records). No download required: simply run `python src/etl.py` and the data is created automatically. The goal is to identify high-risk customers before a delinquency event occurs, enabling proactive action from the credit and collections team.

---

## Problem Statement

Late payments and defaults in accounts receivable directly impact cash flow forecasting and working capital management. Traditional rule-based approaches (e.g., days overdue thresholds) are reactive. This project builds a **predictive scoring model** that assigns each customer a risk probability, enabling:

- Early identification of at-risk accounts
- Prioritization of collection efforts
- Scenario analysis for Finance planning

---

## Project Structure

```
ar-risk-scoring/
│
├── data/
│   ├── raw/               # Original dataset (not versioned)
│   └── processed/         # Cleaned and feature-engineered dataset
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_feature_eng.ipynb
│   └── 03_modeling.ipynb   # Baseline → Random Forest → SHAP
│
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_eda_queries.sql       # CTEs, window functions, complex joins
│   └── 03_feature_engineering.sql
│
├── src/
│   ├── etl.py             # Data loading and transformation pipeline
│   ├── features.py        # Feature engineering functions
│   └── model.py           # Training, evaluation, and export
│
├── reports/
│   └── executive_summary.pdf
│
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Database | SQLite + DBeaver | SQL Server equivalent — CTEs, window functions, complex joins |
| ETL & Analysis | Python (pandas, numpy) | Data cleaning, feature engineering, pipeline |
| Modeling | scikit-learn, SHAP | Logistic Regression baseline → Random Forest |
| Visualization | matplotlib, seaborn | EDA charts, ROC curve, feature importance |
| Version Control | GitHub | Full history, reproducible pipeline |

---

## Methodology

### 1. Exploratory Data Analysis
- Default rate by customer segment (age band, income band, debt ratio)
- Distribution of late payment events
- Correlation analysis between payment behavior and delinquency

### 2. Feature Engineering
New features created on top of raw data:

| Feature | Description |
|---|---|
| `weighted_late_score` | Sum of late events weighted by severity (30d × 1, 60d × 2, 90d × 3) |
| `utilisation_ratio` | Debt ratio divided by credit limit |
| `income_band` | Categorical bucketing of monthly income |

### 3. Modeling

| Model | ROC-AUC | Precision (class 1) | Notes |
|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.62 | Baseline, interpretable |
| Random Forest | ~0.87 | ~0.71 | Best performance |

SHAP values are used to explain individual predictions and communicate results to non-technical stakeholders.

### 4. Dashboard
[View on Looker Studio →](#) *(link to be added after deployment)*

The dashboard includes:
- Risk score distribution across the portfolio
- High-risk customer list (score > 0.70)
- Default rate trend over time
- Top features driving each customer's score

---

## Key Results

- Random Forest model achieves **ROC-AUC of ~0.87** on the test set
- The top 10% highest-scored customers account for ~60% of all defaults
- `weighted_late_score` and `debt_ratio` are the strongest predictors

---

## Author

**[Geovana Novelli Soares]**
Data Science | Finance Analytics
---
