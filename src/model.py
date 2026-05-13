import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay,
)

FEATURES_PATH = Path("data/processed/features.csv")
REPORTS_DIR   = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "credit_limit", "age", "debt_ratio", "monthly_income",
    "num_open_credit_lines", "num_real_estate_loans", "num_dependents",
    "weighted_late_score", "utilisation_ratio", "income_band",
    "dependents_income_ratio",
]
TARGET = "target"


def load_data():
    df = pd.read_csv(FEATURES_PATH).dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS]
    y = df[TARGET]
    print(f"Dataset: {len(df):,} rows | default rate: {y.mean():.2%}")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_baseline(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5), scoring="roc_auc")
    print(f"Logistic Regression CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(5), scoring="roc_auc")
    print(f"Random Forest CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    rf.fit(X_train, y_train)
    return rf


def evaluate(model, X_test, y_test, name: str):
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\n{name} — Test AUC: {auc:.4f}")
    print(classification_report(y_test, (proba >= 0.5).astype(int), target_names=["On Time", "Default"]))
    return proba


def plot_roc(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curve — AR Risk Scoring Model")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    print("Saved: reports/roc_curve.png")


def plot_shap(rf_model, X_test):
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test.sample(500, random_state=42))

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values[1], X_test.sample(500, random_state=42),
                      feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Feature Importance — Default Class")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: reports/shap_summary.png")


def run():
    X_train, X_test, y_train, y_test = load_data()

    print("\n--- Baseline: Logistic Regression ---")
    lr = train_baseline(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")

    print("\n--- Random Forest ---")
    rf = train_random_forest(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")

    plot_roc({"Logistic Regression": lr, "Random Forest": rf}, X_test, y_test)
    plot_shap(rf, X_test)

    joblib.dump(rf, REPORTS_DIR / "rf_model.joblib")
    print("\nModel saved: reports/rf_model.joblib")


if __name__ == "__main__":
    run()
