"""
train_model.py

Trains and compares three models (Logistic Regression, Decision Tree,
Random Forest) on the Pima Indians Diabetes dataset, then saves the
best-performing model to disk for use in app.py.

Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

Usage:
    python src/train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "src/model.pkl"
METRICS_PATH = "src/metrics.json"
RANDOM_STATE = 42

# Columns where a value of 0 actually means "missing", not a real zero.
# (Pregnancies and Outcome are excluded on purpose — 0 is a valid value there.)
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COL = "Outcome"


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Replace 0s that really mean "missing" with NaN
    df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, np.nan)

    # Impute with the median of each column, computed per Outcome class
    # so imputation doesn't wash out the signal between the two groups.
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df.groupby(TARGET_COL)[col].transform(lambda s: s.fillna(s.median()))

    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(classification_report(y_test, preds, target_names=["No Diabetes", "Diabetes"]))

    return metrics


def main():
    df = load_and_clean_data(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }

    all_metrics = {}
    fitted_models = {}
    for name, model in models.items():
        all_metrics[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test, cv)
        fitted_models[name] = model

    # Pick the best model by test-set ROC-AUC (robust to the class imbalance here)
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["roc_auc"])
    best_model = fitted_models[best_name]

    print(f"\nBest model: {best_name} (ROC-AUC = {all_metrics[best_name]['roc_auc']:.4f})")

    joblib.dump({"model": best_model, "model_name": best_name, "features": FEATURE_COLS}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()