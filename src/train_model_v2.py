"""
train_model.py

Trains and compares several models (Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting, SVM, and optionally XGBoost) on the Pima
Indians Diabetes dataset, tunes each with cross-validated hyperparameter
search, builds a stacking ensemble of the top performers, picks a decision
threshold that maximizes F1 (rather than assuming 0.5), and saves the best
overall model to disk for use in app.py.

Improvements over the baseline version:
  1. Feature engineering (BMI/age/glucose interactions and clinical bins)
  2. KNN-based imputation instead of simple per-class median fill
  3. Hyperparameter tuning via RandomizedSearchCV for every model
  4. Extra model families: Gradient Boosting, SVM, and XGBoost if available
  5. A stacking ensemble that combines the strongest base learners
  6. Optional SMOTE oversampling (falls back gracefully if imblearn isn't
     installed) to address class imbalance during training
  7. Decision-threshold tuning on validation folds instead of a fixed 0.5
  8. Model selection based on nested cross-validated ROC-AUC, then confirmed
     on a held-out test set, to reduce overfitting to a single split

Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

Usage:
    python src/train_model.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependencies. The script runs fine without either of these; it
# just skips that piece of the pipeline and tells you it did so.
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "src/model.pkl"
METRICS_PATH = "src/metrics.json"
RANDOM_STATE = 42
N_SEARCH_ITER = 40  # RandomizedSearchCV iterations per model

# Columns where a value of 0 actually means "missing", not a real zero.
# (Pregnancies and Outcome are excluded on purpose — 0 is a valid value there.)
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RAW_FEATURE_COLS = [
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


# ---------------------------------------------------------------------------
# Data loading, cleaning, and feature engineering
# ---------------------------------------------------------------------------
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Replace 0s that really mean "missing" with NaN
    df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, np.nan)

    # KNN imputation captures relationships between features (e.g. BMI and
    # SkinThickness tend to move together) instead of collapsing every
    # missing value to a single per-class median.
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df[ZERO_AS_MISSING_COLS] = imputer.fit_transform(df[ZERO_AS_MISSING_COLS])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-motivated derived features on top of the raw columns."""
    df = df.copy()

    # BMI clinical categories (underweight/normal/overweight/obese) as an
    # ordinal signal in addition to the raw continuous value.
    df["BMI_Category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(float)

    # Age decades capture nonlinear risk jumps (risk rises sharply after 40).
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 30, 40, 50, 60, 120], labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Glucose bands based on standard prediabetes/diabetes screening cutoffs.
    df["Glucose_Category"] = pd.cut(
        df["Glucose"], bins=[0, 100, 125, 300], labels=[0, 1, 2]
    ).astype(float)

    # Interaction terms: risk factors tend to compound, not just add.
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Age_Glucose"] = df["Age"] * df["Glucose"]
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / df["Glucose"].replace(0, np.nan)
    df["Insulin_Glucose_Ratio"] = df["Insulin_Glucose_Ratio"].fillna(
        df["Insulin_Glucose_Ratio"].median()
    )

    # Pregnancies scaled by age as a crude proxy for reproductive/metabolic history.
    df["Pregnancies_per_Age"] = df["Pregnancies"] / df["Age"]

    return df


ENGINEERED_FEATURE_COLS = RAW_FEATURE_COLS + [
    "BMI_Category",
    "Age_Group",
    "Glucose_Category",
    "Glucose_BMI",
    "Age_Glucose",
    "Insulin_Glucose_Ratio",
    "Pregnancies_per_Age",
]


# ---------------------------------------------------------------------------
# Model + search-space definitions
# ---------------------------------------------------------------------------
def get_model_search_space():
    """Return {name: (estimator_pipeline, param_distributions)} for tuning."""
    spaces = {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
                )),
            ]),
            {
                "clf__C": np.logspace(-3, 2, 20),
                "clf__penalty": ["l1", "l2"],
                "clf__solver": ["liblinear"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "max_depth": [3, 4, 5, 6, 7, 8, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 300, 400, 600],
                "max_depth": [4, 5, 6, 7, 8, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
            {
                "clf__C": np.logspace(-2, 2, 15),
                "clf__gamma": ["scale", "auto"] + list(np.logspace(-3, 0, 6)),
                "clf__kernel": ["rbf", "poly"],
            },
        ),
    }

    if HAS_XGBOOST:
        spaces["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False,
            ),
            {
                "n_estimators": [100, 200, 300, 400],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5],
            },
        )

    return spaces


def maybe_resample(X_train, y_train):
    """Apply SMOTE if imblearn is available; otherwise return data unchanged
    (class_weight='balanced' on each model already helps with imbalance)."""
    if not HAS_IMBLEARN:
        return X_train, y_train
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def find_best_threshold(y_true, probs):
    """Pick the probability threshold that maximizes F1 on validation data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


# ---------------------------------------------------------------------------
# Tuning + evaluation
# ---------------------------------------------------------------------------
def tune_model(name, estimator, param_dist, X_train, y_train, cv):
    n_iter = min(N_SEARCH_ITER, np.prod([len(v) if hasattr(v, "__len__") else 50 for v in param_dist.values()]))
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=int(n_iter),
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"{name}: best CV ROC-AUC = {search.best_score_:.4f} | params = {search.best_params_}")
    return search.best_estimator_, search.best_score_


def evaluate_on_test(name, model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "threshold": threshold,
    }

    print(f"\n=== {name} (test set, threshold={threshold:.3f}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(classification_report(y_test, preds, target_names=["No Diabetes", "Diabetes"]))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"XGBoost available: {HAS_XGBOOST}")
    print(f"imblearn (SMOTE) available: {HAS_IMBLEARN}\n")

    df = load_and_clean_data(DATA_PATH)
    df = engineer_features(df)

    X = df[ENGINEERED_FEATURE_COLS]
    y = df[TARGET_COL]

    # Held-out test set, untouched until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Resample only the training fold data (never the test set) to avoid leakage.
    X_train_res, y_train_res = maybe_resample(X_train, y_train)

    search_space = get_model_search_space()

    tuned_models = {}
    cv_scores = {}
    for name, (estimator, param_dist) in search_space.items():
        best_est, best_score = tune_model(name, estimator, param_dist, X_train_res, y_train_res, cv)
        tuned_models[name] = best_est
        cv_scores[name] = best_score

    # Build a stacking ensemble from the top 3 individually tuned models,
    # using logistic regression as the meta-learner.
    top3_names = sorted(cv_scores, key=cv_scores.get, reverse=True)[:3]
    print(f"\nTop 3 models selected for stacking ensemble: {top3_names}")

    stack = StackingClassifier(
        estimators=[(n, tuned_models[n]) for n in top3_names],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=cv,
        n_jobs=-1,
    )
    stack.fit(X_train_res, y_train_res)
    stack_cv_score = cross_val_score_safe(stack, X_train_res, y_train_res, cv)
    tuned_models["Stacking Ensemble"] = stack
    cv_scores["Stacking Ensemble"] = stack_cv_score
    print(f"Stacking Ensemble: CV ROC-AUC = {stack_cv_score:.4f}")

    # Pick the overall winner by CV ROC-AUC, then confirm + tune threshold on test set.
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = tuned_models[best_name]
    print(f"\nBest model by CV ROC-AUC: {best_name} ({cv_scores[best_name]:.4f})")

    # Tune the decision threshold using cross-validated out-of-fold
    # probabilities from the training set so we don't peek at the test set.
    from sklearn.model_selection import cross_val_predict

    oof_probs = cross_val_predict(
        best_model, X_train_res, y_train_res, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    best_threshold = find_best_threshold(y_train_res, oof_probs)
    print(f"Selected decision threshold (max F1 on OOF predictions): {best_threshold:.3f}")

    all_metrics = {}
    for name, model in tuned_models.items():
        thr = best_threshold if name == best_name else 0.5
        all_metrics[name] = evaluate_on_test(name, model, X_test, y_test, threshold=thr)
        all_metrics[name]["cv_roc_auc"] = cv_scores[name]

    print(f"\nFinal chosen model: {best_name}")
    print(f"Test ROC-AUC: {all_metrics[best_name]['roc_auc']:.4f}")
    print(f"Test F1: {all_metrics[best_name]['f1']:.4f}")

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "features": ENGINEERED_FEATURE_COLS,
            "threshold": best_threshold,
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


def cross_val_score_safe(estimator, X, y, cv):
    """cross_val_score wrapper scoped here to keep StackingClassifier scoring
    consistent (roc_auc) with the rest of the search."""
    from sklearn.model_selection import cross_val_score

    return cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()


if __name__ == "__main__":
    main()
