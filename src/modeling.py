"""Small, reusable modeling functions shared by training, tests, and the app."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MINIMUM_SCREENING_RECALL = 0.70
LOW_BAND_RECALL = 0.90
INTERPRETABILITY_TOLERANCE = 0.01


def build_candidate_models() -> dict[str, Pipeline]:
    """Return the three models promised in the original project plan."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=100,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=10,
                        min_samples_leaf=20,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def choose_model(cv_summary: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Choose a strong model while honoring the plan's interpretability goal."""
    best_name = max(
        cv_summary, key=lambda name: cv_summary[name]["roc_auc_mean"]
    )
    best_auc = cv_summary[best_name]["roc_auc_mean"]
    logistic_auc = cv_summary["Logistic Regression"]["roc_auc_mean"]

    if best_auc - logistic_auc <= INTERPRETABILITY_TOLERANCE:
        return (
            "Logistic Regression",
            "Logistic Regression was selected because its mean cross-validated "
            f"ROC-AUC ({logistic_auc:.3f}) was within "
            f"{INTERPRETABILITY_TOLERANCE:.2f} of the strongest model "
            f"({best_name}, {best_auc:.3f}), while remaining easier to explain.",
        )

    return (
        best_name,
        f"{best_name} was selected because it had the strongest mean "
        f"cross-validated ROC-AUC ({best_auc:.3f}) by more than the "
        "predefined interpretability tolerance.",
    )


def choose_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_screening_recall: float = MINIMUM_SCREENING_RECALL,
) -> tuple[float, float, dict]:
    """Choose statistical risk-band thresholds from training predictions only.

    The elevated threshold maximizes F1 among thresholds that retain the minimum
    screening recall. The lower threshold is the largest threshold that still
    retains 90% recall. These are model operating points, not medical cutoffs.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    eligible = np.flatnonzero(recall >= minimum_screening_recall)
    if len(eligible) == 0:
        raise ValueError("No threshold meets the minimum screening recall.")
    best_index = eligible[np.argmax(f1_values[eligible])]
    screening_threshold = float(thresholds[best_index])

    low_eligible = np.flatnonzero(recall >= LOW_BAND_RECALL)
    low_index = low_eligible[-1] if len(low_eligible) else 0
    low_threshold = float(thresholds[low_index])
    low_threshold = min(low_threshold, screening_threshold)

    report = {
        "source": "out-of-fold predictions from the training set only",
        "minimum_screening_recall": minimum_screening_recall,
        "low_band_target_recall": LOW_BAND_RECALL,
        "low_threshold": low_threshold,
        "screening_threshold": screening_threshold,
        "screening_threshold_training_recall": float(recall[best_index]),
        "screening_threshold_training_precision": float(precision[best_index]),
        "screening_threshold_training_f1": float(f1_values[best_index]),
        "medical_cutoffs": False,
    }
    return low_threshold, screening_threshold, report


def risk_band(
    probability: float, low_threshold: float, screening_threshold: float
) -> str:
    """Translate the score into statistically defined, non-diagnostic bands."""
    if probability >= screening_threshold:
        return "Elevated screening signal"
    if probability >= low_threshold:
        return "Moderate screening signal"
    return "Lower screening signal"


def explain_prediction(
    pipeline: Pipeline,
    row: pd.DataFrame,
    reference_values: dict[str, float],
    top_n: int = 5,
) -> list[dict]:
    """Explain one prediction by changing one feature to a typical value.

    This simple model-agnostic method asks: how much would the model score change
    if this one answer were replaced by the training-set median while all other
    answers stayed the same? It describes model behavior, not causation.
    """
    numeric_row = row.astype(float)
    original_probability = float(pipeline.predict_proba(numeric_row)[0, 1])
    contributions = []

    for feature in row.columns:
        comparison = numeric_row.copy()
        comparison.loc[comparison.index[0], feature] = reference_values[feature]
        comparison_probability = float(pipeline.predict_proba(comparison)[0, 1])
        contributions.append(
            {
                "feature": feature,
                "effect": original_probability - comparison_probability,
                "reference_value": reference_values[feature],
                "user_value": float(row.iloc[0][feature]),
            }
        )

    contributions.sort(key=lambda item: abs(item["effect"]), reverse=True)
    return contributions[:top_n]


def get_global_importance(
    pipeline: Pipeline, feature_columns: list[str]
) -> list[dict]:
    """Extract a simple global importance summary from the fitted model."""
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        return [
            {
                "feature": feature,
                "importance": float(abs(value)),
                "direction": "higher score" if value > 0 else "lower score",
                "signed_coefficient": float(value),
            }
            for feature, value in sorted(
                zip(feature_columns, values),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]

    values = classifier.feature_importances_
    return [
        {
            "feature": feature,
            "importance": float(value),
            "direction": "non-directional",
        }
        for feature, value in sorted(
            zip(feature_columns, values),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
