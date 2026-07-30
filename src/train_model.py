"""Train and evaluate the BRFSS diabetes screening models.

Run from the repository root:

    python -m src.train_model

The important design rule in this file is that the test set is separated before
anything is learned. Model selection and threshold selection use only the
training set. The test set is evaluated once at the end.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src.modeling import (
    MINIMUM_SCREENING_RECALL,
    RANDOM_STATE,
    build_candidate_models,
    choose_model,
    choose_thresholds,
    get_global_importance,
)
from src.schema import FEATURE_COLUMNS, TARGET_COLUMN, load_and_validate_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
TEST_SIZE = 0.20
CV_FOLDS = 5


def file_sha256(path: Path) -> str:
    """Return a checksum so the exact training data can be identified."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_predictions(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict:
    """Calculate understandable classification and probability metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "test_examples": int(len(y_true)),
        "test_positive_examples": int(np.sum(y_true)),
    }


def evaluate_subgroups(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, list[dict]]:
    """Report a small fairness audit without changing thresholds by group."""
    audit_frame = X_test[["Sex", "Age", "Income"]].copy()
    audit_frame["target"] = y_test
    audit_frame["prediction"] = (probabilities >= threshold).astype(int)

    audit_frame["Sex group"] = audit_frame["Sex"].map(
        {0: "Female survey code", 1: "Male survey code"}
    )
    audit_frame["Age group"] = pd.cut(
        audit_frame["Age"],
        bins=[0, 4, 8, 13],
        labels=["18–39", "40–59", "60 or older"],
    )
    audit_frame["Income group"] = pd.cut(
        audit_frame["Income"],
        bins=[0, 3, 6, 8],
        labels=["Below $20,000", "$20,000–$49,999", "$50,000 or more"],
    )

    report: dict[str, list[dict]] = {}
    for group_column in ("Sex group", "Age group", "Income group"):
        rows = []
        for group_name, group in audit_frame.groupby(group_column, observed=True):
            tn, fp, fn, tp = confusion_matrix(
                group["target"], group["prediction"], labels=[0, 1]
            ).ravel()
            rows.append(
                {
                    "group": str(group_name),
                    "rows": int(len(group)),
                    "positive_rows": int(group["target"].sum()),
                    "accuracy": float(accuracy_score(group["target"], group["prediction"])),
                    "precision": float(
                        precision_score(
                            group["target"], group["prediction"], zero_division=0
                        )
                    ),
                    "recall": float(
                        recall_score(
                            group["target"], group["prediction"], zero_division=0
                        )
                    ),
                    "false_positive_rate": float(fp / (fp + tn)),
                }
            )
        report[group_column] = rows
    return report


def main() -> None:
    print("Loading and validating the BRFSS dataset...")
    data, data_report = load_and_validate_data(DATA_PATH)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # This is the only place the final test set is created. It remains untouched
    # throughout model comparison and threshold selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    models = build_candidate_models()
    cv_summary: dict[str, dict[str, float]] = {}

    print("\nComparing models with cross-validation on the training set only...")
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        cv_summary[name] = {}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            cv_summary[name][f"{metric}_mean"] = float(np.mean(values))
            cv_summary[name][f"{metric}_std"] = float(np.std(values))

        print(
            f"{name:20s} "
            f"ROC-AUC={cv_summary[name]['roc_auc_mean']:.3f} | "
            f"PR-AUC={cv_summary[name]['pr_auc_mean']:.3f} | "
            f"Recall={cv_summary[name]['recall_mean']:.3f}"
        )

    selected_name, selection_reason = choose_model(cv_summary)
    selected_model = models[selected_name]
    print(f"\nSelected model: {selected_name}")
    print(selection_reason)

    # These predictions are out-of-fold: every training example is predicted by
    # a model that was not trained on that example.
    oof_probabilities = cross_val_predict(
        selected_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    low_threshold, screening_threshold, threshold_report = choose_thresholds(
        y_train.to_numpy(),
        oof_probabilities,
        minimum_screening_recall=MINIMUM_SCREENING_RECALL,
    )

    print(
        f"Training-only thresholds: lower={low_threshold:.3f}, "
        f"elevated={screening_threshold:.3f}"
    )

    # Only after the model and thresholds have been chosen do we fit on all
    # training rows and evaluate the held-out test rows.
    selected_model.fit(X_train, y_train)
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(
        y_test, test_probabilities, screening_threshold
    )
    default_threshold_metrics = evaluate_predictions(
        y_test, test_probabilities, threshold=0.5
    )
    training_prevalence = float(y_train.mean())
    simple_baselines = {
        "always_predict_negative_accuracy": float((y_test == 0).mean()),
        "always_predict_negative_recall": 0.0,
        "training_prevalence_probability": training_prevalence,
        "constant_prevalence_brier_score": float(
            brier_score_loss(
                y_test, np.full(len(y_test), training_prevalence)
            )
        ),
    }
    subgroup_metrics = evaluate_subgroups(
        X_test, y_test, test_probabilities, screening_threshold
    )

    importance_sample = X_test.sample(
        n=min(10_000, len(X_test)), random_state=RANDOM_STATE
    )
    importance_targets = y_test.loc[importance_sample.index]
    permutation = permutation_importance(
        selected_model,
        importance_sample,
        importance_targets,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    permutation_ranking = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in sorted(
            zip(
                FEATURE_COLUMNS,
                permutation.importances_mean,
                permutation.importances_std,
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    reference_values = {
        column: float(X_train[column].median()) for column in FEATURE_COLUMNS
    }
    global_importance = get_global_importance(selected_model, FEATURE_COLUMNS)

    metrics_document = {
        "model_name": selected_name,
        "selection_reason": selection_reason,
        "dataset": {
            **data_report,
            "path": str(DATA_PATH.relative_to(ROOT)),
            "sha256": file_sha256(DATA_PATH),
        },
        "split": {
            "random_state": RANDOM_STATE,
            "test_fraction": TEST_SIZE,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "stratified": True,
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "training_set_only": True,
            "models": cv_summary,
        },
        "threshold_selection": threshold_report,
        "test_metrics": test_metrics,
        "default_0_5_threshold_test_metrics": default_threshold_metrics,
        "simple_baselines": simple_baselines,
        "subgroup_audit": subgroup_metrics,
        "permutation_importance": permutation_ranking,
    }

    model_bundle = {
        "pipeline": selected_model,
        "model_name": selected_name,
        "feature_columns": FEATURE_COLUMNS,
        "reference_values": reference_values,
        "low_threshold": low_threshold,
        "screening_threshold": screening_threshold,
        "test_metrics": test_metrics,
        "global_importance": global_importance,
        "dataset_sha256": metrics_document["dataset"]["sha256"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_version": 1,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_document, handle, indent=2)

    print("\nFinal held-out test results")
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ):
        print(f"{metric:18s}: {test_metrics[metric]:.3f}")
    print(f"\nSaved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
