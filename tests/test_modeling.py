import unittest

import numpy as np
import pandas as pd

from src.modeling import (
    build_candidate_models,
    choose_model,
    choose_thresholds,
    explain_prediction,
    risk_band,
)
from src.schema import FEATURE_COLUMNS


def small_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    rows = []
    targets = []
    for index in range(80):
        high_bp = index % 2
        row = {column: int(rng.integers(0, 2)) for column in FEATURE_COLUMNS}
        row.update(
            {
                "HighBP": high_bp,
                "BMI": int(rng.integers(18, 45)),
                "GenHlth": int(rng.integers(1, 6)),
                "MentHlth": int(rng.integers(0, 31)),
                "PhysHlth": int(rng.integers(0, 31)),
                "Age": int(rng.integers(1, 14)),
                "Education": int(rng.integers(1, 7)),
                "Income": int(rng.integers(1, 9)),
            }
        )
        rows.append(row)
        targets.append(high_bp)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), pd.Series(targets)


class ModelingTests(unittest.TestCase):
    def test_candidate_models_are_leakage_safe_pipelines(self):
        for model in build_candidate_models().values():
            self.assertIn("imputer", model.named_steps)
            self.assertIn("classifier", model.named_steps)

    def test_interpretable_model_is_preferred_when_performance_is_close(self):
        summary = {
            "Logistic Regression": {"roc_auc_mean": 0.800},
            "Decision Tree": {"roc_auc_mean": 0.790},
            "Random Forest": {"roc_auc_mean": 0.807},
        }
        name, reason = choose_model(summary)
        self.assertEqual(name, "Logistic Regression")
        self.assertIn("easier to explain", reason)

    def test_thresholds_come_from_provided_training_predictions(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.60, 0.75, 0.90])
        low, elevated, report = choose_thresholds(y, probabilities, 0.70)

        self.assertLessEqual(low, elevated)
        self.assertGreaterEqual(report["screening_threshold_training_recall"], 0.70)
        self.assertFalse(report["medical_cutoffs"])

    def test_risk_bands_are_ordered(self):
        self.assertEqual(risk_band(0.05, 0.10, 0.30), "Lower screening signal")
        self.assertEqual(risk_band(0.20, 0.10, 0.30), "Moderate screening signal")
        self.assertEqual(risk_band(0.40, 0.10, 0.30), "Elevated screening signal")

    def test_local_explanation_returns_ranked_effects(self):
        X, y = small_training_data()
        model = build_candidate_models()["Logistic Regression"]
        model.fit(X, y)
        references = {column: float(X[column].median()) for column in FEATURE_COLUMNS}

        explanation = explain_prediction(model, X.iloc[[0]], references, top_n=3)

        self.assertEqual(len(explanation), 3)
        self.assertGreaterEqual(
            abs(explanation[0]["effect"]), abs(explanation[-1]["effect"])
        )


if __name__ == "__main__":
    unittest.main()
