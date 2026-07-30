import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import explain_prediction, risk_band
from src.schema import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"


class TrainedArtifactTests(unittest.TestCase):
    @unittest.skipUnless(MODEL_PATH.exists(), "Train the model before this test.")
    def test_saved_model_matches_the_app_contract(self):
        bundle = joblib.load(MODEL_PATH)
        row = pd.DataFrame(
            [bundle["reference_values"]], columns=FEATURE_COLUMNS
        )

        probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
        band = risk_band(
            probability,
            bundle["low_threshold"],
            bundle["screening_threshold"],
        )
        explanation = explain_prediction(
            bundle["pipeline"], row, bundle["reference_values"], top_n=5
        )

        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
        self.assertIn(band, {
            "Lower screening signal",
            "Moderate screening signal",
            "Elevated screening signal",
        })
        self.assertEqual(len(explanation), 5)
        self.assertEqual(bundle["feature_columns"], FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
