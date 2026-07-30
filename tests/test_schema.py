import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.schema import FEATURE_COLUMNS, TARGET_COLUMN, load_and_validate_data


def valid_row() -> dict:
    row = {column: 0 for column in FEATURE_COLUMNS}
    row.update(
        {
            TARGET_COLUMN: 0,
            "BMI": 25,
            "GenHlth": 2,
            "Age": 5,
            "Education": 5,
            "Income": 6,
        }
    )
    return row


class SchemaTests(unittest.TestCase):
    def write_csv(self, rows: list[dict], directory: str) -> Path:
        path = Path(directory) / "sample.csv"
        pd.DataFrame(rows, columns=[TARGET_COLUMN, *FEATURE_COLUMNS]).to_csv(
            path, index=False
        )
        return path

    def test_valid_data_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            data, report = load_and_validate_data(
                self.write_csv([valid_row()], directory)
            )

        self.assertEqual(len(data), 1)
        self.assertEqual(report["features"], 21)
        self.assertEqual(report["missing_values"], 0)

    def test_out_of_range_value_is_rejected(self):
        row = valid_row()
        row["GenHlth"] = 9

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv([row], directory)
            with self.assertRaisesRegex(ValueError, "GenHlth"):
                load_and_validate_data(path)

    def test_missing_value_is_rejected_in_clean_training_data(self):
        row = valid_row()
        row["BMI"] = None

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv([row], directory)
            with self.assertRaisesRegex(ValueError, "missing values"):
                load_and_validate_data(path)


if __name__ == "__main__":
    unittest.main()
