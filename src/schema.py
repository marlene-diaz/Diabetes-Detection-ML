"""Dataset schema and beginner-friendly feature descriptions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Diabetes_binary"

FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

BINARY_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

FEATURE_RANGES = {
    **{column: (0, 1) for column in BINARY_FEATURES},
    "BMI": (12, 98),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

FEATURE_LABELS = {
    "HighBP": "High blood pressure history",
    "HighChol": "High cholesterol history",
    "CholCheck": "Cholesterol checked in the past 5 years",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoked at least 100 cigarettes in lifetime",
    "Stroke": "Stroke history",
    "HeartDiseaseorAttack": "Heart disease or heart attack history",
    "PhysActivity": "Physical activity in the past 30 days",
    "Fruits": "Fruit at least once per day",
    "Veggies": "Vegetables at least once per day",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has healthcare coverage",
    "NoDocbcCost": "Could not see a doctor because of cost",
    "GenHlth": "General health",
    "MentHlth": "Poor mental-health days in the past 30 days",
    "PhysHlth": "Poor physical-health days in the past 30 days",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "Sex": "Survey sex category",
    "Age": "Age group",
    "Education": "Education level",
    "Income": "Household income group",
}

GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}

AGE_OPTIONS = {
    "18–24": 1,
    "25–29": 2,
    "30–34": 3,
    "35–39": 4,
    "40–44": 5,
    "45–49": 6,
    "50–54": 7,
    "55–59": 8,
    "60–64": 9,
    "65–69": 10,
    "70–74": 11,
    "75–79": 12,
    "80 or older": 13,
}

EDUCATION_OPTIONS = {
    "Never attended school / kindergarten only": 1,
    "Elementary school": 2,
    "Some high school": 3,
    "High school graduate": 4,
    "Some college or technical school": 5,
    "College graduate": 6,
}

INCOME_OPTIONS = {
    "Less than $10,000": 1,
    "$10,000–$14,999": 2,
    "$15,000–$19,999": 3,
    "$20,000–$24,999": 4,
    "$25,000–$34,999": 5,
    "$35,000–$49,999": 6,
    "$50,000–$74,999": 7,
    "$75,000 or more": 8,
}


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the CSV and stop with a clear error if its structure is unexpected."""
    data = pd.read_csv(path)
    expected = [TARGET_COLUMN, *FEATURE_COLUMNS]

    missing_columns = sorted(set(expected) - set(data.columns))
    extra_columns = sorted(set(data.columns) - set(expected))
    if missing_columns or extra_columns:
        raise ValueError(
            f"Unexpected dataset columns. Missing={missing_columns}; extra={extra_columns}"
        )

    data = data[expected].apply(pd.to_numeric, errors="raise")
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"The cleaned dataset contains missing values:\n{missing[missing > 0]}")

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    for column, (minimum, maximum) in FEATURE_RANGES.items():
        invalid = ~data[column].between(minimum, maximum)
        if invalid.any():
            raise ValueError(
                f"{column} has {int(invalid.sum())} values outside "
                f"the expected range {minimum}–{maximum}."
            )

    integer_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    for column in integer_columns:
        if not (data[column] % 1 == 0).all():
            raise ValueError(f"{column} should contain whole-number survey codes.")
        data[column] = data[column].astype("int16")

    duplicate_rows = int(data.duplicated().sum())
    positive_count = int(data[TARGET_COLUMN].sum())
    report = {
        "rows": int(len(data)),
        "features": len(FEATURE_COLUMNS),
        "positive_rows": positive_count,
        "negative_rows": int(len(data) - positive_count),
        "positive_rate": float(data[TARGET_COLUMN].mean()),
        "missing_values": int(data.isna().sum().sum()),
        "identical_rows": duplicate_rows,
        "identical_rows_removed": 0,
    }
    return data, report
