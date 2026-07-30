"""Create a few focused exploratory plots for the BRFSS dataset.

This is deliberately a small EDA script rather than a large notebook. It answers
three beginner questions:

1. How imbalanced is the target?
2. How does the positive rate vary with general health?
3. How does the positive rate vary with age group?

Run:
    python -m notebooks.visualization
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.schema import TARGET_COLUMN, load_and_validate_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTPUT_PATH = ROOT / "notebooks" / "brfss_eda.png"


def main() -> None:
    data, report = load_and_validate_data(DATA_PATH)
    print(pd.Series(report).to_string())

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    counts = data[TARGET_COLUMN].value_counts().sort_index()
    axes[0].bar(["No diabetes", "Prediabetes/\ndiabetes"], counts.values)
    axes[0].set_title("Target class counts")
    axes[0].set_ylabel("Survey responses")

    general_health = data.groupby("GenHlth")[TARGET_COLUMN].mean()
    axes[1].bar(general_health.index.astype(str), general_health.values)
    axes[1].set_title("Positive rate by general health")
    axes[1].set_xlabel("1 = excellent, 5 = poor")
    axes[1].set_ylabel("Positive-class proportion")

    age = data.groupby("Age")[TARGET_COLUMN].mean()
    axes[2].plot(age.index, age.values, marker="o")
    axes[2].set_title("Positive rate by age group")
    axes[2].set_xlabel("Age category (1 = youngest, 13 = oldest)")
    axes[2].set_ylabel("Positive-class proportion")

    figure.suptitle("BRFSS 2015 Diabetes Health Indicators")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
