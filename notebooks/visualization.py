"""
visualization.py

Plots histogram distributions for each feature in the Pima Indians
Diabetes dataset (diabetes.csv).

Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set
"""

import pandas as pd
import matplotlib.pyplot as plt

# Column names for this dataset (the Kaggle CSV already has a header row,
# but this list is kept for clarity / reuse in other scripts)
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Columns where a value of 0 actually represents a missing measurement
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DATA_PATH = "data/diabetes.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Distribution of Features — Pima Indians Diabetes Dataset", fontsize=14)

    for ax, col in zip(axes.flatten(), COLUMNS):
        title = f"{col} (0 = missing)" if col in ZERO_AS_MISSING else col
        ax.hist(df[col], bins=20, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("notebooks/feature_distributions.png", dpi=150)
    plt.show()


def main():
    df = load_data()
    print(df.describe())
    plot_distributions(df)


if __name__ == "__main__":
    main()
