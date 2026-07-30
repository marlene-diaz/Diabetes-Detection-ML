# Explainable Diabetes Screening with BRFSS Survey Data

This is a fundamentals-first binary classification project. It compares three
machine-learning models and uses the most appropriate one in a Streamlit
screening demonstration.

The model estimates whether a person's survey-response pattern resembles the
CDC BRFSS dataset's:

- `0`: no-diabetes group
- `1`: combined prediabetes/diabetes group

It is a **current-status screening classifier**, not a diagnosis and not a
forecast of who will develop diabetes later.

## Final result

Logistic Regression was selected because its five-fold cross-validated ROC-AUC
was `0.823`, only `0.003` behind Random Forest's `0.826`. The project defined in
advance that Logistic Regression would be preferred when it was within `0.01`
of the strongest model because it is easier to explain.

At the training-selected screening threshold, the untouched test-set results
were:

| Metric | Result |
|---|---:|
| Accuracy | 76.7% |
| Balanced accuracy | 73.7% |
| Recall | 69.7% |
| Specificity | 77.8% |
| Precision | 33.7% |
| F1 | 45.4% |
| ROC-AUC | 0.819 |
| PR-AUC | 0.394 |
| Brier score | 0.100 |

The test set contained 50,736 responses:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actually negative | 33,971 | 9,696 |
| Actually positive | 2,145 | 4,924 |

Accuracy alone is not enough here. Because 86.1% of rows are negative, a useless
model that always predicts negative would have 86.1% accuracy while finding
zero positive cases. The selected screening threshold intentionally trades some
accuracy for substantially higher recall.

## What changed from the original repository

| Previous implementation | This branch |
|---|---|
| Pima dataset: 768 rows, 8 clinical measurements | BRFSS dataset: 253,680 rows, 21 survey indicators |
| Diabetes only | Prediabetes/diabetes combined |
| Required glucose, insulin, and skin-thickness inputs | Non-invasive health and lifestyle questionnaire |
| Target-based imputation leaked the correct answer | Preprocessing is fitted inside training-only pipelines |
| Test set used to choose the winner | Cross-validation chooses the model before test evaluation |
| Unvalidated 0.5 threshold | Threshold chosen from out-of-fold training predictions |
| Only a label and probability | Risk band, local explanation, metrics, and safety context |
| Model and app preprocessing could diverge | One saved pipeline performs preprocessing and prediction |
| README claims did not match files | Documentation follows the implemented workflow and actual results |

## Dataset

The project uses
`data/diabetes_binary_health_indicators_BRFSS2015.csv`:

- 253,680 survey responses
- 21 input features
- 35,346 positive responses
- 218,334 negative responses
- 13.93% positive-class prevalence
- no missing values in the cleaned release

Source and checksum details are in [data/README.md](data/README.md). The dataset
is cataloged by the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
and is derived from the CDC Behavioral Risk Factor Surveillance System.

## The three models

1. **Logistic Regression** — simple, fast, and coefficient-based.
2. **Decision Tree** — a shallow set of branching rules.
3. **Random Forest** — an average of many trees and the nonlinear benchmark.

There is deliberately no stacking, XGBoost, SMOTE, or large hyperparameter
search. The purpose is to demonstrate the classification fundamentals clearly.

## Project flow

```text
Validate BRFSS file
        |
        v
Stratified 80% training / 20% final test split
        |
        v
Five-fold cross-validation on training data
        |
        v
Select model using cross-validated ROC-AUC
with a predefined preference for interpretability
        |
        v
Create out-of-fold training probabilities
and select screening thresholds
        |
        v
Fit selected pipeline on all training rows
        |
        v
Evaluate once on untouched test rows
        |
        v
Save model + thresholds + schema + metrics
        |
        v
Streamlit questionnaire and explanations
```

## Repository structure

```text
.
├── app.py                     # Streamlit questionnaire and results
├── artifacts/
│   ├── model.joblib           # Fitted pipeline and app metadata
│   └── metrics.json           # Complete reproducible evaluation
├── data/
│   ├── README.md              # Dataset provenance and decisions
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
├── docs/
│   └── COMPLETE_BEGINNER_GUIDE.md  # Canonical detailed explanation
├── notebooks/
│   ├── visualization.py       # Focused EDA script
│   └── brfss_eda.png          # Generated EDA figure
├── src/
│   ├── schema.py              # Feature names, ranges, and validation
│   ├── modeling.py            # Models, thresholds, explanations
│   └── train_model.py         # Training and final evaluation
├── tests/
│   ├── test_app.py
│   ├── test_artifact.py
│   ├── test_modeling.py
│   └── test_schema.py
└── requirements.txt
```

## Run the project

Python 3.12 or newer is recommended.

### 1. Create and activate an environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run tests

```bash
python -m unittest discover -s tests -v
```

### 4. Reproduce training

```bash
python -m src.train_model
```

This regenerates `artifacts/model.joblib` and `artifacts/metrics.json`.

### 5. Generate the EDA figure

```bash
python -m notebooks.visualization
```

### 6. Start Streamlit

```bash
python -m streamlit run app.py
```

## Tests

The test suite checks:

- dataset columns, missing values, ranges, and survey codes;
- preprocessing being inside each model pipeline;
- the interpretable model-selection rule;
- training-based threshold selection;
- risk-band ordering;
- local explanation generation;
- saved-model compatibility with the app;
- complete Streamlit form submission without an exception.

All 10 tests pass in the completed branch.

## Important limitations

- BRFSS responses are self-reported.
- The data are cross-sectional, so this is not future-risk prediction.
- The target combines prediabetes and diabetes.
- The app is not a diagnostic device.
- `predict_proba` is a model-estimated likelihood, not a clinical probability.
- Age, sex, and income subgroup performance differs.
- The historical cleaned dataset includes only binary sex coding.
- Income, education, and healthcare access can encode structural inequities.
- Identical response rows cannot safely be called accidental duplicates because
  the cleaned file has no respondent identifier.
- A strong random-split result is not external clinical validation.

For the single complete beginner-friendly explanation—including the mental
model, terminology, feature dictionary, algorithms, pipeline, metrics,
explainability, neutral version comparison, local testing, retraining, and
isolated Streamlit deployment—read
[docs/COMPLETE_BEGINNER_GUIDE.md](docs/COMPLETE_BEGINNER_GUIDE.md).
