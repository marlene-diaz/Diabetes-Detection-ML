# Complete Beginner Guide: BRFSS Diabetes Screening Project

This is the single detailed guide for the `Diabetes-Dataset-Change` version of
the project. It explains the mental model, terminology, data, algorithms,
training pipeline, results, files, Streamlit application, tests, retraining, and
safe separate deployment.

## 1. What the project does

This is a supervised binary-classification project. A user provides 21
BRFSS-style survey answers. The model estimates whether that response pattern
more closely resembles the dataset's:

| Target | Meaning |
|---:|---|
| `0` | No-diabetes group |
| `1` | Combined prediabetes/diabetes group |

The app displays:

- a model-estimated likelihood;
- a lower, moderate, or elevated screening signal;
- the answers that changed the model score most;
- model-performance information;
- clear screening-not-diagnosis language.

This is a current-status educational screening classifier. It is not a medical
diagnosis, future-disease forecast, or treatment recommendation.

## 2. The simplest mental model

Think of the project this way:

| Component | Mental model |
|---|---|
| Dataset | Textbook of labeled examples |
| Training script | Teacher following a lesson plan |
| Algorithm | Mathematical learning method |
| Cross-validation | Several practice examinations |
| Test set | Final examination |
| `.joblib` artifact | Saved learned notes |
| Streamlit | Friendly calculator using those notes |

The complete lifecycle is:

```text
Historical survey rows with known groups
        ↓
Validate the data
        ↓
Separate training and testing rows
        ↓
Train and compare three algorithms
        ↓
Select one model and screening thresholds
        ↓
Evaluate on untouched test rows
        ↓
Save the complete fitted pipeline
        ↓
Streamlit loads the saved pipeline
        ↓
New questionnaire answers produce a score and explanation
```

The model does not understand diabetes like a clinician. It learns mathematical
associations from historical examples.

---

# Dataset and features

## 3. Dataset

The project uses:

```text
data/diabetes_binary_health_indicators_BRFSS2015.csv
```

It is the cleaned full binary 2015 CDC Behavioral Risk Factor Surveillance
System health-indicators dataset.

| Property | Value |
|---|---:|
| Survey responses | 253,680 |
| Input features | 21 |
| Negative rows | 218,334 |
| Positive rows | 35,346 |
| Positive prevalence | 13.93% |
| Missing values in cleaned release | 0 |

Sources:

- UCI catalog: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Dataset release: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Exact CSV SHA-256 checksum:

```text
19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0
```

A checksum is a file fingerprint. The model and metrics store this value so
they can be connected to the exact training file.

## 4. Why the natural class distribution is retained

The project uses the full naturally imbalanced file:

```text
86.07% negative
13.93% positive
```

It does not use the provided artificial 50/50 derivative. This allows the
project to demonstrate how prevalence affects accuracy, precision, recall, and
threshold choice.

## 5. Identical response rows

There are 24,206 rows identical to an earlier row across all columns. They are
retained.

The fields are broad survey categories and the cleaned file has no respondent
identifier. Different respondents can provide the same response pattern.
Removing those rows would require assuming they are accidental copies.

## 6. Feature dictionary

### Health history

| Feature | Values | Meaning |
|---|---|---|
| `HighBP` | 0/1 | Told they have high blood pressure |
| `HighChol` | 0/1 | Told they have high cholesterol |
| `CholCheck` | 0/1 | Cholesterol checked within five years |
| `Stroke` | 0/1 | Stroke history |
| `HeartDiseaseorAttack` | 0/1 | Coronary heart disease or heart-attack history |
| `DiffWalk` | 0/1 | Serious difficulty walking or climbing stairs |

### General health and wellbeing

| Feature | Values | Meaning |
|---|---|---|
| `BMI` | 12–98 | Body Mass Index stored as a whole number |
| `GenHlth` | 1–5 | General health: excellent through poor |
| `MentHlth` | 0–30 | Poor mental-health days in past 30 days |
| `PhysHlth` | 0–30 | Poor physical-health days in past 30 days |

### Lifestyle

| Feature | Values | Meaning |
|---|---|---|
| `Smoker` | 0/1 | Smoked at least 100 cigarettes in lifetime |
| `PhysActivity` | 0/1 | Physical activity outside work in past 30 days |
| `Fruits` | 0/1 | Fruit at least once daily |
| `Veggies` | 0/1 | Vegetables at least once daily |
| `HvyAlcoholConsump` | 0/1 | Meets the survey's heavy-consumption definition |

### Healthcare access

| Feature | Values | Meaning |
|---|---|---|
| `AnyHealthcare` | 0/1 | Has healthcare coverage |
| `NoDocbcCost` | 0/1 | Cost prevented a doctor visit in the past year |

### Demographic context

| Feature | Values | Meaning |
|---|---|---|
| `Sex` | 0/1 | Female/male coding in the cleaned historical file |
| `Age` | 1–13 | Age group from 18–24 through 80+ |
| `Education` | 1–6 | Ordered education category |
| `Income` | 1–8 | Ordered household-income category |

These are predictive inputs, not proven causes. A learned association describes
the model and dataset, not what would medically happen if one answer changed.

---

# Data validation and preprocessing

## 7. Schema validation

`src/schema.py` checks:

1. all expected columns exist;
2. no unexpected columns exist;
3. values are numeric;
4. the cleaned file has no missing values;
5. the target contains only `0` and `1`;
6. every feature is inside its documented range;
7. survey codes are whole numbers.

Training stops with a clear error if the file does not match the schema.

## 8. `X` and `y`

Common ML notation:

```text
X = input-feature matrix
y = target vector
```

Here:

```text
X = the 21 survey columns
y = Diabetes_binary
```

Each row in `X` is paired with the target in the same row of `y`.

## 9. Imputation

An imputer fills missing values. The cleaned training file currently has no
missing values, but each pipeline includes:

```text
SimpleImputer(strategy="median")
```

If a future numeric value is missing, it can use a median learned from training
data. Because the imputer is inside the pipeline, each cross-validation fold
learns its medians from that fold's training portion.

## 10. Standardization

The feature scales differ:

```text
HighBP: 0 or 1
BMI: 12 to 98
PhysHlth: 0 to 30
Age: 1 to 13
```

Logistic Regression uses StandardScaler:

```text
scaled value =
    (original value - training mean) / training standard deviation
```

Example:

```text
Training BMI mean = 28
Training standard deviation = 6
New BMI = 34

Scaled BMI = (34 - 28) / 6 = 1
```

The value is one training standard deviation above the mean.

Trees do not need scaling because they split directly on thresholds.

## 11. Pipeline

A pipeline combines preprocessing and classification:

```text
Raw survey row
    ↓
Median imputer
    ↓
StandardScaler when needed
    ↓
Classifier
```

The same fitted pipeline is used for training, testing, and Streamlit. The app
does not manually recreate preprocessing.

---

# Training, testing, and cross-validation

## 12. Train/test split

| Portion | Rows | Purpose |
|---|---:|---|
| Training | 202,944 | Learn, compare, select, and fit |
| Test | 50,736 | Final evaluation |

The split is:

- 80/20;
- stratified so class proportions remain similar;
- reproducible using `random_state=42`.

The model and thresholds are selected using training data. The final test set is
evaluated after those choices.

## 13. Five-fold cross-validation

The training set is divided into five stratified folds:

```text
Round 1: validate fold 1, train on folds 2–5
Round 2: validate fold 2, train on folds 1,3–5
Round 3: validate fold 3, train on folds 1–2,4–5
Round 4: validate fold 4, train on folds 1–3,5
Round 5: validate fold 5, train on folds 1–4
```

Each row is validated once by a model that did not fit on that row. The results
are averaged to compare candidates.

## 14. Out-of-fold predictions

Out-of-fold probabilities are training-set predictions made by models that did
not train on those specific rows.

```text
Training rows
    ↓
Five-fold prediction
    ↓
One out-of-fold likelihood per training row
    ↓
Threshold selection
```

This lets the project choose thresholds without using final test labels.

---

# The three model techniques

## 15. Logistic Regression

Logistic Regression is a classifier. It learns:

- one intercept;
- one coefficient for each of 21 features.

Conceptually:

```text
linear score =
    intercept
    + coefficient_1 × standardized feature_1
    + ...
    + coefficient_21 × standardized feature_21
```

It converts the unrestricted linear score into a value between zero and one:

```text
likelihood = 1 / (1 + e^(-linear score))
```

A positive coefficient tends to move the model score upward as the coded value
increases while the other model inputs are held fixed. A negative coefficient
tends to move it downward. This is model interpretation, not causation.

Strengths:

- fast;
- compact;
- stable;
- coefficient-based;
- easy to explain.

Tradeoffs:

- uses an additive linear structure in log-odds;
- does not automatically represent every complex interaction;
- correlated features can affect coefficient interpretation.

## 16. Decision Tree

A Decision Tree learns branching threshold questions:

```text
Is general health fair or poor?
├── Yes
│   └── Is BMI above a learned value?
│       ├── Yes → one leaf
│       └── No  → another leaf
└── No
    └── Is age above a learned category?
        ├── Yes → one leaf
        └── No  → another leaf
```

The tree is limited to:

```text
maximum depth = 5
minimum rows per leaf = 100
```

Strengths:

- intuitive rules;
- nonlinear thresholds;
- no scaling requirement.

Tradeoffs:

- one tree can vary with the data;
- a small tree represents broad patterns rather than every interaction.

## 17. Random Forest

A Random Forest trains many Decision Trees using randomized information and
averages their outputs.

This project uses 150 trees:

```text
Tree 1 estimate
Tree 2 estimate
...
Tree 150 estimate
        ↓
Average estimate
```

Strengths:

- nonlinear patterns;
- interactions;
- more stable than one tree;
- strong tabular benchmark.

Tradeoffs:

- more computation;
- less direct interpretation than one coefficient table;
- individual explanations need an additional method.

---

# Model selection and thresholds

## 18. Cross-validation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 86.39% | 53.91% | 15.80% | 24.43% | 0.823 | 0.408 |
| Decision Tree | 86.47% | 56.71% | 12.84% | 20.84% | 0.805 | 0.376 |
| Random Forest | 86.59% | 62.91% | 9.17% | 16.01% | 0.826 | 0.429 |

These class metrics use the ordinary 0.5 threshold. ROC-AUC and PR-AUC measure
ranking across thresholds.

## 19. Selection rule

1. Find the highest mean cross-validated ROC-AUC.
2. If Logistic Regression is within `0.01`, select it for interpretability.
3. Otherwise select the stronger candidate.

```text
Random Forest ROC-AUC = 0.826
Logistic Regression ROC-AUC = 0.823
Difference = 0.003
```

The difference is inside the `0.01` tolerance, so Logistic Regression is
selected.

## 20. Thresholds

A likelihood becomes a classification only after applying a threshold.

At the common 0.5 threshold on the final test set:

| Metric | Result |
|---|---:|
| Accuracy | 86.2% |
| Recall | 15.8% |
| Precision | 51.7% |
| True positives | 1,119 |
| False negatives | 5,950 |

For a screening emphasis, the elevated threshold must achieve at least 70%
recall on out-of-fold training predictions. Among eligible values, the project
selects the strongest F1.

Selected elevated threshold:

```text
0.170847
```

Training out-of-fold behavior:

| Metric | Result |
|---|---:|
| Recall | 70.00% |
| Precision | 33.76% |
| F1 | 45.55% |

The lower threshold is the largest training threshold retaining 90% recall:

```text
0.079517
```

Risk bands:

| Band | Rule |
|---|---|
| Lower | Below 0.0795 |
| Moderate | 0.0795 to below 0.1708 |
| Elevated | At least 0.1708 |

These are statistical operating points, not clinical diagnostic cutoffs.

---

# Final results and metric definitions

## 21. Confusion matrix

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actually negative | 33,971 | 9,696 |
| Actually positive | 2,145 | 4,924 |

| Term | Meaning | Count |
|---|---|---:|
| True negative | Negative row correctly unflagged | 33,971 |
| False positive | Negative row flagged | 9,696 |
| False negative | Positive row missed | 2,145 |
| True positive | Positive row correctly flagged | 4,924 |

## 22. Final metrics

| Metric | Result |
|---|---:|
| Accuracy | 76.7% |
| Balanced accuracy | 73.7% |
| Recall/sensitivity | 69.7% |
| Specificity | 77.8% |
| Precision | 33.7% |
| F1 | 45.4% |
| ROC-AUC | 0.819 |
| PR-AUC | 0.394 |
| Brier score | 0.100 |

### Accuracy

```text
(true positives + true negatives) / all rows
```

The always-negative baseline has 86.1% accuracy because 86.1% of rows are
negative, but its positive recall is zero. Accuracy is therefore read alongside
recall, specificity, and precision.

### Recall

Out of actual positives, how many were caught?

```text
TP / (TP + FN) = 69.7%
```

### Specificity

Out of actual negatives, how many were left unflagged?

```text
TN / (TN + FP) = 77.8%
```

### Precision

Out of elevated flags, how many belonged to the positive group?

```text
TP / (TP + FP) = 33.7%
```

Precision is strongly affected by positive-class prevalence.

### F1

The harmonic mean of precision and recall:

```text
2 × precision × recall / (precision + recall)
```

### Balanced accuracy

The average of recall and specificity. It gives both classes equal weight.

### ROC-AUC

Measures how well the model ranks positive rows above negative rows across
thresholds:

```text
0.5 = random ranking
1.0 = perfect ranking
```

Cross-validation ROC-AUC was 0.823 and final test ROC-AUC was 0.819.

### PR-AUC

Summarizes precision/recall tradeoffs. The no-skill reference is approximately
the positive prevalence:

```text
prevalence = 0.139
test PR-AUC = 0.394
```

### Brier score

Average squared difference between likelihood and target. Lower is better:

| Predictor | Brier score |
|---|---:|
| Model | 0.100 |
| Constant training prevalence | 0.120 |

This is useful probability evidence but is not full clinical calibration.

---

# Explainability and subgroup evaluation

## 23. Global importance

Permutation importance:

1. measures held-out ROC-AUC;
2. shuffles one feature;
3. measures ROC-AUC again;
4. records the decrease.

Top features:

| Feature | Mean ROC-AUC decrease |
|---|---:|
| General health | 0.0610 |
| BMI | 0.0292 |
| Age | 0.0212 |
| High blood pressure | 0.0190 |
| High cholesterol | 0.0104 |

Importance describes model reliance, not medical causation.

## 24. Individual explanations

For each input:

1. calculate the original likelihood;
2. replace one feature with its training median;
3. calculate the comparison likelihood;
4. subtract comparison from original;
5. rank absolute changes;
6. display the five largest.

Example:

```text
Original likelihood = 30%
Likelihood with typical HighBP value = 23%
Effect = +7 percentage points
```

This explains model behavior relative to a typical training value.

Smoke tests:

```text
Lower-profile row: 0.58%, lower signal
Higher-profile row: 84.14%, elevated signal
```

## 25. Subgroup audit

The same threshold is evaluated across broad groups.

### Historical survey sex code

| Group | Recall | Precision | False-positive rate |
|---|---:|---:|---:|
| Female code | 68.9% | 33.6% | 20.1% |
| Male code | 70.5% | 33.7% | 24.9% |

### Age

| Group | Recall | Precision | False-positive rate |
|---|---:|---:|---:|
| 18–39 | 34.5% | 27.0% | 3.2% |
| 40–59 | 61.2% | 33.9% | 14.2% |
| 60+ | 74.9% | 33.8% | 36.1% |

### Income

| Group | Recall | Precision | False-positive rate |
|---|---:|---:|---:|
| Below $20,000 | 84.6% | 37.8% | 44.9% |
| $20,000–$49,999 | 73.5% | 33.0% | 29.4% |
| $50,000+ | 55.0% | 30.9% | 12.9% |

This is a basic model-behavior audit, not a complete fairness certification.

---

# The `.joblib` file

## 26. Purpose

`artifacts/model.joblib` is a serialized Python object.

Serialization converts a fitted in-memory model into bytes:

```python
joblib.dump(bundle, "artifacts/model.joblib")
```

The app loads it later:

```python
bundle = joblib.load("artifacts/model.joblib")
```

## 27. Contents

The bundle contains:

- fitted preprocessing/model pipeline;
- model name;
- exact feature order;
- training reference values;
- lower threshold;
- elevated threshold;
- test metrics;
- global importance;
- dataset checksum;
- training timestamp;
- bundle version.

The fitted pipeline contains learned medians, scaling values, the Logistic
Regression intercept, and 21 coefficients.

## 28. Why save it?

```text
Train once
    ↓
Save artifact
    ↓
Load many times for prediction
```

Streamlit therefore does not need to load the entire dataset or retrain whenever
the app starts.

Only load trusted `.joblib` or `.pkl` files. Python serialization can execute
code while loading. Compatible Python, scikit-learn, NumPy, and joblib versions
also matter; `requirements.txt` pins the verified versions.

---

# File-by-file guide

## 29. Runtime files

| File | Necessity |
|---|---|
| `app.py` | Streamlit questionnaire, prediction, display |
| `artifacts/model.joblib` | Fitted pipeline and app metadata |
| `src/modeling.py` | Bands and explanation functions |
| `src/schema.py` | Feature order, labels, ranges, options |
| `src/__init__.py` | Makes `src` importable as a package |
| `requirements.txt` | Dependency versions |

## 30. Training files

| File | Necessity |
|---|---|
| BRFSS CSV | Labeled training examples |
| `src/train_model.py` | Complete training/evaluation workflow |
| `src/modeling.py` | Three candidate models and selection logic |
| `src/schema.py` | Data validation |

## 31. Generated outputs

| File | Meaning |
|---|---|
| `artifacts/model.joblib` | Saved fitted pipeline |
| `artifacts/metrics.json` | Detailed machine-readable evaluation |
| `notebooks/brfss_eda.png` | Generated EDA image |

## 32. Analysis and tests

| File | Purpose |
|---|---|
| `notebooks/visualization.py` | Regenerates EDA |
| `tests/test_schema.py` | Validates data rules |
| `tests/test_modeling.py` | Validates pipelines, thresholds, explanations |
| `tests/test_artifact.py` | Validates saved model/app contract |
| `tests/test_app.py` | Loads and submits Streamlit app |

## 33. Documentation

| File | Purpose |
|---|---|
| `README.md` | Concise overview and quick start |
| `data/README.md` | Dataset provenance |
| `docs/COMPLETE_BEGINNER_GUIDE.md` | This complete canonical guide |

---

# Streamlit

## 34. Streamlit mental model

Streamlit converts Python UI commands into a web application:

```python
st.title(...)
st.selectbox(...)
st.number_input(...)
st.metric(...)
```

When a user interacts, Streamlit reruns the script while preserving widget
state.

## 35. App startup

```text
Start Streamlit
    ↓
Import schema and modeling functions
    ↓
Load model.joblib
    ↓
Render questionnaire
```

The artifact is cached so it is not repeatedly loaded during one app process.

## 36. Form submission

```text
User answers 21 questions
    ↓
App maps labels to dataset codes
    ↓
Create one-row DataFrame in exact feature order
    ↓
pipeline.predict_proba(input_row)
    ↓
Extract positive-class likelihood
    ↓
Apply saved thresholds
    ↓
Generate local explanation
    ↓
Display result and scope language
```

`predict_proba` returns:

```text
[class-0 likelihood, class-1 likelihood]
```

The app uses the class-1 value and calls it a model-estimated likelihood rather
than a clinically validated personal probability.

---

# Local setup, testing, and retraining

## 37. Create the environment

macOS/Linux:

```bash
cd /Users/Sangeeta/AI4ALL/Diabetes-Detection-ML
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install:

```bash
python -m pip install -r requirements.txt
```

## 38. Run tests

```bash
python -m unittest discover -s tests -v
```

Expected ending:

```text
Ran 10 tests
OK
```

Dependency warnings can appear during bare Streamlit or joblib tests. `OK`
indicates success.

## 39. Run Streamlit locally

```bash
python -m streamlit run app.py
```

Open the printed URL, usually:

```text
http://localhost:8501
```

Complete the questionnaire and click **Estimate screening signal**.

Stop with:

```text
Control + C
```

## 40. Regenerate EDA

```bash
python -m notebooks.visualization
```

## 41. Retrain

Run:

```bash
python -m unittest discover -s tests -v
python -m src.train_model
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Training replaces:

```text
artifacts/model.joblib
artifacts/metrics.json
```

Starting Streamlit does not retrain.

---

# Neutral comparison with the earlier version

## 42. Shared approach

Both versions use supervised binary classification:

```text
Labeled dataset
    ↓
Train candidate models
    ↓
Select and save a model
    ↓
Streamlit loads saved model
    ↓
New input produces a result
```

Both train a project-specific model. Neither uses a downloaded general-purpose
pretrained diabetes classifier.

## 43. Differences

| Area | Earlier version | BRFSS changed version |
|---|---|---|
| Dataset | Pima diabetes data | BRFSS 2015 health indicators |
| Rows | 768 | 253,680 |
| Inputs | 8 | 21 |
| Input style | Clinical/demographic measurements | Survey health/lifestyle indicators |
| Target | Diabetes/no diabetes | Prediabetes-or-diabetes/no diabetes |
| Candidate algorithms | Logistic, Tree, Forest | Logistic, Tree, Forest |
| Selected algorithm | Random Forest | Logistic Regression |
| Artifact | `src/model.pkl` | `artifacts/model.joblib` |
| User form | Eight measurements | 21 survey questions |
| Selection emphasis | Reported ROC-AUC | Cross-validation plus interpretability rule |
| Output | Classification and likelihood | Likelihood, band, and local explanation |
| Documentation | Concise overview | Single detailed teaching guide |

Earlier lifecycle:

```text
Pima CSV
    ↓
Train three candidates
    ↓
Select Random Forest
    ↓
Save model.pkl
    ↓
Eight-input Streamlit form
```

Changed-version lifecycle:

```text
BRFSS CSV
    ↓
Validate 21 survey inputs
    ↓
Cross-validate three candidates
    ↓
Select Logistic Regression
    ↓
Select screening thresholds
    ↓
Save model.joblib
    ↓
21-question Streamlit form with explanation
```

The two versions demonstrate related approaches to an educational diabetes
classification task.

---

# Separate Streamlit deployment

## 44. Will deployment change the teammate's branch?

No.

Streamlit reads a selected GitHub branch. Creating a deployment does not write
changes into that branch or another branch.

Branches change only through Git actions such as commits, pushes, or merges.

## 45. Will it change the teammate's deployed app?

No, provided you create a **new Streamlit app** with:

```text
Repository: shared repository
Branch: Diabetes-Dataset-Change
Entrypoint: app.py
URL: a new separate URL
```

The deployments are:

```text
Existing app:
repository + teammate branch + app.py → existing URL

New app:
repository + Diabetes-Dataset-Change + app.py → new URL
```

Official deployment guide:

https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 46. Keep the apps isolated

- Create a new app instead of editing the existing deployment.
- Select `Diabetes-Dataset-Change`.
- Choose a new URL.
- Do not change the existing deployment's branch or entrypoint.
- Do not merge this branch into the teammate's tracked branch unless the team
  later chooses to.

## 47. Review, commit, and push

```bash
git branch --show-current
git status
git diff --stat
git diff
```

Expected branch:

```text
Diabetes-Dataset-Change
```

Then:

```bash
git add .
git commit -m "Rebuild diabetes screening project with BRFSS data"
git push -u origin Diabetes-Dataset-Change
```

Pushing this branch does not merge it into `main`.

## 48. Deploy

1. Open Streamlit Community Cloud.
2. Select **Create app**.
3. Select the shared repository.
4. Select `Diabetes-Dataset-Change`.
5. Select `app.py`.
6. Choose a new URL.
7. Deploy.
8. Review build logs.
9. Submit several test profiles.

Runtime requirements include:

```text
app.py
requirements.txt
src/
artifacts/model.joblib
```

The CSV and training code remain in the branch for reproducibility, but ordinary
app startup loads the saved artifact.

---

# Responsible scope and final mental model

## 49. Appropriate description

> An explainable educational binary-classification project using BRFSS survey
> indicators to identify response patterns associated with the dataset's
> combined prediabetes/diabetes group.

## 50. Main limitations

- Survey responses are self-reported.
- The data are cross-sectional.
- Prediabetes and diabetes are combined.
- Historical sex coding is binary.
- Income and access-to-care variables reflect social context.
- Group performance varies.
- Model likelihood is not a clinically validated probability.
- Explanations describe model behavior, not causation.
- Random-split evaluation is not external clinical validation.

## 51. Verification completed locally

- 10 automated tests pass.
- Python compilation passes.
- Dataset/model/metrics checksum contract passes.
- Streamlit form submission passes.
- Local Streamlit health endpoint returned `ok`.
- EDA output exists.
- Lower- and higher-profile smoke tests behave consistently.

The project has not been committed, pushed, merged, or deployed. Those actions
remain under team control.

## 52. Complete mental model

```text
1. CSV provides historical survey rows and labels.
2. schema.py validates their structure.
3. train_model.py separates training and test rows.
4. modeling.py builds three candidate pipelines.
5. Cross-validation compares candidates on training data.
6. Logistic Regression is selected under the predefined rule.
7. Out-of-fold training predictions select thresholds.
8. The selected pipeline fits all training rows.
9. Test rows generate final metrics.
10. model.joblib stores the fitted pipeline and metadata.
11. metrics.json stores the experiment report.
12. app.py loads model.joblib without retraining.
13. The user submits 21 answers.
14. The pipeline preprocesses them consistently.
15. Logistic Regression calculates the likelihood.
16. Saved thresholds assign the screening band.
17. Feature comparisons generate the explanation.
18. Streamlit displays the result and responsible-use language.
```

The essential idea is:

> Learn once from labeled historical examples, save the complete learned
> pipeline, and consistently apply it to new survey responses.

