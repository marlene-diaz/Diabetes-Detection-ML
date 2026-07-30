# Dataset

This branch uses:

`diabetes_binary_health_indicators_BRFSS2015.csv`

It is the cleaned binary version of the CDC Behavioral Risk Factor Surveillance
System (BRFSS) health-indicators dataset referenced by the original project
plan.

## Source

- UCI catalog and variable descriptions:
  https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Dataset release referenced by UCI:
  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- UCI DOI: https://doi.org/10.24432/C53919

The archive was downloaded from the public Kaggle dataset API. The exact binary
CSV used by this project has this SHA-256 checksum:

```text
19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0
```

## Shape and target

- 253,680 survey responses
- 21 input features
- `Diabetes_binary` target
- `0`: no diabetes
- `1`: prediabetes or diabetes
- 35,346 positive rows (13.93%)
- no missing values in this cleaned release

The project uses the naturally imbalanced full dataset, not the artificially
balanced 50/50 version. This keeps the class prevalence closer to the source
release and makes accuracy, precision, and probability estimates more
meaningful.

## Identical rows

There are 24,206 rows that are identical to an earlier row across all 22
columns. They are not automatically removed.

Most features are binary or broad categories, and the cleaned file does not
include a respondent identifier. Two different respondents can therefore have
exactly the same response pattern. Removing every identical row would assume
they are accidental duplicates and would change the observed class distribution
without evidence. The training report records this issue as a limitation.

## Important limitation

This is cross-sectional, self-reported survey data. The target describes the
respondent's recorded diabetes/prediabetes group at survey time. It does not
show who developed diabetes later, so the model is a current-status screening
classifier—not a future-disease forecast or diagnostic test.
