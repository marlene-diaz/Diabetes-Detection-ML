"""
app.py

Streamlit app for the Diabetes Risk Screening tool.
Loads the trained model (src/model.pkl) and lets a user enter basic
health/demographic values to get a risk prediction.

Run locally with:
    streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "src/model.pkl"
METRICS_PATH = "src/metrics.json"

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["model_name"], bundle["features"]


@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


model, model_name, feature_cols = load_model()
metrics = load_metrics()

st.title("🩺 Diabetes Risk Screening")
st.write(
    "This tool estimates diabetes risk from basic health indicators, based on the "
    "Pima Indians Diabetes dataset. It is an educational demo, **not** a medical "
    "diagnosis — please consult a healthcare professional for real medical advice."
)

st.divider()
st.subheader("Enter patient information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        help="A score reflecting family history / genetic predisposition to diabetes.",
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

st.divider()

if st.button("Assess risk", type="primary"):
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=feature_cols,
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Higher risk of diabetes — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ Lower risk of diabetes — estimated probability: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "This estimate is based on a machine learning model trained on historical data "
        "and should not replace professional medical evaluation."
    )

with st.expander("ℹ️ About this model"):
    st.write(f"**Model used:** {model_name}")
    if metrics and model_name in metrics:
        m = metrics[model_name]
        st.write(
            f"- Accuracy: {m['accuracy']:.2%}\n"
            f"- Precision: {m['precision']:.2%}\n"
            f"- Recall: {m['recall']:.2%}\n"
            f"- F1 score: {m['f1']:.2%}\n"
            f"- ROC-AUC: {m['roc_auc']:.3f}"
        )
    st.write(
        "Dataset: [Pima Indians Diabetes Dataset]"
        "(https://www.kaggle.com/datasets/mathchi/diabetes-data-set)"
    )