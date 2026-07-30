"""Streamlit interface for the BRFSS diabetes screening model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.modeling import explain_prediction, risk_band
from src.schema import (
    AGE_OPTIONS,
    EDUCATION_OPTIONS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    GENERAL_HEALTH_OPTIONS,
    INCOME_OPTIONS,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model_bundle() -> dict:
    """Load the complete pipeline and its evaluation metadata once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Run `python -m src.train_model` first."
        )
    return joblib.load(MODEL_PATH)


def yes_no(label: str, help_text: str | None = None, default: str = "No") -> int:
    """Render a consistent yes/no survey question."""
    options = ["No", "Yes"]
    return int(
        st.selectbox(
            label,
            options,
            index=options.index(default),
            help=help_text,
        )
        == "Yes"
    )


try:
    bundle = load_model_bundle()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

pipeline = bundle["pipeline"]

st.title("Diabetes Risk Screening")
st.write(
    "This educational tool uses answers similar to the CDC's 2015 BRFSS health "
    "survey to estimate whether a response pattern resembles the dataset's "
    "**prediabetes/diabetes group** or its **no-diabetes group**."
)
st.warning(
    "This is a screening demonstration, not a diagnosis or a prediction of your "
    "future health. Survey data cannot replace A1C, fasting-glucose, or other "
    "testing interpreted by a qualified healthcare professional."
)

with st.form("screening_form"):
    st.subheader("Health history")
    left, right = st.columns(2)

    with left:
        high_bp = yes_no("Have you been told you have high blood pressure?")
        high_chol = yes_no("Have you been told you have high cholesterol?")
        chol_check = yes_no(
            "Was your cholesterol checked within the past 5 years?", default="Yes"
        )
        stroke = yes_no("Have you ever been told you had a stroke?")
        heart_disease = yes_no(
            "Have you had coronary heart disease or a heart attack?"
        )
        diff_walk = yes_no(
            "Do you have serious difficulty walking or climbing stairs?"
        )

    with right:
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=12,
            max_value=98,
            value=25,
            step=1,
            help="The cleaned BRFSS dataset stores BMI as a whole number.",
        )
        general_health = GENERAL_HEALTH_OPTIONS[
            st.selectbox("How would you rate your general health?", GENERAL_HEALTH_OPTIONS)
        ]
        mental_health = st.number_input(
            "Poor mental-health days during the past 30 days",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
        )
        physical_health = st.number_input(
            "Poor physical-health days during the past 30 days",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
        )

    st.subheader("Lifestyle and access to care")
    left, right = st.columns(2)

    with left:
        smoker = yes_no("Have you smoked at least 100 cigarettes in your lifetime?")
        physical_activity = yes_no(
            "Did you do physical activity outside work in the past 30 days?",
            default="Yes",
        )
        fruits = yes_no("Do you eat fruit at least once per day?", default="Yes")
        vegetables = yes_no(
            "Do you eat vegetables at least once per day?", default="Yes"
        )

    with right:
        heavy_alcohol = yes_no(
            "Does your survey response meet the BRFSS heavy-alcohol threshold?"
        )
        healthcare = yes_no("Do you have healthcare coverage?", default="Yes")
        no_doctor_cost = yes_no(
            "In the past year, did cost prevent you from seeing a doctor?"
        )

    st.subheader("Demographic survey categories")
    left, right = st.columns(2)

    with left:
        age = AGE_OPTIONS[st.selectbox("Age group", AGE_OPTIONS)]
        education = EDUCATION_OPTIONS[
            st.selectbox("Highest education level", EDUCATION_OPTIONS, index=4)
        ]

    with right:
        income = INCOME_OPTIONS[
            st.selectbox("Annual household income", INCOME_OPTIONS, index=5)
        ]
        sex = int(
            st.selectbox(
                "Sex category recorded by the 2015 survey",
                ["Female", "Male"],
                help=(
                    "The cleaned historical dataset contains only this binary "
                    "coding. This is a limitation of the data, not a statement "
                    "about gender identity."
                ),
            )
            == "Male"
        )

    submitted = st.form_submit_button("Estimate screening signal", type="primary")

if submitted:
    answers = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": physical_activity,
        "Fruits": fruits,
        "Veggies": vegetables,
        "HvyAlcoholConsump": heavy_alcohol,
        "AnyHealthcare": healthcare,
        "NoDocbcCost": no_doctor_cost,
        "GenHlth": general_health,
        "MentHlth": mental_health,
        "PhysHlth": physical_health,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income,
    }
    input_row = pd.DataFrame([answers], columns=FEATURE_COLUMNS)
    probability = float(pipeline.predict_proba(input_row)[0, 1])
    band = risk_band(
        probability,
        bundle["low_threshold"],
        bundle["screening_threshold"],
    )

    st.divider()
    st.subheader("Result")
    st.metric("Model-estimated likelihood", f"{probability:.1%}")

    if band == "Elevated screening signal":
        st.error(band)
    elif band == "Moderate screening signal":
        st.warning(band)
    else:
        st.success(band)

    st.caption(
        "The bands are statistical operating points selected from training data; "
        "they are not clinical diagnostic cutoffs."
    )

    st.subheader("What most affected this model output?")
    explanations = explain_prediction(
        pipeline,
        input_row,
        bundle["reference_values"],
        top_n=5,
    )
    meaningful = [item for item in explanations if abs(item["effect"]) >= 0.001]
    if meaningful:
        for item in meaningful:
            direction = "raised" if item["effect"] > 0 else "lowered"
            st.write(
                f"- **{FEATURE_LABELS[item['feature']]}** {direction} the model "
                f"score by about **{abs(item['effect']):.1%}** compared with the "
                "typical training-set value for that one feature."
            )
    else:
        st.write(
            "No single answer changed the score substantially compared with the "
            "typical training profile."
        )

    st.caption(
        "These comparisons describe this model's behavior. They do not prove that "
        "a factor causes or prevents diabetes."
    )
    st.info(
        "If you are concerned about diabetes, discuss appropriate screening with "
        "a healthcare professional. Do not change medication or treatment based "
        "on this demonstration."
    )

with st.expander("How well did the model perform?"):
    metrics = bundle["test_metrics"]
    st.write(
        "These results come from the held-out test set, which was not used to "
        "choose the model or its screening threshold."
    )
    first, second, third = st.columns(3)
    first.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    second.metric("Recall", f"{metrics['recall']:.1%}")
    third.metric("Precision", f"{metrics['precision']:.1%}")

    st.write(
        f"**ROC-AUC:** {metrics['roc_auc']:.3f}  \n"
        f"**PR-AUC:** {metrics['pr_auc']:.3f}  \n"
        f"**Specificity:** {metrics['specificity']:.1%}  \n"
        f"**Brier score:** {metrics['brier_score']:.3f}"
    )
    matrix = metrics["confusion_matrix"]
    st.write(
        "**Confusion-matrix counts:** "
        f"{matrix['true_positive']:,} true positives, "
        f"{matrix['false_negative']:,} false negatives, "
        f"{matrix['true_negative']:,} true negatives, and "
        f"{matrix['false_positive']:,} false positives."
    )

with st.expander("Model and dataset details"):
    st.write(f"**Selected model:** {bundle['model_name']}")
    st.write(
        "**Dataset:** CDC Diabetes Health Indicators, cleaned BRFSS 2015 binary "
        "dataset (253,680 survey responses)."
    )
    st.write(
        "The positive target combines survey respondents labeled as having "
        "prediabetes or diabetes. The model finds associations in self-reported, "
        "cross-sectional survey data; it does not establish causes or predict "
        "future disease."
    )
