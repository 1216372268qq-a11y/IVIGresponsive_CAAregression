"""CAA Regression Prediction Calculator | Final 7-predictor Cox model.

Run: streamlit run app.py
The model is NOT refitted. Its coefficients, baseline survival, and Cox
centering means were verified against the 408 patient-level R predictions.
"""

import json
import math
from pathlib import Path

import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model_parameters.json"


@st.cache_data
def load_model():
    with MODEL_PATH.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    expected = {"month", "BMI", "PLT", "ESR", "ALB", "Fib", "CAA_sizesMedium", "CAA_sizesGiant"}
    if set(model["coefficients"]) != expected:
        raise ValueError("Unexpected Cox model coefficient names")
    if set(model["baseline_survival"]) != {"12", "24", "36"}:
        raise ValueError("Missing baseline survival time points")
    return model


def predict_regression(model, age, bmi, plt, esr, alb, fib, severity):
    """P(complete CAA regression by t) = 1 - S_centered(t)^exp(LP_centered).

    Cox factor indicator columns were not mean-centered in R's coxph fit;
    continuous predictors were centered using coxph's fitted means.
    """
    beta = model["coefficients"]
    means = model["continuous_reference_means"]
    x = {"month": age, "BMI": bmi, "PLT": plt, "ESR": esr, "ALB": alb, "Fib": fib}
    lp = sum(beta[key] * (value - means[key]) for key, value in x.items())
    if severity == "Medium":
        lp += beta["CAA_sizesMedium"]
    elif severity == "Giant":
        lp += beta["CAA_sizesGiant"]
    elif severity != "Small":
        raise ValueError("Unexpected CAA severity")
    hazard_ratio = math.exp(lp)
    probabilities = {}
    for month in (12, 24, 36):
        s0 = model["baseline_survival"][str(month)]
        # -expm1 avoids precision loss when predicted regression is very low.
        probabilities[month] = -math.expm1(math.log(s0) * hazard_ratio)
    return probabilities


st.set_page_config(page_title="CAA Regression Prediction Calculator", page_icon="❤️", layout="centered")
model = load_model()

st.markdown(
    """
    <h1 style="
        font-size: 36px;
        font-weight: 700;
        white-space: nowrap;
        margin-bottom: 20px;
    ">
        CAA Regression Prediction Calculator
    </h1>
    """,
    unsafe_allow_html=True
)
st.caption("Final 7-predictor Cox model | IVIG-responsive Kawasaki disease with coronary artery aneurysms")
st.write(
    "Estimate the probability of **complete coronary artery aneurysm (CAA) regression** "
    "within 12, 24, and 36 months after Kawasaki disease onset."
)

with st.form("patient_data"):
    st.subheader("Clinical characteristics")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age at diagnosis (months)", min_value=0, max_value=240, value=36, step=1)
        plt = st.number_input("Platelet count, PLT (×10⁹/L)", min_value=0.0, value=350.0, step=10.0, format="%.1f")
        alb = st.number_input("Albumin, ALB (g/L)", min_value=0.0, value=38.0, step=0.5, format="%.1f")
    with right:
        bmi = st.number_input("Body mass index, BMI (kg/m²)", min_value=0.0, value=16.0, step=0.1, format="%.1f")
        esr = st.number_input("Erythrocyte sedimentation rate, ESR (mm/h)", min_value=0.0, value=40.0, step=1.0, format="%.1f")
        fib = st.number_input("Fibrinogen, Fib (g/L)", min_value=0.0, value=4.0, step=0.1, format="%.1f")

    severity_labels = {
        "Small (2.5 ≤ Z < 5)": "Small",
        "Medium (5 ≤ Z < 10)": "Medium",
        "Giant (Z ≥ 10)": "Giant",
    }
    selection = st.radio("CAA severity", options=list(severity_labels), horizontal=True)
    st.caption(
        "A giant aneurysm may also be classified by absolute coronary diameter ≥8 mm "
        "under commonly used clinical criteria."
    )
    submitted = st.form_submit_button("Calculate regression probability", type="primary", use_container_width=True)

if submitted:
    probabilities = predict_regression(model, age, bmi, plt, esr, alb, fib, severity_labels[selection])
    st.subheader("Predicted complete CAA regression")
    cols = st.columns(3)
    for col, month in zip(cols, (12, 24, 36)):
        col.metric(f"{month} months", f"{probabilities[month] * 100:.1f}%")
    st.info(
        "These percentages estimate the probability that **all coronary artery segments "
        "will have Z-scores <2** by the indicated time after disease onset. "
        "A higher percentage indicates a greater predicted probability of complete regression."
    )

st.divider()
st.caption(
    "This tool does not replace clinical judgment or scheduled echocardiographic follow-up."
)
