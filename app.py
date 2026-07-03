"""Streamlit demo: interactive breast cancer diagnosis with a clinically
appropriate decision threshold and original-feature-space explainability.

Loads pipelines and test data already produced by notebooks/exploration.ipynb
(see models/) rather than retraining live.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bca.data import load_data
from bca.explain import shap_explanation
from bca.metrics import confusion_counts, sensitivity, specificity

MODELS_DIR = Path(__file__).parent / "models"

# The five features with the strongest, most clinically legible signal for
# malignancy in this dataset — kept to five so the sidebar stays usable.
FEATURE_SLIDERS = [
    "worst radius",
    "worst concave points",
    "mean concave points",
    "worst perimeter",
    "worst area",
]


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "random_forest.joblib")
    thresholds = json.loads((MODELS_DIR / "thresholds.json").read_text())
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")
    y_test = pd.read_csv(MODELS_DIR / "y_test.csv").squeeze("columns")
    X_background = pd.read_csv(MODELS_DIR / "shap_background.csv")
    return pipeline, thresholds, X_test, y_test, X_background


@st.cache_data
def dataset_reference():
    X, y = load_data()
    return X.median(), X.min(), X.max()


def main():
    st.set_page_config(page_title="Breast Cancer Diagnostic Audit", layout="wide")
    st.title("Breast Cancer Diagnostic Classifier")
    st.caption(
        "A leakage-free, clinically-labeled random forest classifier — see the "
        "README for the metric-attribution bug this project was built to audit."
    )

    if not (MODELS_DIR / "random_forest.joblib").exists():
        st.error(
            "No trained model found in models/. Run notebooks/exploration.ipynb "
            "once to generate models/random_forest.joblib and friends."
        )
        return

    pipeline, thresholds, X_test, y_test, X_background = load_artifacts()
    medians, mins, maxs = dataset_reference()
    tuned_threshold = thresholds["random_forest"]

    st.sidebar.header("Patient measurements")
    st.sidebar.caption("Sliders default to the dataset median for each feature.")
    inputs = {}
    for feature in FEATURE_SLIDERS:
        inputs[feature] = st.sidebar.slider(
            feature,
            min_value=float(mins[feature]),
            max_value=float(maxs[feature]),
            value=float(medians[feature]),
        )

    threshold = st.sidebar.slider(
        "Decision threshold (predicted malignancy probability)",
        min_value=0.0,
        max_value=1.0,
        value=float(tuned_threshold),
        step=0.01,
        help=(
            "Defaults to the cost-sensitive threshold tuned in the notebook "
            "(maximizes sensitivity subject to a 90% precision floor)."
        ),
    )

    # Build a full feature row: slider features overridden, all others held
    # at the dataset median so the model always receives a complete input.
    row = medians.copy()
    for feature, value in inputs.items():
        row[feature] = value
    X_input = pd.DataFrame([row])[X_test.columns]

    proba_malignant = pipeline.predict_proba(X_input)[0, 1]
    prediction = "Malignant" if proba_malignant >= threshold else "Benign"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted probability of malignancy", f"{proba_malignant:.1%}")
        st.metric("Classification at chosen threshold", prediction)

    with col2:
        st.subheader("Why the model said this (SHAP)")
        explanation = shap_explanation(pipeline, X_background=X_background, X_explain=X_input)
        contrib = pd.Series(explanation.values[0], index=X_input.columns).sort_values(
            key=np.abs, ascending=True
        ).tail(8)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in contrib.values]
        ax.barh(contrib.index, contrib.values, color=colors)
        ax.set_xlabel("SHAP value (pushes toward malignant → / ← benign)")
        st.pyplot(fig)

    st.divider()
    st.subheader(f"Test-set performance at threshold = {threshold:.2f}")
    y_score_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_score_test >= threshold).astype(int)

    counts = confusion_counts(y_test, y_pred_test)
    sens = sensitivity(y_test, y_pred_test)
    spec = specificity(y_test, y_pred_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensitivity (recall on malignant)", f"{sens:.1%}")
    m2.metric("Specificity (recall on benign)", f"{spec:.1%}")
    m3.metric("False negatives", counts["fn"])
    m4.metric("False positives", counts["fp"])

    cm_df = pd.DataFrame(
        [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
        index=["Actual: Benign", "Actual: Malignant"],
        columns=["Predicted: Benign", "Predicted: Malignant"],
    )
    st.dataframe(cm_df)


if __name__ == "__main__":
    main()
