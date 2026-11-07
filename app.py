"""
Bank Marketing Prediction - Binary Classification Dashboard
BITS Pilani — M.Tech (AIML) | Machine Learning Assignment 2
GitHub Repository: https://github.com/msarathrahul/bank_marketing.git

Problem Statement:
Direct telemarketing campaigns remain one of the most proactive strategies used by banking
institutions to sell long term term deposit products. However, indiscriminate cold calling is
resource intensive and often yields low conversion rates while causing customer fatigue.

The objective of this assignment is to develop and evaluate multiple supervised machine learning
classification models to predict whether a banking client will subscribe to a term deposit
(y = "yes" or "no"). By leveraging historical customer demographics, financial indicators, and
past campaign interaction features, the bank can optimize campaign targeting, maximize deposit
conversions, and minimize operational costs.

Reference: ML_Assignment_2_AIML_26Oct26.pdf
Features implemented:
a. Dataset upload option (CSV) [with fallback to bundled test_data.csv]
b. Model selection dropdown (5 ML models + All Models Comparison)
c. Display of 6 evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
d. Confusion matrix and classification report
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Prediction - Binary Classification Dashboard",
    page_icon="🏦",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. Paths & Model Loading
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
DEFAULT_TEST_CSV = os.path.join(CURRENT_DIR, "test_data.csv")

MODEL_CONFIGS = {
    "Logistic Regression": "model_logistic_regression.joblib",
    "Decision Tree": "model_decision_tree.joblib",
    "KNN": "model_knn.joblib",
    "Naive Bayes": "model_naive_bayes.joblib",
    "Support Vector Machine": "model_svm.joblib",
}


@st.cache_resource
def load_models_and_preprocessor():
    """Load the pre-trained models and preprocessing pipeline."""
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor artifact not found at: {preprocessor_path}")
        st.stop()

    preprocessor = joblib.load(preprocessor_path)
    models = {}
    for name, filename in MODEL_CONFIGS.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
        else:
            st.warning(f"Warning: Model artifact for '{name}' not found at {model_path}.")
    return models, preprocessor


models_dict, preprocessor = load_models_and_preprocessor()

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------
def parse_target(series: pd.Series) -> np.ndarray:
    """Standardize target column to binary integers (0 or 1)."""
    series_str = series.astype(str).str.strip().str.lower()
    if set(series_str.unique()).issubset({"yes", "no", "y", "n", "true", "false", "0", "1"}):
        return series_str.isin(["yes", "y", "true", "1"]).astype(int).values
    try:
        return series.astype(int).values
    except Exception:
        return (series_str == "yes").astype(int).values


@st.cache_data
def evaluate_model(model_name: str, _model, _preprocessor, df: pd.DataFrame, target_col: str = "y"):
    """Evaluate a trained model on the provided test dataset."""
    X = df.drop(columns=[target_col])
    y_true = parse_target(df[target_col])

    # Transform features with preprocessor pipeline
    X_proc = _preprocessor.transform(X)

    # Predictions
    y_pred = _model.predict(X_proc)

    # Probabilities for AUC
    if hasattr(_model, "predict_proba"):
        y_proba = _model.predict_proba(X_proc)[:, 1]
    elif hasattr(_model, "decision_function"):
        y_proba = _model.decision_function(X_proc)
    else:
        y_proba = y_pred

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["No", "Yes"], output_dict=True)
    return metrics, cm, cr


def predict_single_instance(_model, _preprocessor, input_df: pd.DataFrame):
    """Run inference for a single customer sample and return predicted class and probability."""
    X_proc = _preprocessor.transform(input_df)
    pred = int(_model.predict(X_proc)[0])
    proba = None
    if hasattr(_model, "predict_proba"):
        proba = float(_model.predict_proba(X_proc)[0, 1])
    elif hasattr(_model, "decision_function"):
        decision = float(_model.decision_function(X_proc)[0])
        proba = 1.0 / (1.0 + np.exp(-decision))
    return pred, proba


# -----------------------------------------------------------------------------
# 4. Sidebar Controls (Upload & Model Dropdown)
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

# Feature a: Dataset Upload Option (CSV)
st.sidebar.subheader("1. Test Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"],
    help="Upload a test CSV containing the 16 feature columns and target 'y'.",
)

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        data_source_label = f"Uploaded file: `{uploaded_file.name}`"
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
else:
    if os.path.exists(DEFAULT_TEST_CSV):
        test_df = pd.read_csv(DEFAULT_TEST_CSV)
        data_source_label = "Bundled benchmark test data (`test_data.csv`)"
    else:
        st.sidebar.error("Default test_data.csv not found. Please upload a CSV.")
        st.stop()

# Feature: Add a Single Data Point for Inference (under the upload test data)
with st.sidebar.expander("➕ Single Data Point for Inference", expanded=False):
    st.caption("Provide client attributes to run real-time model prediction.")
    with st.form("single_point_inference_form"):
        sp_age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        sp_job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
            ],
            index=4,
        )
        sp_marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"], index=0)
        sp_education = st.selectbox("Education", ["tertiary", "secondary", "primary", "unknown"], index=0)
        sp_default = st.selectbox("Credit Default?", ["no", "yes", "unknown"], index=0)
        sp_balance = st.number_input("Yearly Balance (€)", value=1000, step=100)
        sp_housing = st.selectbox("Housing Loan?", ["no", "yes", "unknown"], index=0)
        sp_loan = st.selectbox("Personal Loan?", ["no", "yes", "unknown"], index=0)
        sp_contact = st.selectbox("Contact Communication", ["cellular", "telephone", "unknown"], index=0)
        sp_month = st.selectbox(
            "Last Contact Month",
            ["may", "jun", "jul", "aug", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "sep"],
            index=0,
        )
        sp_day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15, step=1)
        sp_duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=250, step=10)
        sp_campaign = st.number_input("Campaign Contacts", min_value=1, max_value=50, value=1, step=1)
        sp_pdays = st.number_input("Pdays (-1 for not contacted)", min_value=-1, max_value=1000, value=-1, step=1)
        sp_previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0, step=1)
        sp_poutcome = st.selectbox("Previous Outcome", ["failure", "other", "success", "unknown"], index=1)

        btn_predict_single = st.form_submit_button("🔮 Predict Single Data Point", use_container_width=True)

    if btn_predict_single:
        single_row = {
            "age": sp_age,
            "job": sp_job,
            "marital": sp_marital,
            "education": sp_education,
            "default": sp_default,
            "balance": sp_balance,
            "housing": sp_housing,
            "loan": sp_loan,
            "contact": sp_contact,
            "day_of_week": sp_day,
            "month": sp_month,
            "duration": sp_duration,
            "campaign": sp_campaign,
            "pdays": sp_pdays,
            "previous": sp_previous,
            "poutcome": sp_poutcome,
        }
        st.session_state["single_sample"] = pd.DataFrame([single_row])
        st.session_state["single_prediction_active"] = True

# Feature b: Model Selection Dropdown
st.sidebar.subheader("2. Model Selection")
model_options = list(MODEL_CONFIGS.keys()) + ["All Models (Comparison)"]
selected_option = st.sidebar.selectbox("Select Model", options=model_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Info**
    - **Course Name:** Machine Learning
    - **Course Code:** AIMLZG565
    - **Assignment:** Assignment 2
    - **Name:** Sarath Rahul Malla
    - **Student ID:** 2026AJ05622
    - **GitHub:** [Repository](https://github.com/msarathrahul/bank_marketing.git)
    """
)

# -----------------------------------------------------------------------------
# 5. Main App UI Header & Dataset Overview
# -----------------------------------------------------------------------------
st.title("🏦 Bank Marketing Prediction - Binary Classification Dashboard")
st.markdown(
    "Interactive evaluation dashboard for machine learning classification models "
    "predicting whether a client subscribes to a bank term deposit (`y = yes / no`)."
)

# Problem statement expander
with st.expander("📋 Problem Statement", expanded=True):
    st.markdown(
        """
        Direct telemarketing campaigns remain one of the most proactive strategies used by banking institutions to sell long term term deposit products. However, indiscriminate cold calling is resource intensive and often yields low conversion rates while causing customer fatigue.

        The objective of this assignment is to develop and evaluate multiple supervised machine learning classification models to predict whether a banking client will subscribe to a term deposit (`y = "yes"` or `"no"`). By leveraging historical customer demographics, financial indicators, and past campaign interaction features, the bank can optimize campaign targeting, maximize deposit conversions, and minimize operational costs.
        """
    )

# Dataset details expander
with st.expander("ℹ️ Dataset & Citation Information"):
    st.markdown(
        """
        - **Dataset:** Bank Marketing (with social/economic context)
        - **Citation:** S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014. [DOI: 10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001)
        - **Classification Goal:** Predict if the client will subscribe a bank term deposit (`y`).
        - **Data Source in Use:** """ + data_source_label + """
        """
    )

# Validate dataset columns
target_candidates = ["y", "deposit", "target"]
target_col = next((c for c in target_candidates if c in test_df.columns), None)

if target_col is None:
    st.error(
        f"Target column 'y' not found in uploaded dataset. Available columns: {list(test_df.columns)}"
    )
    st.stop()

# Display brief dataset stats
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
y_parsed = parse_target(test_df[target_col])
total_samples = len(test_df)
pos_samples = int((y_parsed == 1).sum())
neg_samples = int((y_parsed == 0).sum())

col_stat1.metric("Total Test Samples", f"{total_samples:,}")
col_stat2.metric("Features", f"{test_df.shape[1] - 1}")
col_stat3.metric("Class 'No' (Negative)", f"{neg_samples:,} ({neg_samples / total_samples * 100:.1f}%)")
col_stat4.metric("Class 'Yes' (Positive)", f"{pos_samples:,} ({pos_samples / total_samples * 100:.1f}%)")

with st.expander("🔍 View Test Dataset Sample (First 5 Rows)"):
    st.dataframe(test_df.head(5), use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. Single Data Point Inference View (if active)
# -----------------------------------------------------------------------------
if st.session_state.get("single_prediction_active") and "single_sample" in st.session_state:
    sample_df = st.session_state["single_sample"]
    st.subheader("🔮 Single Data Point Inference Result")

    if selected_option != "All Models (Comparison)":
        model = models_dict[selected_option]
        pred, proba = predict_single_instance(model, preprocessor, sample_df)
        pred_label = "Deposit Subscribed (Yes)" if pred == 1 else "Deposit Not Subscribed (No)"

        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            if pred == 1:
                st.success(f"**Predicted Outcome ({selected_option}):** {pred_label}")
            else:
                st.info(f"**Predicted Outcome ({selected_option}):** {pred_label}")
            st.metric(
                label=f"{selected_option} Prediction",
                value="Yes (Subscribed)" if pred == 1 else "No (Not Subscribed)",
                delta=f"Prob: {proba * 100:.1f}%" if proba is not None else None,
            )
        with col_res2:
            if proba is not None:
                st.markdown(f"**Probability of Subscription (`y = yes`):** `{proba:.4f}` ({proba * 100:.1f}%)")
                st.progress(float(np.clip(proba, 0.0, 1.0)))
            else:
                st.caption("Probability score not available for this model configuration.")
    else:
        st.info("Single data point prediction across **All 5 Machine Learning Models**:")

    st.markdown("##### Predictions Across All 5 Models for this Data Point:")
    summary_rows = []
    for m_name, m_obj in models_dict.items():
        p, pr = predict_single_instance(m_obj, preprocessor, sample_df)
        summary_rows.append({
            "ML Model Name": m_name,
            "Predicted Outcome": "Yes (Subscribed)" if p == 1 else "No (Not Subscribed)",
            "Probability (Yes)": f"{pr:.4f}" if pr is not None else "N/A",
            "Confidence": f"{pr * 100:.1f}%" if pr is not None else "N/A",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    with st.expander("🔍 View Input Attributes for this Data Point", expanded=False):
        st.dataframe(sample_df, use_container_width=True)

    if st.button("❌ Dismiss Single Data Point Result", key="dismiss_single_point"):
        st.session_state["single_prediction_active"] = False
        st.rerun()

    st.markdown("---")

# -----------------------------------------------------------------------------
# 7. Evaluation Views
# -----------------------------------------------------------------------------

# Case A: Single Model Selected
if selected_option != "All Models (Comparison)":
    st.subheader(f"📊 Evaluation Results: {selected_option}")

    model = models_dict[selected_option]
    with st.spinner(f"Evaluating {selected_option} on test data..."):
        metrics, cm, cr = evaluate_model(selected_option, model, preprocessor, test_df, target_col=target_col)

    # Feature c: Display of 6 evaluation metrics
    st.markdown("#### 🎯 Evaluation Metrics")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    m_col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
    m_col3.metric("Precision", f"{metrics['Precision']:.4f}")
    m_col4.metric("Recall", f"{metrics['Recall']:.4f}")
    m_col5.metric("F1 Score", f"{metrics['F1']:.4f}")
    m_col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

    st.markdown("---")

    # Feature d: Confusion matrix or classification report
    st.markdown("#### 🔍 Diagnostic Visualizations & Report")
    vis_col1, vis_col2 = st.columns([1, 1])

    with vis_col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"],
            ax=ax,
        )
        ax.set_title(f"{selected_option} Confusion Matrix", fontsize=11, pad=10)
        st.pyplot(fig)
        plt.close(fig)

    with vis_col2:
        st.markdown("**Classification Report**")
        cr_df = pd.DataFrame(cr).transpose()
        # Clean formatting for dataframe
        st.dataframe(
            cr_df.style.format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

# Case B: All Models Comparison
else:
    st.subheader("📋 All Models Comparison Table & Performance Summary")
    st.caption("Evaluation of all 5 classifiers implemented on the same dataset.")

    comparison_records = []
    all_metrics = {}

    with st.spinner("Evaluating all 5 models on test dataset..."):
        for model_name, model in models_dict.items():
            m, _, _ = evaluate_model(model_name, model, preprocessor, test_df, target_col=target_col)
            all_metrics[model_name] = m
            comparison_records.append(
                {
                    "ML Model Name": model_name,
                    "Accuracy": round(m["Accuracy"], 4),
                    "AUC": round(m["AUC"], 4),
                    "Precision": round(m["Precision"], 4),
                    "Recall": round(m["Recall"], 4),
                    "F1": round(m["F1"], 4),
                    "MCC": round(m["MCC"], 4),
                }
            )

    comp_df = pd.DataFrame(comparison_records)

    # Comparison Table required by Step 5 of Assignment PDF
    st.markdown("#### Comparison Table")
    st.dataframe(
        comp_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ).highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#d1fae5",
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Multi-metric comparative bar chart
    st.markdown("#### 📊 Comparative Metric Visualization")
    chart_df = comp_df.set_index("ML Model Name")
    st.bar_chart(chart_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]])

    st.markdown("---")

    # Model Observations & Overall Winner (Step 5 of Assignment PDF)
    st.markdown("#### 📝 Model Performance Observations & Winner")
    observations_data = [
        {
            "ML Model Name": "Logistic Regression",
            "Observation about model performance": "Strong linear baseline with high Recall (0.8299) and competitive AUC (0.9056). Captures potential subscribers well under class imbalance.",
        },
        {
            "ML Model Name": "Decision Tree",
            "Observation about model performance": "Captures non-linear decision thresholds with fast inference; achieved 0.8403 Accuracy and 0.5392 F1 Score, slightly lower AUC than SVM.",
        },
        {
            "ML Model Name": "KNN",
            "Observation about model performance": "Highest Precision (0.5746) and highest Accuracy (0.8943), but lower Recall (0.3715) due to feature sparsity in high-dimensional one-hot encoded space.",
        },
        {
            "ML Model Name": "Naive Bayes",
            "Observation about model performance": "Fastest training and inference with decent AUC (0.8441), but conditional feature independence assumption limits Precision and F1 (0.4878).",
        },
        {
            "ML Model Name": "Support Vector Machine",
            "Observation about model performance": "Best overall classifier. Achieved the highest AUC (0.9230), highest Recall (0.8800+), and highest F1 Score (0.5597) with robust margin separation.",
        },
        {
            "ML Model Name": "Overall Winner",
            "Observation about model performance": "Support Vector Machine (SVM) — Superior discriminative power under class imbalance with top AUC (0.9230) and F1 Score (0.5597).",
        },
    ]
    obs_df = pd.DataFrame(observations_data)
    st.table(obs_df)
