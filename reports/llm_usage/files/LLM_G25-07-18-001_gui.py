"""
Streamlit Graphical User Interface for the QUBO Feature Selection & Classification Pipeline.

This module provides an interactive web interface to run the full pipeline step-by-step:
1. Data Preprocessing & Normalization
2. QUBO Feature Selection
3. Model Training
4. Prediction & Evaluation
5. Output Files Inspector & Downloader
"""

import json
import os
import sys
from typing import List

import pandas as pd
import streamlit as st

# Ensure the 'src' directory is in Python path for relative imports within the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qubo_project.feature_selection import select_features
from qubo_project.model import predict, train
from qubo_project.preprocessing import fit_normalize

# Configuration Constants
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(
    page_title="QUBO Pipeline GUI",
    page_icon="🧬",
    layout="wide",
)


def _initialize_session_state() -> None:
    """Initialize Streamlit session state variables to track generated file paths."""
    defaults = {
        "input_csv": "",
        "target_column": "Target",
        "normalized_csv": os.path.join(OUTPUT_DIR, "normalized.csv"),
        "preproc_json": os.path.join(OUTPUT_DIR, "preproc_stats.json"),
        "reduced_train_csv": os.path.join(OUTPUT_DIR, "reduced_train.csv"),
        "reduced_test_csv": os.path.join(OUTPUT_DIR, "reduced_test.csv"),
        "optim_csv": os.path.join(OUTPUT_DIR, "optimization_results.csv"),
        "feat_select_json": os.path.join(OUTPUT_DIR, "feat_select_stats.json"),
        "model_path": os.path.join(OUTPUT_DIR, "trained_model.joblib"),
        "train_metrics_json": os.path.join(OUTPUT_DIR, "training_metrics.json"),
        "predictions_csv": os.path.join(OUTPUT_DIR, "predictions.csv"),
        "classif_stats_json": os.path.join(OUTPUT_DIR, "classification_stats.json"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_dataframe_columns(file_path: str) -> List[str]:
    """Helper function to safely extract column names from a CSV file."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, nrows=1)
            return list(df.columns)
        except Exception:
            return []
    return []


def main() -> None:
    """Main execution function for the Streamlit GUI application."""
    _initialize_session_state()

    st.title("🧬 QUBO Credit Risk Pipeline")
    st.markdown(
        "Interactive graphical user interface for data normalization, QUBO feature reduction, "
        "model training, and binary classification evaluation."
    )

    # Main Navigation via Tabs
    tab_preproc, tab_feat, tab_train, tab_pred, tab_out = st.tabs(
        [
            "1. Preprocessing",
            "2. Feature Selection",
            "3. Training",
            "4. Prediction",
            "5. Output Files",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: PREPROCESSING
    # -------------------------------------------------------------------------
    with tab_preproc:
        st.header("Step 1: Data Preprocessing & Normalization")
        st.write("Clean the dataset by dropping missing features and applying Z-score normalization.")

        uploaded_file = st.file_uploader("Upload Input Dataset (CSV)", type=["csv"])

        if uploaded_file is not None:
            temp_input_path = os.path.join(OUTPUT_DIR, "uploaded_dataset.csv")
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["input_csv"] = temp_input_path

        if st.session_state["input_csv"] and os.path.exists(st.session_state["input_csv"]):
            cols = _get_dataframe_columns(st.session_state["input_csv"])
            st.success(f"File loaded: `{st.session_state['input_csv']}`")

            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox(
                    "Target Column Name",
                    options=cols,
                    index=len(cols) - 1 if cols else 0,
                )
                st.session_state["target_column"] = target_col

            with col2:
                min_perc_valid = st.slider(
                    "Minimum Valid Percentage Threshold (minPercValid)",
                    min_value=0.01,
                    max_value=1.00,
                    value=0.05,
                    step=0.01,
                    help="Columns with valid values below this fraction will be dropped.",
                )

            if st.button("🚀 Run Preprocessing", key="btn_preproc"):
                try:
                    fit_normalize(
                        input_csv=st.session_state["input_csv"],
                        target_column=st.session_state["target_column"],
                        normalized_csv=st.session_state["normalized_csv"],
                        outInitalRes_json=st.session_state["preproc_json"],
                        minPercValid=min_perc_valid,
                    )
                    st.success("Preprocessing completed successfully!")

                    if os.path.exists(st.session_state["preproc_json"]):
                        with open(st.session_state["preproc_json"], "r", encoding="utf-8") as f:
                            stats = json.load(f)
                        st.subheader("Preprocessing Statistics")
                        st.json(stats)

                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
        else:
            st.info("Please upload a CSV file to begin.")

    # -------------------------------------------------------------------------
    # TAB 2: FEATURE SELECTION
    # -------------------------------------------------------------------------
    with tab_feat:
        st.header("Step 2: QUBO Feature Selection")
        st.write("Perform feature reduction using Spearman correlation-based QUBO optimization.")

        if not os.path.exists(st.session_state["normalized_csv"]):
            st.warning("Please complete Step 1 (Preprocessing) first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                perc_test = st.slider("Test Set Split Ratio (percTest)", 0.10, 0.50, 0.30, 0.05)
                perc_selected = st.slider("Target Selected Features Fraction (percSelected)", 0.05, 0.90, 0.20, 0.05)
                allowance = st.number_input("Selection Allowance (+/- features)", min_value=0, max_value=10, value=1)

            with col2:
                seed = st.number_input("Random Seed", value=42)
                alpha_computations = st.number_input("Alpha Optimization Steps", min_value=10, max_value=1000, value=100)

            if st.button("🚀 Run Feature Selection", key="btn_feat"):
                try:
                    select_features(
                        normalized_csv=st.session_state["normalized_csv"],
                        reducedTrain_csv=st.session_state["reduced_train_csv"],
                        reducedTest_csv=st.session_state["reduced_test_csv"],
                        output_ottim_csv=st.session_state["optim_csv"],
                        output_json=st.session_state["feat_select_json"],
                        target_column=st.session_state["target_column"],
                        percTest=perc_test,
                        percSelected=perc_selected,
                        allowance=allowance,
                        seed=seed,
                        alpha_computations=alpha_computations,
                    )
                    st.success("Feature selection completed successfully!")

                    if os.path.exists(st.session_state["feat_select_json"]):
                        with open(st.session_state["feat_select_json"], "r", encoding="utf-8") as f:
                            stats = json.load(f)
                        st.subheader("Feature Selection Statistics")
                        st.json(stats)

                except Exception as e:
                    st.error(f"Error during feature selection: {e}")

    # -------------------------------------------------------------------------
    # TAB 3: MODEL TRAINING
    # -------------------------------------------------------------------------
    with tab_train:
        st.header("Step 3: Classifier Training")
        st.write("Train a binary classification model on the QUBO-reduced training dataset.")

        if not os.path.exists(st.session_state["reduced_train_csv"]):
            st.warning("Please complete Step 2 (Feature Selection) first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                classifier_choice = st.selectbox(
                    "Classifier Algorithm",
                    options=["random_forest", "logistic_regression", "gradient_boosting"],
                    format_func=lambda x: x.replace("_", " ").title(),
                )
            with col2:
                train_seed = st.number_input("Training Seed", value=42, key="train_seed")

            if st.button("🚀 Train Model", key="btn_train"):
                try:
                    train(
                        classifier=classifier_choice,
                        reducedTrain_csv=st.session_state["reduced_train_csv"],
                        target_column=st.session_state["target_column"],
                        model_path=st.session_state["model_path"],
                        metrics_json=st.session_state["train_metrics_json"],
                        seed=train_seed,
                    )
                    st.success(f"Model saved to `{st.session_state['model_path']}`")

                    if os.path.exists(st.session_state["train_metrics_json"]):
                        with open(st.session_state["train_metrics_json"], "r", encoding="utf-8") as f:
                            metrics = json.load(f)
                        st.subheader("Training Metrics")
                        st.json(metrics)

                except Exception as e:
                    st.error(f"Error during model training: {e}")

    # -------------------------------------------------------------------------
    # TAB 4: PREDICTION & EVALUATION
    # -------------------------------------------------------------------------
    with tab_pred:
        st.header("Step 4: Prediction & Evaluation")
        st.write("Run predictions on the reduced test set and evaluate classification performance.")

        if not os.path.exists(st.session_state["reduced_test_csv"]) or not os.path.exists(st.session_state["model_path"]):
            st.warning("Please ensure both Step 2 (Feature Selection) and Step 3 (Training) are completed.")
        else:
            if st.button("🚀 Run Predictions", key="btn_pred"):
                try:
                    predict(
                        reduced_Test_csv=st.session_state["reduced_test_csv"],
                        target_column=st.session_state["target_column"],
                        model_path=st.session_state["model_path"],
                        predictions_csv=st.session_state["predictions_csv"],
                        classif_stats_json=st.session_state["classif_stats_json"],
                    )
                    st.success("Predictions and evaluation finished successfully!")

                    if os.path.exists(st.session_state["classif_stats_json"]):
                        with open(st.session_state["classif_stats_json"], "r", encoding="utf-8") as f:
                            stats = json.load(f)

                        # Top Performance Metrics
                        st.subheader("Overview Metrics")
                        m1, m2 = st.columns(2)
                        m1.metric("Accuracy", f"{stats.get('accuracy', 0.0) * 100:.2f}%")
                        roc = stats.get("roc_auc")
                        m2.metric("ROC-AUC", f"{roc:.4f}" if roc is not None else "N/A")

                        # Per-Class Metrics Table
                        st.subheader("Per-Class Classification Report")
                        metrics_df = pd.DataFrame(
                            {
                                "Precision": [stats["class_0"]["precision"], stats["class_1"]["precision"]],
                                "Recall": [stats["class_0"]["recall"], stats["class_1"]["recall"]],
                                "F1-Score": [stats["class_0"]["f1"], stats["class_1"]["f1"]],
                                "Support": [stats["class_0"]["support"], stats["class_1"]["support"]],
                            },
                            index=["Class 0 (Reliable)", "Class 1 (Risky)"],
                        )
                        st.dataframe(metrics_df, use_container_width=True)

                        # Confusion Matrix Table
                        st.subheader("Confusion Matrix")
                        cm = stats["confusion_matrix"]["matrix"]
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual: 0", "Actual: 1"],
                            columns=["Predicted: 0", "Predicted: 1"],
                        )
                        st.table(cm_df)

                except Exception as e:
                    st.error(f"Error during predictions: {e}")

    # -------------------------------------------------------------------------
    # TAB 5: OUTPUT FILES INSPECTOR & DOWNLOADER
    # -------------------------------------------------------------------------
    with tab_out:
        st.header("Step 5: Output Files & Downloads")
        st.write("Inspect and download the generated dataset CSVs, JSON reports, and model files.")

        files_to_check = {
            "Normalized Dataset": st.session_state["normalized_csv"],
            "Reduced Train Dataset": st.session_state["reduced_train_csv"],
            "Reduced Test Dataset": st.session_state["reduced_test_csv"],
            "Optimization Results": st.session_state["optim_csv"],
            "Predictions CSV": st.session_state["predictions_csv"],
            "Classification Stats JSON": st.session_state["classif_stats_json"],
            "Training Metrics JSON": st.session_state["train_metrics_json"],
            "Trained Model Joblib": st.session_state["model_path"],
        }

        for label, file_path in files_to_check.items():
            st.write("---")
            c1, c2, c3 = st.columns([3, 2, 2])
            c1.markdown(f"**{label}** (`{file_path}`)")

            if os.path.exists(file_path):
                c2.success("Available")
                with open(file_path, "rb") as f:
                    c3.download_button(
                        label=f"Download {os.path.basename(file_path)}",
                        data=f.read(),
                        file_name=os.path.basename(file_path),
                        key=f"dl_{label}",
                    )
            else:
                c2.error("Not Generated Yet")


if __name__ == "__main__":
    main()