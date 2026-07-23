"""
gui.py

Streamlit-based graphical user interface for the QUBO feature selection
classification project.

Lets the user run the full pipeline interactively:
1. Select a dataset.
2. Run preprocessing.
3. Run QUBO-based feature selection.
4. Train a classifier.
5. Run predictions and view results.

Run with:
    streamlit run src/qubo_project/gui.py

This module only calls the existing pipeline functions (fit_normalize,
select_features, train, predict) and does not duplicate their logic, so it is
not required for the pipeline to work from the command line.
"""

import os

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
from qubo_project.model import train as model_train
from qubo_project.model import predict as model_predict

OUTPUTS_DIR = "outputs"


def _ensure_outputs_dir() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _init_session_state() -> None:
    defaults = {
        "dataset_path": None,
        "target_column": "",
        "preprocessing_done": False,
        "normalized_csv": None,
        "preprocessing_json": None,
        "feature_selection_done": False,
        "train_reduced_csv": None,
        "test_reduced_csv": None,
        "feature_selection_json": None,
        "optimizations_csv": None,
        "training_done": False,
        "model_path": None,
        "training_metrics_json": None,
        "prediction_done": False,
        "predictions_csv": None,
        "classification_stats_json": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _dataset_selection_section() -> None:
    st.header("1. Dataset selection")

    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    target_column = st.text_input(
        "Target column name", value=st.session_state["target_column"]
    )

    if uploaded_file is not None:
        _ensure_outputs_dir()
        dataset_path = os.path.join(OUTPUTS_DIR, "uploaded_dataset.csv")
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["dataset_path"] = dataset_path
        st.success(f"Dataset saved to: {dataset_path}")

    if target_column:
        st.session_state["target_column"] = target_column

    if st.session_state["dataset_path"]:
        st.info(f"Current dataset: {st.session_state['dataset_path']}")
        try:
            preview_df = pd.read_csv(st.session_state["dataset_path"], nrows=5)
            st.dataframe(preview_df)
        except Exception as e:
            st.error(f"Could not preview dataset: {e}")


def _preprocessing_section() -> None:
    st.header("2. Preprocessing")

    min_perc_valid = st.slider(
        "Minimum percentage of valid values to keep a column",
        min_value=0.0, max_value=1.0, value=0.05, step=0.01,
    )

    if st.button("Run preprocessing"):
        if not st.session_state["dataset_path"]:
            st.error("Please select a dataset first (step 1).")
        elif not st.session_state["target_column"]:
            st.error("Please provide the target column name (step 1).")
        else:
            _ensure_outputs_dir()
            normalized_csv = os.path.join(OUTPUTS_DIR, "normalized.csv")
            preprocessing_json = os.path.join(
                OUTPUTS_DIR, "preprocessing_result.json"
            )
            try:
                fit_normalize(
                    input_csv=st.session_state["dataset_path"],
                    target_column=st.session_state["target_column"],
                    normalized_csv=normalized_csv,
                    outInitalRes_json=preprocessing_json,
                    minPercValid=min_perc_valid,
                )
                st.session_state["normalized_csv"] = normalized_csv
                st.session_state["preprocessing_json"] = preprocessing_json
                st.session_state["preprocessing_done"] = True
                st.success("Preprocessing completed.")
            except Exception as e:
                st.session_state["preprocessing_done"] = False
                st.error(f"Preprocessing failed: {e}")

    if st.session_state["preprocessing_done"]:
        st.subheader("Preprocessing results")
        df_json = pd.read_json(st.session_state["preprocessing_json"], typ="series")
        st.json(df_json.to_dict())


def _feature_selection_section() -> None:
    st.header("3. Feature selection (QUBO)")

    col1, col2 = st.columns(2)
    with col1:
        perc_selected = st.slider(
            "Percentage of features to select", 0.01, 1.0, 0.20, 0.01
        )
        allowance = st.number_input("Allowance (tolerance, # features)", 0, 20, 1)
        perc_test = st.slider("Percentage of test set", 0.05, 0.95, 0.30, 0.05)
    with col2:
        seed = st.number_input("Random seed", 0, 10_000, 42)
        alpha_computations = st.number_input(
            "Max alpha search attempts", 1, 1000, 100
        )

    if st.button("Run feature selection"):
        if not st.session_state["preprocessing_done"]:
            st.error("Please run preprocessing first (step 2).")
        else:
            _ensure_outputs_dir()
            train_csv = os.path.join(OUTPUTS_DIR, "training_reduced.csv")
            test_csv = os.path.join(OUTPUTS_DIR, "test_reduced.csv")
            ottim_csv = os.path.join(OUTPUTS_DIR, "optimizations.csv")
            fs_json = os.path.join(OUTPUTS_DIR, "feature_selection_result.json")
            try:
                select_features(
                    normalized_csv=st.session_state["normalized_csv"],
                    reducedTrain_csv=train_csv,
                    reducedTest_csv=test_csv,
                    output_ottim_csv=ottim_csv,
                    output_json=fs_json,
                    target_column=st.session_state["target_column"],
                    percTest=perc_test,
                    percSelected=perc_selected,
                    allowance=int(allowance),
                    seed=int(seed),
                    alpha_computations=int(alpha_computations),
                )
                st.session_state["train_reduced_csv"] = train_csv
                st.session_state["test_reduced_csv"] = test_csv
                st.session_state["optimizations_csv"] = ottim_csv
                st.session_state["feature_selection_json"] = fs_json
                st.session_state["feature_selection_done"] = True
                st.success("Feature selection completed.")
            except Exception as e:
                st.session_state["feature_selection_done"] = False
                st.error(f"Feature selection failed: {e}")

    if st.session_state["feature_selection_done"]:
        st.subheader("Feature selection results")
        fs_result = pd.read_json(
            st.session_state["feature_selection_json"], typ="series"
        )
        st.json(fs_result.to_dict())

        st.subheader("Alpha search attempts")
        ottim_df = pd.read_csv(st.session_state["optimizations_csv"])
        st.line_chart(ottim_df.set_index("alpha")["n_selected_features"])
        st.dataframe(ottim_df)


def _training_section() -> None:
    st.header("4. Model training")

    classifier = st.selectbox(
        "Classifier", ["random_forest", "logistic_regression", "gradient_boosting"]
    )
    seed = st.number_input("Training random seed", 0, 10_000, 42, key="train_seed")

    if st.button("Run training"):
        if not st.session_state["feature_selection_done"]:
            st.error("Please run feature selection first (step 3).")
        else:
            _ensure_outputs_dir()
            model_path = os.path.join(OUTPUTS_DIR, "model.joblib")
            metrics_json = os.path.join(OUTPUTS_DIR, "training_metrics.json")
            try:
                model_train(
                    classifier=classifier,
                    reducedTrain_csv=st.session_state["train_reduced_csv"],
                    target_column=st.session_state["target_column"],
                    model_path=model_path,
                    metrics_json=metrics_json,
                    seed=int(seed),
                )
                st.session_state["model_path"] = model_path
                st.session_state["training_metrics_json"] = metrics_json
                st.session_state["training_done"] = True
                st.success("Training completed.")
            except Exception as e:
                st.session_state["training_done"] = False
                st.error(f"Training failed: {e}")

    if st.session_state["training_done"]:
        st.subheader("Training metrics")
        metrics = pd.read_json(
            st.session_state["training_metrics_json"], typ="series"
        )
        st.json(metrics.to_dict())


def _prediction_section() -> None:
    st.header("5. Prediction and evaluation")

    if st.button("Run prediction on test set"):
        if not st.session_state["training_done"]:
            st.error("Please run training first (step 4).")
        else:
            _ensure_outputs_dir()
            predictions_csv = os.path.join(OUTPUTS_DIR, "predictions.csv")
            stats_json = os.path.join(OUTPUTS_DIR, "classification_stats.json")
            try:
                model_predict(
                    reduced_Test_csv=st.session_state["test_reduced_csv"],
                    target_column=st.session_state["target_column"],
                    model_path=st.session_state["model_path"],
                    predictions_csv=predictions_csv,
                    classif_stats_json=stats_json,
                )
                st.session_state["predictions_csv"] = predictions_csv
                st.session_state["classification_stats_json"] = stats_json
                st.session_state["prediction_done"] = True
                st.success("Prediction completed.")
            except Exception as e:
                st.session_state["prediction_done"] = False
                st.error(f"Prediction failed: {e}")

    if st.session_state["prediction_done"]:
        st.subheader("Classification statistics")
        stats = pd.read_json(
            st.session_state["classification_stats_json"], typ="series"
        ).to_dict()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{stats['accuracy']:.4f}")
        col2.metric("ROC-AUC", f"{stats['roc_auc']:.4f}")
        col3.metric("Target=1 samples", stats["target_1_count"])

        st.write("Class 0 metrics:", stats["class_0"])
        st.write("Class 1 metrics:", stats["class_1"])

        st.write("Confusion matrix:")
        cm = stats["confusion_matrix"]["matrix"]
        cm_df = pd.DataFrame(
            cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
        )
        st.dataframe(cm_df)

        st.subheader("Predictions preview")
        predictions_df = pd.read_csv(st.session_state["predictions_csv"])
        st.dataframe(predictions_df.head(20))

        with open(st.session_state["predictions_csv"], "rb") as f:
            st.download_button(
                "Download full predictions CSV",
                data=f,
                file_name="predictions.csv",
                mime="text/csv",
            )


def main() -> None:
    st.set_page_config(page_title="QUBO Feature Selection Pipeline", layout="wide")
    st.title("QUBO-based Feature Selection - Classification Pipeline")

    _init_session_state()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Dataset",
            "2. Preprocessing",
            "3. Feature Selection",
            "4. Training",
            "5. Prediction",
        ]
    )

    with tab1:
        _dataset_selection_section()
    with tab2:
        _preprocessing_section()
    with tab3:
        _feature_selection_section()
    with tab4:
        _training_section()
    with tab5:
        _prediction_section()


if __name__ == "__main__":
    main()