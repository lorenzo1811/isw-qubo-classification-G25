"""
Streamlit Graphical User Interface for the QUBO Feature Selection & Classification Pipeline.
"""

import json
import os
import sys
from typing import List

import pandas as pd
import streamlit as st

# Assicura che la directory 'src' sia nel PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qubo_project.feature_selection import select_features
from qubo_project.model import predict, train
from qubo_project.preprocessing import fit_normalize

# Configurazione directory di output
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(
    page_title="QUBO Pipeline GUI",
    page_icon="🧬",
    layout="wide",
)

def _initialize_session_state() -> None:
    """Inizializza le variabili di stato per tracciare i file generati."""
    defaults = {
        "input_csv": "",
        "target_column": "Target",
        "normalized_csv": os.path.join(OUTPUT_DIR, "normalized_dataset.csv"),
        "preproc_json": os.path.join(OUTPUT_DIR, "preprocessing_stats.json"),
        "reduced_train_csv": os.path.join(OUTPUT_DIR, "train_dataset.csv"),
        "reduced_test_csv": os.path.join(OUTPUT_DIR, "test_dataset.csv"),
        "ottim_csv": os.path.join(OUTPUT_DIR, "qubo_optimization_results.csv"),
        "selection_json": os.path.join(OUTPUT_DIR, "feature_selection_stats.json"),
        "model_path": os.path.join(OUTPUT_DIR, "trained_model.joblib"),
        "train_metrics_json": os.path.join(OUTPUT_DIR, "training_metrics.json"),
        "predictions_csv": os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        "classification_stats_json": os.path.join(OUTPUT_DIR, "classification_stats.json"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_initialize_session_state()

st.title("🧬 Pipeline di Classificazione con Ottimizzazione QUBO")
st.markdown("Carica il tuo dataset, imposta i parametri ed esegui l'intera pipeline step-by-step.")

# --- SIDEBAR: Caricamento e Parametri Globali ---
st.sidebar.header("⚙️ Parametri di Configurazione")
uploaded_file = st.sidebar.file_uploader("Carica il Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    temp_input_path = os.path.join(OUTPUT_DIR, "uploaded_dataset.csv")
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["input_csv"] = temp_input_path
    st.sidebar.success("Dataset caricato con successo!")

st.session_state["target_column"] = st.sidebar.text_input("Nome Colonna Target", value=st.session_state["target_column"])

# --- TAB PRINCIPALI ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Preprocessing", 
    "2. Feature Selection", 
    "3. Training", 
    "4. Prediction", 
    "5. Output Files"
])

# --- TAB 1: PREPROCESSING ---
with tab1:
    st.header("1. Preprocessing & Normalizzazione")
    min_perc_valid = st.slider("Soglia Minima Valori Validi (minPercValid)", 0.0, 1.0, 0.8, 0.05)
    
    if st.button("Run Preprocessing"):
        if not st.session_state["input_csv"]:
            st.error("Seleziona prima un file CSV dalla sidebar!")
        else:
            try:
                fit_normalize(
                    input_csv=st.session_state["input_csv"],
                    target_column=st.session_state["target_column"],
                    normalized_csv=st.session_state["normalized_csv"],
                    outInitalRes_json=st.session_state["preproc_json"],
                    minPercValid=min_perc_valid
                )
                st.success("Preprocessing completato con successo!")
                
                if os.path.exists(st.session_state["preproc_json"]):
                    with open(st.session_state["preproc_json"], "r") as f:
                        stats = json.load(f)
                    st.json(stats)
            except Exception as e:
                st.error(f"Errore durante il preprocessing: {e}")

# --- TAB 2: FEATURE SELECTION ---
with tab2:
    st.header("2. Selezione Feature tramite QUBO")
    col1, col2 = st.columns(2)
    with col1:
        perc_test = st.slider("Percentuale Test Set (percTest)", 0.1, 0.5, 0.3, 0.05)
        perc_selected = st.slider("Percentuale Feature da Selezionare (percSelected)", 0.05, 0.5, 0.2, 0.05)
    with col2:
        allowance = st.number_input("Tolleranza Feature (allowance)", min_value=1, value=1)
        seed = st.number_input("Seed Casuale", min_value=0, value=42)
        alpha_computations = st.number_input("Numero di Alfa (alpha_computations)", min_value=10, value=100)

    if st.button("Run Feature Selection"):
        if not os.path.exists(st.session_state["normalized_csv"]):
            st.error("Esegui prima il preprocessing!")
        else:
            try:
                select_features(
                    normalized_csv=st.session_state["normalized_csv"],
                    reducedTrain_csv=st.session_state["reduced_train_csv"],
                    reducedTest_csv=st.session_state["reduced_test_csv"],
                    output_ottim_csv=st.session_state["ottim_csv"],
                    output_json=st.session_state["selection_json"],
                    target_column=st.session_state["target_column"],
                    percTest=perc_test,
                    percSelected=perc_selected,
                    allowance=allowance,
                    seed=seed,
                    alpha_computations=alpha_computations
                )
                st.success("Feature Selection QUBO completata!")
                
                if os.path.exists(st.session_state["selection_json"]):
                    with open(st.session_state["selection_json"], "r") as f:
                        stats = json.load(f)
                    st.json(stats)
            except Exception as e:
                st.error(f"Errore durante la Feature Selection: {e}")

# --- TAB 3: TRAINING ---
with tab3:
    st.header("3. Addestramento Modello")
    classifier = st.selectbox("Seleziona Classificatore", ["random_forest", "logistic_regression", "svm"])
    
    if st.button("Train Model"):
        if not os.path.exists(st.session_state["reduced_train_csv"]):
            st.error("Esegui prima la Feature Selection!")
        else:
            try:
                train(
                    classifier=classifier,
                    reducedTrain_csv=st.session_state["reduced_train_csv"],
                    target_column=st.session_state["target_column"],
                    model_path=st.session_state["model_path"],
                    metrics_json=st.session_state["train_metrics_json"],
                    seed=seed
                )
                st.success("Modello addestrato con successo!")
                
                if os.path.exists(st.session_state["train_metrics_json"]):
                    with open(st.session_state["train_metrics_json"], "r") as f:
                        metrics = json.load(f)
                    st.json(metrics)
            except Exception as e:
                st.error(f"Errore durante l'addestramento: {e}")

# --- TAB 4: PREDICTION ---
with tab4:
    st.header("4. Predizione e Valutazione Modello")
    
    if st.button("Run Predictions"):
        if not os.path.exists(st.session_state["model_path"]) or not os.path.exists(st.session_state["reduced_test_csv"]):
            st.error("Assicurati di aver completato Feature Selection e Training!")
        else:
            try:
                predict(
                    model_path=st.session_state["model_path"],
                    reducedTest_csv=st.session_state["reduced_test_csv"],
                    target_column=st.session_state["target_column"],
                    predictions_csv=st.session_state["predictions_csv"],
                    stats_json=st.session_state["classification_stats_json"]
                )
                st.success("Valutazione completata!")
                
                if os.path.exists(st.session_state["classification_stats_json"]):
                    with open(st.session_state["classification_stats_json"], "r") as f:
                        stats = json.load(f)
                    st.subheader("Metriche di Valutazione")
                    st.json(stats)
            except Exception as e:
                st.error(f"Errore durante la predizione: {e}")

# --- TAB 5: OUTPUT FILES ---
with tab5:
    st.header("5. Ispezione e Download File Generati")
    
    output_files = [
        ("Normalized Dataset", st.session_state["normalized_csv"]),
        ("Reduced Train Dataset", st.session_state["reduced_train_csv"]),
        ("Reduced Test Dataset", st.session_state["reduced_test_csv"]),
        ("QUBO Optimization Results", st.session_state["ottim_csv"]),
        ("Test Predictions", st.session_state["predictions_csv"]),
    ]
    
    for label, filepath in output_files:
        if os.path.exists(filepath):
            st.subheader(label)
            df = pd.read_csv(filepath)
            st.dataframe(df.head(10))
            with open(filepath, "rb") as f:
                st.download_button(
                    label=f"Download {label}",
                    data=f,
                    file_name=os.path.basename(filepath),
                    mime="text/csv"
                )
        else:
            st.info(f"File {label} non ancora generato.")