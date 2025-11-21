"""
Streamlit app for Credit Card Fraud Detection
Features:
- Upload creditcard.csv or use default path
- Full pipeline training (undersampling, SMOTE optional, anomaly detection, Random Forest)
- Option to load a pretrained model (.pkl/.joblib)
- Evaluation on full test set
- Plots: class distribution, confusion matrix, ROC curve
- Export trained model for later use
- Simple prediction UI for single-row inputs

Usage:
    streamlit run streamlit_creditcard_fraud_app.py

Dependencies:
    pip install streamlit scikit-learn pandas matplotlib seaborn imbalanced-learn joblib
    (imblearn optional but recommended)

"""

import io
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_fscore_support)

# Optional imblearn
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# Constants
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
DEFAULT_FILEPATH = "creditcard.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# --- Utility functions ---
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        # file is an UploadedFile
        df = pd.read_csv(file)
    return df

@st.cache_data
def preprocess(df: pd.DataFrame, scaler=None):
    df = df.copy()
    # Drop NaNs
    df.dropna(inplace=True)
    # Create NormalizedAmount
    if scaler is None:
        scaler = StandardScaler()
        df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    else:
        df['NormalizedAmount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
    # Drop Time and Amount
    if 'Time' in df.columns:
        df = df.drop(['Time', 'Amount'], axis=1)
    else:
        df = df.drop(['Amount'], axis=1)
    return df, scaler

@st.cache_data
def undersample(X, y, random_state=RANDOM_SEED):
    # return undersampled X,y balanced by minority class
    fraud_count = y.sum()
    fraud_indices = y[y == 1].index
    normal_indices = y[y == 0].index
    random_normal_indices = np.random.choice(normal_indices, int(fraud_count), replace=False)
    undersample_indices = np.concatenate([fraud_indices, random_normal_indices])
    return X.loc[undersample_indices].reset_index(drop=True), y.loc[undersample_indices].reset_index(drop=True)

@st.cache_data
def split_full(X, y, test_size=0.3, random_state=RANDOM_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

@st.cache_data
def train_random_forest(X_train, y_train, n_estimators=100, random_state=RANDOM_SEED):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

# Anomaly detectors
def train_anomaly_models(X_train, contamination=None):
    models = {}
    # If contamination not specified, heuristically use  fraction of frauds in training
    if contamination is None:
        contamination = 0.01

    models['IsolationForest'] = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination,
                                                random_state=RANDOM_SEED, n_jobs=-1)
    models['LocalOutlierFactor'] = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    models['OneClassSVM'] = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination)

    for name, m in models.items():
        # LOF must use fit for novelty=True
        m.fit(X_train)
    return models

# Metrics and plotting

def plot_confusion_matrix(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = None
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    if fig:
        st.pyplot(fig)
    else:
        return ax


def prediction_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_pred)
    except Exception:
        roc = None
    rep = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    return acc, roc, rep

# --- Streamlit UI ---
st.title("🕵️ Credit Card Fraud Detection — Streamlit App")

# Sidebar: Options
st.sidebar.header("App Configuration")
app_mode = st.sidebar.selectbox("Choose mode", ["Train & Evaluate", "Predict (Single)", "Load Pretrained Model", "README"]) 

uploaded_file = st.sidebar.file_uploader("Upload creditcard.csv (optional)", type=['csv'])
use_default = st.sidebar.checkbox("Use default creditcard.csv in working dir (if exists)", value=True)

if uploaded_file is None and use_default and os.path.exists(DEFAULT_FILEPATH):
    df = load_csv(DEFAULT_FILEPATH)
elif uploaded_file is not None:
    df = load_csv(uploaded_file)
else:
    df = None

# Global placeholders
trained_model = None
scaler_obj = None

# README Tab
if app_mode == "README":
    st.header("README — Credit Card Fraud Detection Streamlit App")
    st.markdown("""
    **Features:**
    - Upload dataset or use default `creditcard.csv`.
    - Full training pipeline with options: undersampling, SMOTE, class weights.
    - Train anomaly detection models (IsolationForest, LOF, OneClassSVM) on *normal* data.
    - Train RandomForest classifier on balanced/SMOTE/undersampled data.
    - Evaluate on full held-out test set and visualize results.
    - Save/Load trained models for fast predictions.

    **How to use:**
    1. Choose `Train & Evaluate` to run pipeline and train models.
    2. Optionally download the trained RandomForest model to use later.
    3. Choose `Predict (Single)` to input a single transaction and get a prediction.
    4. Choose `Load Pretrained Model` to upload a `.pkl`/`.joblib` model.

    **Notes:**
    - `imblearn` (SMOTE) is optional but recommended to improve supervised performance.
    - Anomaly detectors should be trained on normal examples only — this app does that by default.
    """)
    st.stop()

# TRAIN & EVALUATE
if app_mode == "Train & Evaluate":
    st.header("Train & Evaluate Pipeline")
    if df is None:
        st.warning("No dataset found. Upload a creditcard.csv in the sidebar or enable default file.")
        st.stop()

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.sidebar.markdown("---")
    st.sidebar.subheader("Training Options")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30)
    apply_undersample = st.sidebar.checkbox("Use Undersampling (balanced by minority)", value=True)
    apply_smote = st.sidebar.checkbox("Use SMOTE (if imblearn available)", value=False)
    smote_ratio = st.sidebar.slider("SMOTE sampling ratio (when enabled)", 0.1, 1.0, 1.0)
    use_class_weight = st.sidebar.checkbox("Use class_weight='balanced' for RF", value=True)
    n_estimators = st.sidebar.number_input("RandomForest n_estimators", min_value=10, max_value=1000, value=100)

    if apply_smote and not IMBLEARN_AVAILABLE:
        st.sidebar.error("imblearn not available. Install: pip install imbalanced-learn")

    # Preprocess
    with st.spinner("Preprocessing data..."):
        df_proc, scaler_obj = preprocess(df)
        X = df_proc.drop('Class', axis=1)
        y = df_proc['Class']

    st.markdown(f"**Dataset size:** {df_proc.shape[0]} rows — Fraud cases: {int(y.sum())}")

    # Split full dataset
    X_train_full, X_test_full, y_train_full, y_test_full = split_full(X, y, test_size=test_size/100)

    # Prepare data for anomaly detection (train only on normal samples)
    X_train_norm = X_train_full[y_train_full == 0]
    st.markdown(f"Training anomaly detectors on normal-only data: {X_train_norm.shape[0]} rows")

    # Train Anomaly models
    contamination_est = max(0.001, y_train_full.mean())
    with st.spinner("Training anomaly detection models..."):
        anom_models = train_anomaly_models(X_train_norm, contamination=contamination_est)

    st.success("Anomaly detectors trained.")

    # Evaluate anomaly detectors on test subset - convert predictions
    st.subheader("Anomaly Detection Evaluation (on holdout from undersampled balanced set)")
    # For evaluation of anomaly detectors, we create an evaluation set from the test split (keeping original imbalance)
    X_eval = X_test_full
    y_eval = y_test_full

    for name, model in anom_models.items():
        st.markdown(f"**{name}**")
        y_pred_raw = model.predict(X_eval)
        # LOF and others: predict -> 1 inlier, -1 outlier
        y_pred = np.where(y_pred_raw == 1, 0, 1)
        acc, roc, rep = prediction_metrics(y_eval, y_pred)
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"ROC AUC (approx): {roc}")
        st.json(rep)
        fig, ax = plt.subplots()
        plot_confusion_matrix(y_eval, y_pred, ax=ax)
        st.pyplot(fig)

    # Prepare supervised training data
    st.subheader("Supervised Training — Random Forest")
    # Option: undersample or SMOTE or class-weight
    if apply_undersample:
        X_train_sup, y_train_sup = undersample(X_train_full, y_train_full)
        st.write(f"Undersampled training size: {X_train_sup.shape[0]}")
    else:
        X_train_sup, y_train_sup = X_train_full.copy(), y_train_full.copy()

    if apply_smote and IMBLEARN_AVAILABLE:
        st.write("Applying SMOTE to training data...")
        sm = SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_SEED)
        X_train_sup, y_train_sup = sm.fit_resample(X_train_sup, y_train_sup)
        st.write(f"After SMOTE: {X_train_sup.shape[0]} rows — Fraud count: {int(y_train_sup.sum())}")

    # Train RandomForest
    rf_params = {"n_estimators": int(n_estimators), "random_state": RANDOM_SEED, "n_jobs": -1}
    if use_class_weight:
        rf_params['class_weight'] = 'balanced'
    with st.spinner("Training Random Forest..."):
        clf_rf = RandomForestClassifier(**rf_params)
        clf_rf.fit(X_train_sup, y_train_sup)
    st.success("Random Forest trained.")

    # Evaluate on full test set
    y_pred_rf = clf_rf.predict(X_test_full)
    acc_rf, roc_rf, rep_rf = prediction_metrics(y_test_full, y_pred_rf)

    st.write(f"**Random Forest — Test accuracy:** {acc_rf:.4f}")
    if roc_rf is not None:
        st.write(f"**ROC AUC (approx):** {roc_rf:.4f}")

    st.json(rep_rf)
    fig, ax = plt.subplots()
    plot_confusion_matrix(y_test_full, y_pred_rf, ax=ax)
    st.pyplot(fig)

    # Feature importances
    st.subheader("Feature Importances")
    importances = pd.Series(clf_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances.rename('importance'))

    # Allow saving model
    model_fname = st.text_input("Filename to save model (without extension)", value=f"rf_model_{int(time.time())}")
    if st.button("Save Model"):
        save_path = os.path.join(MODEL_DIR, model_fname + '.joblib')
        joblib.dump({'model': clf_rf, 'scaler': scaler_obj}, save_path)
        st.success(f"Model saved to {save_path}")
        st.markdown(f"Download link (from server): `{save_path}` — or copy the file from server filesystem.")

    # Make trained model available in session state for prediction tab
    st.session_state['trained_model'] = {'model': clf_rf, 'scaler': scaler_obj}

# LOAD PRETRAINED MODEL
elif app_mode == "Load Pretrained Model":
    st.header("Load Pretrained Model")
    uploaded_model = st.file_uploader("Upload a .joblib or .pkl trained model", type=['joblib', 'pkl'])
    if uploaded_model is not None:
        try:
            m = joblib.load(uploaded_model)
            st.success("Model loaded successfully.")
            st.write("Model keys:", list(m.keys()) if isinstance(m, dict) else str(type(m)))
            st.session_state['trained_model'] = m
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.info("Upload a model file to use for predictions.")

# PREDICT SINGLE
elif app_mode == "Predict (Single)":
    st.header("Single Transaction Prediction")
    if 'trained_model' not in st.session_state:
        st.warning("No trained model found in session. You can train one in 'Train & Evaluate' or upload in 'Load Pretrained Model'.")
        st.stop()

    model_bundle = st.session_state['trained_model']
    # model_bundle can be either {'model': clf, 'scaler': scaler} or clf object
    if isinstance(model_bundle, dict) and 'model' in model_bundle:
        clf = model_bundle['model']
        scaler = model_bundle.get('scaler', None)
    else:
        clf = model_bundle
        scaler = None

    # Build input form from expected columns
    # We assume original dataset columns V1..V28 + NormalizedAmount
    sample_cols = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else None
    if sample_cols is None:
        # Fallback: try common columns
        sample_cols = [c for c in ['V' + str(i) for i in range(1, 29)] + ['NormalizedAmount'] if True]

    st.write("Enter feature values for a single transaction. Leave blank to use sample (mean) values.")
    user_input = {}
    df_for_defaults = load_csv(DEFAULT_FILEPATH) if os.path.exists(DEFAULT_FILEPATH) else None
    if df_for_defaults is not None:
        df_defaults, _ = preprocess(df_for_defaults)
    else:
        df_defaults = None

    cols_to_show = sample_cols
    for col in cols_to_show:
        default_val = None
        if df_defaults is not None and col in df_defaults.columns:
            default_val = float(df_defaults[col].mean())
        user_input[col] = st.number_input(col, value=default_val if default_val is not None else 0.0)

    if st.button("Predict"):
        X_single = pd.DataFrame([user_input])
        # Ensure columns align
        X_single = X_single.reindex(columns=clf.feature_names_in_) if hasattr(clf, 'feature_names_in_') else X_single
        pred = clf.predict(X_single)[0]
        proba = clf.predict_proba(X_single)[0] if hasattr(clf, 'predict_proba') else None
        st.write("**Prediction:**", LABELS[int(pred)])
        if proba is not None:
            st.write("Probability (Normal, Fraud):", proba)

# Default fallback
else:
    st.write("Choose an option from the sidebar.")

# Footer
st.markdown("---")
st.caption("App created to demonstrate full pipeline for credit card fraud detection. Modify parameters in the sidebar as needed.")
