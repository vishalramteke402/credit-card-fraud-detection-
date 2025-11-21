import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

# --- Configuration ---
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Credit Card Fraud Detection Demo")
st.markdown("Using Machine Learning (Random Forest & Isolation Forest) on an undersampled dataset.")
st.divider()

# --- Data Loading and Preprocessing (Cached for Performance) ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads the dataset, scales 'Amount', and drops 'Time'."""
    try:
        data = pd.read_csv(file_path, sep=',')
    except FileNotFoundError:
        st.error(f"Error: File not found at '{file_path}'. Please ensure 'creditcard.csv' is in the same directory.")
        return None, None
    
    # Preprocessing
    data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time','Amount'],axis=1)
    
    # Separate features (X) and target (y)
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']

    return X, y, data

@st.cache_resource
def train_models(X, y):
    """Performs undersampling, splits data, and trains models."""
    
    # 1. Undersampling
    number_records_fraud = len(y[y['Class'] == 1])
    fraud_indices = y[y['Class'] == 1].index
    normal_indices = y[y['Class'] == 0].index
    
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    undersample_indices = np.concatenate([fraud_indices, random_normal_indices])
    
    undersample_data = X.loc[undersample_indices]
    undersample_data['Class'] = y.loc[undersample_indices]
    
    X_undersample = undersample_data.drop('Class', axis=1)
    y_undersample = undersample_data['Class']

    # 2. Splitting (using original data for test set evaluation)
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        X_undersample, y_undersample, test_size=0.3, random_state=RANDOM_SEED
    )
    
    # Split the original full dataset for the test set (as used in the notebook)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # 3. Model Training
    
    # Random Forest (Supervised)
    st.info("Training Random Forest Classifier...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf_rf.fit(X_train_u, y_train_u.values.ravel())
    
    # Isolation Forest (Unsupervised)
    st.info("Training Isolation Forest...")
    # Contamination is the percentage of outliers in the data, approximately 0.172%
    clf_if = IsolationForest(
        n_estimators=100, 
        max_samples=len(X_train_u), 
        contamination=0.00172, 
        random_state=RANDOM_SEED, 
        n_jobs=-1
    )
    clf_if.fit(X_train_u)

    return clf_rf, clf_if, X_test_u, y_test_u, X_test, y_test

# --- Main Application Logic ---

X, y, data_raw = load_and_preprocess_data('creditcard.csv')

if X is None:
    st.stop()

# Train models and get test sets
clf_rf, clf_if, X_test_u, y_test_u, X_test, y_test = train_models(X, y)

# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Performance", "Transaction Prediction Demo"])


# --- Data Overview Section ---
if page == "Data Overview":
    st.header("1. Data Overview")
    st.subheader("Raw Data Sample (Pre-processing)")
    st.dataframe(pd.read_csv('creditcard.csv', sep=',').head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Shape and Info")
        st.write(f"**Total Records:** {data_raw.shape[0]:,}")
        st.write(f"**Total Features:** {data_raw.shape[1]}")
        st.markdown("**Note:** Features V1-V28 are PCA components. 'NormalizedAmount' is the scaled transaction amount.")

    with col2:
        st.subheader("Class Distribution (Imbalance)")
        
        # Plotting Class Distribution
        count_classes = data_raw['Class'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=count_classes.index, y=count_classes.values, ax=ax, palette=['#4CAF50', '#FF4B4B'])
        ax.set_title("Transaction Class Distribution", fontsize=16)
        ax.set_xticks(range(2))
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_yscale('log')
        st.pyplot(fig)
        
        fraud_percent = (count_classes[1] / count_classes.sum()) * 100
        st.markdown(f"**Fraudulent transactions: {count_classes[1]:,} ({fraud_percent:.3f}%)**")


# --- Model Performance Section ---
elif page == "Model Performance":
    st.header("2. Model Performance")
    st.markdown("Performance metrics for the **Random Forest** classifier, trained on the **undersampled** data but tested on the **full, original test set**.")

    # 1. Random Forest Prediction and Evaluation
    y_pred_rf = clf_rf.predict(X_test)
    
    st.subheader("Random Forest Classifier Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Accuracy Score (Full Test Set)", value=f"{accuracy_score(y_test.values.ravel(), y_pred_rf):.4f}")
        st.text("Classification Report:")
        st.code(classification_report(y_test.values.ravel(), y_pred_rf, target_names=LABELS))
        
    with col2:
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test.values.ravel(), y_pred_rf)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Random Forest Confusion Matrix')
        st.pyplot(fig_cm)
        
    st.caption("Note: The Random Forest model shows high accuracy but the metrics for the minority class (Fraud) are more important for real-world application.")
    
    st.subheader("Isolation Forest Results (on Undersampled Test Set)")
    
    # 2. Isolation Forest Prediction and Evaluation (Requires transformation of labels)
    y_pred_if_raw = clf_if.predict(X_test_u)
    y_pred_if = np.array(y_pred_if_raw)
    y_pred_if[y_pred_if == 1] = 0   # 1 (inlier) -> 0 (Normal)
    y_pred_if[y_pred_if == -1] = 1  # -1 (outlier) -> 1 (Fraud)

    col3, col4 = st.columns(2)
    with col3:
        st.metric(label="Accuracy Score (Undersampled Test Set)", value=f"{accuracy_score(y_test_u, y_pred_if):.4f}")
        st.text("Classification Report:")
        st.code(classification_report(y_test_u, y_pred_if, target_names=LABELS))
    
    with col4:
        st.text("Confusion Matrix:")
        cm_if = confusion_matrix(y_test_u, y_pred_if)
        fig_cm_if, ax_cm_if = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax_cm_if)
        ax_cm_if.set_xlabel('Predicted Label')
        ax_cm_if.set_ylabel('True Label')
        ax_cm_if.set_title('Isolation Forest Confusion Matrix')
        st.pyplot(fig_cm_if)
    

# --- Transaction Prediction Demo Section ---
elif page == "Transaction Prediction Demo":
    st.header("3. Transaction Prediction Demo")
    st.markdown("Use a sample transaction from the test set to see how the **Random Forest** model classifies it.")
    
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    # Find indices for demonstration
    normal_indices = y_test_reset[y_test_reset['Class'] == 0].index.tolist()
    fraud_indices = y_test_reset[y_test_reset['Class'] == 1].index.tolist()

    # Create a selection list
    selection_options = {
        "Random Normal Transaction": random.choice(normal_indices),
        "Random Fraudulent Transaction": random.choice(fraud_indices),
    }

    st.subheader("Select a Transaction Sample")
    sample_type = st.selectbox("Choose a sample type:", list(selection_options.keys()))
    
    # Get the selected index
    selected_idx = selection_options[sample_type]

    # Get the feature values for the selected sample
    sample_features = X_test_reset.iloc[selected_idx].values.reshape(1, -1)
    true_label = y_test_reset.iloc[selected_idx]['Class']
    
    st.markdown(f"---")
    
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("Selected Transaction Features")
        st.dataframe(pd.DataFrame(sample_features, columns=X_test.columns).T.rename(columns={0: "Value"}))
        st.warning(f"**True Label:** **{LABELS[true_label]}**")

    with col_output:
        st.subheader("Model Prediction")
        
        # Make prediction
        prediction_rf = clf_rf.predict(sample_features)[0]
        prediction_label = LABELS[prediction_rf]
        
        if prediction_rf == 1:
            st.error(f"Prediction: **{prediction_label}**")
        else:
            st.success(f"Prediction: **{prediction_label}**")
            
        # Display probability (for Random Forest)
        proba = clf_rf.predict_proba(sample_features)[0]
        st.markdown(f"**Probability of Normal:** `{proba[0]:.4f}`")
        st.markdown(f"**Probability of Fraud:** `{proba[1]:.4f}`")
        
        # Check if the prediction was correct
        if prediction_rf == true_label:
            st.success("✅ Prediction is correct.")
        else:
            st.error("❌ Prediction is incorrect (Misclassification).")

