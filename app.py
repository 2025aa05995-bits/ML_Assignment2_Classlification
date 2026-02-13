import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title='Bank Marketing — ML Models', layout='wide')
st.title('Bank Marketing — Classification Models (Streamlit)')
st.write('Upload **test CSV** (schema like UCI bank marketing), pick a model, and view metrics.')

# --- Load initial test data from data/ and show table ---
from pathlib import Path as _Path
init_path = _Path('data/test_data.csv')
initial_df = None
if init_path.exists():
    try:
        initial_df = pd.read_csv(init_path, sep=';')
    except Exception:
        initial_df = pd.read_csv(init_path)

st.subheader('Initial test data (from data/test_data.csv)')
if initial_df is not None:
    st.dataframe(initial_df.head(50))
else:
    st.info('No initial test data found at data/test_data.csv')

ARTIFACTS = Path('model/artifacts')
models_available = [p.stem for p in ARTIFACTS.glob('*.joblib')]

if not models_available:
    st.warning('No saved models found in `model/artifacts`. Please run `python model/train_bank_marketing.py` first to generate pipelines and metrics.')
else:
    model_name = st.selectbox('Select a trained model', options=models_available)
    pipe = joblib.load(ARTIFACTS / f'{model_name}.joblib')

    uploaded = st.file_uploader('Upload **test data** CSV (semicolon or comma delimiter accepted)', type=['csv'])
    label_col = st.text_input('Optional: Name of label column (e.g., `y` or `target`). If provided, metrics will be computed.', value='')

    if uploaded is not None:
        try: df = pd.read_csv(uploaded, sep=';')
        except Exception:
            uploaded.seek(0); df = pd.read_csv(uploaded)
        y_true = None
        if label_col and label_col in df.columns:
            y_raw = df[label_col]
            if y_raw.dtype == 'object': y_true = (y_raw.str.strip().str.lower() == 'yes').astype(int)
            else: y_true = y_raw.astype(int)
            df = df.drop(columns=[label_col])
        y_pred = pipe.predict(df)
        st.subheader('Predictions')
        st.write(pd.DataFrame({'prediction': y_pred}))

        if y_true is not None:
            proba = None
            if hasattr(pipe.named_steps['clf'], 'predict_proba'): proba = pipe.predict_proba(df)[:,1]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, proba) if proba is not None else float('nan')
            st.subheader('Evaluation Metrics')
            col1, col2, col3 = st.columns(3)
            with col1: st.metric('Accuracy', f'{acc:.4f}')
            with col2: st.metric('Precision', f'{prec:.4f}')
            with col3: st.metric('Recall', f'{rec:.4f}')
            col4, col5 = st.columns(2)
            with col4: st.metric('F1', f'{f1:.4f}')
            with col5: st.metric('MCC', f'{mcc:.4f}')
            st.metric('AUC', '-' if np.isnan(auc) else f'{auc:.4f}')

            st.subheader('Confusion Matrix')
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

            st.subheader('Classification Report')
            st.text(classification_report(y_true, y_pred, digits=4))
        else: st.info('No label provided — showing predictions only.')

    # --- Use initial test data if no upload and user requests ---
    if uploaded is None and initial_df is not None and st.checkbox('Use initial test data from data/test_data.csv'):
        df = initial_df.copy()
        y_true = None
        for _lbl in ['target','y']:
            if _lbl in df.columns:
                y_raw = df[_lbl]
                if y_raw.dtype == 'object': y_true = (y_raw.str.strip().str.lower() == 'yes').astype(int)
                else: y_true = y_raw.astype(int)
                df = df.drop(columns=[_lbl])
                break
        y_pred = pipe.predict(df)
        st.subheader('Predictions (initial test data)')
        st.write(pd.DataFrame({'prediction': y_pred}))
        if y_true is not None:
            proba = None
            if hasattr(pipe.named_steps['clf'], 'predict_proba'): proba = pipe.predict_proba(df)[:,1]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, proba) if proba is not None else float('nan')
            st.subheader('Evaluation Metrics (initial test data)')
            col1, col2, col3 = st.columns(3)
            with col1: st.metric('Accuracy', f'{acc:.4f}')
            with col2: st.metric('Precision', f'{prec:.4f}')
            with col3: st.metric('Recall', f'{rec:.4f}')
            col4, col5 = st.columns(2)
            with col4: st.metric('F1', f'{f1:.4f}')
            with col5: st.metric('MCC', f'{mcc:.4f}')
            st.metric('AUC', '-' if np.isnan(auc) else f'{auc:.4f}')
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

# If no upload, show trained metrics table (if present)
metrics_path = Path('model/metrics_summary.csv')
if metrics_path.exists():
    st.subheader('Trained Model Metrics (held‑out test set)')
    st.dataframe(pd.read_csv(metrics_path))
else: st.info('Train the models to generate metrics (model/metrics_summary.csv).')
