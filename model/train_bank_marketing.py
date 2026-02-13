
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import urllib.request
import zipfile
import io
import os
import pandas as pd
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


METRIC_COLUMNS = [
    'model', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc'
]


ARTIFACTS = Path('model/artifacts')
ARTIFACTS.mkdir(parents=True, exist_ok=True)


UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"


def fetch_uci_bank_additional_full(local_path: str | Path = "data/bank-additional-full.csv",
                                   url: str = UCI_URL,
                                   verbose: bool = True) -> pd.DataFrame:
    """
    Fetch UCI 'Bank Marketing' (bank-additional-full.csv) from the web; if that fails,
    load the CSV from a local fallback path (default: data/bank-additional-full.csv).

    Parameters
    ----------
    local_path : str | Path
        Path to the local fallback CSV.
    url : str
        URL of the UCI zip file.
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    pd.DataFrame
        The dataset loaded into a DataFrame.

    Raises
    ------
    FileNotFoundError
        If web download fails AND local file is not found.
    Exception
        Propagates non-FileNotFound errors when reading the local file.
    """
    local_path = Path(local_path)

    # 1) Try fetching from web
    try:
        if verbose:
            print("Downloading UCI bank marketing zip...")
        with urllib.request.urlopen(url) as resp:
            zbytes = resp.read()

        if verbose:
            print("Opening ZIP and locating 'bank-additional-full.csv'...")
        zf = zipfile.ZipFile(io.BytesIO(zbytes))

        # Prefer the specific path within the zip if present
        inner = None
        for name in zf.namelist():
            if name.lower().endswith("bank-additional/bank-additional-full.csv"):
                inner = name
                break

        if inner is None:
            # Fallback: search for any file with the expected name
            candidates = [n for n in zf.namelist(
            ) if "bank-additional-full.csv" in n.lower()]
            if not candidates:
                raise FileNotFoundError(
                    "bank-additional-full.csv not found inside the UCI zip.")
            inner = candidates[0]

        if verbose:
            print(f"Reading '{inner}' from ZIP...")
        with zf.open(inner) as f:
            df = pd.read_csv(f, sep=";")
        if verbose:
            print("Loaded dataset from web successfully.")
        return df

    except Exception as web_err:
        if verbose:
            print(f"Web fetch failed: {web_err}")
            print(f"Falling back to local file: {local_path}")

        # 2) Fallback: read from local CSV
        if not local_path.exists():
            raise FileNotFoundError(
                f"Web fetch failed and local fallback not found at '{local_path}'. "
                "Please ensure the file exists or check your network."
            ) from web_err

        # Read local file
        df = pd.read_csv(local_path, sep=";")
        if verbose:
            print("Loaded dataset from local file successfully.")
        return df


def prepare_dataframe(df):
    # Avoid leakage: drop duration
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
    # Rename target and convert to 0/1
    df = df.rename(columns={'y': 'target'})
    df['target'] = (df['target'].astype(
        str).str.strip().str.lower() == 'yes').astype(int)
    # Split features
    feature_cols = [c for c in df.columns if c != 'target']
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return df, feature_cols, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    cat_proc = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    num_proc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    pre = ColumnTransformer([
        ('cat', cat_proc, cat_cols),
        ('num', num_proc, num_cols),
    ])
    return pre


def build_models():
    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB(),
        'random_forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'xgboost': XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ),
    }
    return models


def train_and_save(df):
    df, feature_cols, cat_cols, num_cols = prepare_dataframe(df)
    print(df.head)
    X, y = df[feature_cols], df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = build_preprocessor(cat_cols, num_cols)
    models = build_models()

    rows = []

    for name, clf in models.items():
        # GaussianNB expects dense input after preprocessing; others can handle sparse.
        pipe = Pipeline([
            ('pre', pre),
            ('clf', clf)
        ])
        print(f'Training {name}...')
        pipe.fit(X_train, y_train)

        # Predictions & probabilities
        y_pred = pipe.predict(X_test)
        proba = None
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            proba = pipe.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = compute_binary_metrics(y_test, y_pred, proba)
        rows.append({'model': name, **metrics})

        # Save model pipeline
        joblib.dump(pipe, ARTIFACTS / f'{name}.joblib')

    # Save metrics summary

    metrics_df = pd.DataFrame(rows, columns=METRIC_COLUMNS)

    metrics_df.to_csv('model/metrics_summary.csv', index=False)
    Path('model/metrics_summary.md').write_text(metrics_df.to_markdown(index=False, floatfmt='.4f'))

    print('Saved: model/artifacts/*.joblib, model/metrics_summary.csv, model/metrics_summary.md')


def compute_binary_metrics(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    except Exception:
        metrics['auc'] = float('nan')
    return metrics


def report_confusion_and_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return cm, report


if __name__ == '__main__':
    df = fetch_uci_bank_additional_full()
    train_and_save(df)
