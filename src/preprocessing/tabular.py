"""
Tabular data preprocessing pipelines for all 5 disease datasets.
Each disease has a dedicated loader that returns (X_train, X_test, y_train, y_test).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_DIR, DISEASE_CONFIGS, RANDOM_STATE, TEST_SIZE


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun: python -m data.download_datasets"
        )
    return pd.read_csv(path)


# ──────────────────────────────────────────────
# DIABETES
# ──────────────────────────────────────────────
def preprocess_diabetes():
    """Pima Indians Diabetes Dataset preprocessing."""
    cfg = DISEASE_CONFIGS["diabetes"]
    df = _load_csv(cfg["filename"])

    # Replace 0s with NaN in columns where 0 is not valid
    zero_cols = cfg.get("zero_impute_columns", [])
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    X = df[cfg["feature_columns"]]
    y = df[cfg["target_column"]]

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=cfg["feature_columns"])

    # Scale
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=cfg["feature_columns"])

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# HEART DISEASE
# ──────────────────────────────────────────────
def preprocess_heart():
    """Cleveland Heart Disease Dataset preprocessing."""
    cfg = DISEASE_CONFIGS["heart"]
    df = _load_csv(cfg["filename"])

    # Handle '?' values
    df = df.replace("?", np.nan)

    # Select available features
    available = [c for c in cfg["feature_columns"] if c in df.columns]
    X = df[available].copy()

    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Target
    y = df[cfg["target_column"]].astype(int)
    # Binarize: 0 = no disease, 1+ = disease
    y = (y > 0).astype(int)

    # Impute + Scale
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=available)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=available)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# KIDNEY DISEASE
# ──────────────────────────────────────────────
def preprocess_kidney():
    """UCI Chronic Kidney Disease Dataset preprocessing."""
    cfg = DISEASE_CONFIGS["kidney"]
    df = _load_csv(cfg["filename"])

    # Clean target column
    df[cfg["target_column"]] = df[cfg["target_column"]].str.strip()
    df[cfg["target_column"]] = df[cfg["target_column"]].map({"ckd": 1, "notckd": 0, "ckd\t": 1})

    # Drop rows with missing target
    df = df.dropna(subset=[cfg["target_column"]])

    # Replace '?' with NaN
    df = df.replace({"?": np.nan, "\t?": np.nan, "\t": np.nan})

    # Select available features
    available = [c for c in cfg["feature_columns"] if c in df.columns]
    X = df[available].copy()

    # Encode categorical columns
    cat_cols = [c for c in cfg.get("categorical_columns", []) if c in X.columns]
    num_cols = [c for c in available if c not in cat_cols]

    for col in cat_cols:
        X[col] = X[col].astype(str).str.strip()
        le = LabelEncoder()
        mask = X[col].notna() & (X[col] != "nan")
        X.loc[mask, col] = le.fit_transform(X.loc[mask, col])
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = df[cfg["target_column"]].astype(int)

    # Impute + Scale
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=available)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=available)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# LIVER DISEASE
# ──────────────────────────────────────────────
def preprocess_liver():
    """Indian Liver Patient Dataset preprocessing."""
    cfg = DISEASE_CONFIGS["liver"]
    df = _load_csv(cfg["filename"])

    # Encode Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Target: 1 = liver disease, 2 = no liver disease → remap
    df[cfg["target_column"]] = df[cfg["target_column"]].map({1: 1, 2: 0})

    available = [c for c in cfg["feature_columns"] if c in df.columns]
    X = df[available].copy()
    y = df[cfg["target_column"]]

    # Drop NaN targets
    mask = y.notna()
    X = X[mask]
    y = y[mask].astype(int)

    # Impute + Scale
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=available)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=available)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# BREAST CANCER
# ──────────────────────────────────────────────
def preprocess_breast_cancer():
    """Wisconsin Breast Cancer Dataset preprocessing."""
    cfg = DISEASE_CONFIGS["breast_cancer"]
    df = _load_csv(cfg["filename"])

    # Encode diagnosis: M = 1 (malignant), B = 0 (benign)
    if df[cfg["target_column"]].dtype == object:
        df[cfg["target_column"]] = df[cfg["target_column"]].map({"M": 1, "B": 0})

    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    available = [c for c in cfg["feature_columns"] if c in df.columns]
    X = df[available].copy()
    y = df[cfg["target_column"]].astype(int)

    # Scale
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=available)

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ──────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────
PREPROCESSORS = {
    "diabetes": preprocess_diabetes,
    "heart": preprocess_heart,
    "kidney": preprocess_kidney,
    "liver": preprocess_liver,
    "breast_cancer": preprocess_breast_cancer,
}


def get_preprocessed_data(disease: str):
    """
    Load and preprocess data for a specific disease.

    Args:
        disease: One of 'diabetes', 'heart', 'kidney', 'liver', 'breast_cancer'

    Returns:
        (X_train, X_test, y_train, y_test) as DataFrames/Series
    """
    if disease not in PREPROCESSORS:
        raise ValueError(
            f"Unknown disease '{disease}'. Choose from: {list(PREPROCESSORS.keys())}"
        )
    return PREPROCESSORS[disease]()
