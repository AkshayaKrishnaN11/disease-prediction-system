"""
Global configuration for the Disease Prediction System.
Centralizes paths, hyperparameters, disease definitions, and feature columns.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost defaults
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# Random Forest defaults
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# CNN defaults
CNN_PARAMS = {
    "model_name": "resnet50",
    "num_classes": 2,        # Normal vs Pneumonia
    "image_size": 224,
    "batch_size": 32,
    "epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 5,           # early stopping patience
}

# ──────────────────────────────────────────────
# Disease Definitions
# ──────────────────────────────────────────────
DISEASE_CONFIGS = {
    "diabetes": {
        "display_name": "Diabetes",
        "filename": "diabetes.csv",
        "target_column": "Outcome",
        "positive_label": "Diabetic",
        "negative_label": "Non-Diabetic",
        "feature_columns": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        ],
        "zero_impute_columns": [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        ],
    },
    "heart": {
        "display_name": "Heart Disease",
        "filename": "heart.csv",
        "target_column": "target",
        "positive_label": "Heart Disease",
        "negative_label": "No Heart Disease",
        "feature_columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ],
        "categorical_columns": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    },
    "kidney": {
        "display_name": "Chronic Kidney Disease",
        "filename": "kidney_disease.csv",
        "target_column": "classification",
        "positive_label": "CKD",
        "negative_label": "Not CKD",
        "feature_columns": [
            "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
            "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
            "htn", "dm", "cad", "appet", "pe", "ane",
        ],
        "categorical_columns": [
            "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
        ],
    },
    "liver": {
        "display_name": "Liver Disease",
        "filename": "liver.csv",
        "target_column": "Dataset",
        "positive_label": "Liver Disease",
        "negative_label": "No Liver Disease",
        "feature_columns": [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio",
        ],
        "categorical_columns": ["Gender"],
    },
    "breast_cancer": {
        "display_name": "Breast Cancer",
        "filename": "breast_cancer.csv",
        "target_column": "diagnosis",
        "positive_label": "Malignant",
        "negative_label": "Benign",
        "feature_columns": [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
        ],
    },
}

# All supported disease keys
SUPPORTED_DISEASES = list(DISEASE_CONFIGS.keys())
