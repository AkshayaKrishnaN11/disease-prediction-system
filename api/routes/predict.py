"""
POST /predict/{disease} — Tabular disease prediction endpoint.
Accepts patient features, returns prediction + SHAP explanation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import TabularPredictionRequest, TabularPredictionResponse, ErrorResponse
from api.dependencies import load_tabular_model
from src.config import DISEASE_CONFIGS, SUPPORTED_DISEASES

router = APIRouter(prefix="/predict", tags=["Tabular Prediction"])


@router.post(
    "/tabular",
    response_model=TabularPredictionResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Predict disease from patient features",
    description="Submit patient features (vitals, lab reports) to get a disease prediction with SHAP explanation.",
)
async def predict_tabular(request: TabularPredictionRequest):
    disease = request.disease.lower()

    # Validate disease
    if disease not in SUPPORTED_DISEASES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown disease '{disease}'. Options: {SUPPORTED_DISEASES}",
        )

    # Load model
    try:
        model, model_type = load_tabular_model(disease)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Get expected features
    cfg = DISEASE_CONFIGS[disease]
    expected_features = cfg["feature_columns"]

    # Build feature vector in correct order
    feature_values = []
    missing_features = []
    for feat in expected_features:
        if feat in request.features:
            feature_values.append(request.features[feat])
        else:
            missing_features.append(feat)
            feature_values.append(0.0)  # default

    if missing_features:
        print(f"  ⚠ Missing features filled with 0: {missing_features}")

    # Create DataFrame with correct column names
    X = pd.DataFrame([feature_values], columns=expected_features)

    # Predict
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    prob_positive = float(probabilities[1])
    confidence = float(max(probabilities))

    # Label
    prediction_label = cfg["positive_label"] if prediction == 1 else cfg["negative_label"]

    # SHAP explanation (optional, may fail)
    shap_explanation = None
    try:
        from src.explainability.shap_explainer import explain_single_prediction
        result = explain_single_prediction(model, X, feature_names=expected_features)
        shap_explanation = result["contributions"]
    except Exception as e:
        print(f"  ⚠ SHAP explanation failed: {e}")

    return TabularPredictionResponse(
        disease=disease,
        prediction=prediction,
        prediction_label=prediction_label,
        probability=round(prob_positive, 4),
        confidence=round(confidence, 4),
        model_type=model_type,
        shap_explanation=shap_explanation,
    )


@router.get(
    "/diseases",
    summary="List supported diseases",
    description="Get list of all supported diseases and their required features.",
)
async def list_diseases():
    diseases = {}
    for key, cfg in DISEASE_CONFIGS.items():
        diseases[key] = {
            "display_name": cfg["display_name"],
            "required_features": cfg["feature_columns"],
            "positive_label": cfg["positive_label"],
            "negative_label": cfg["negative_label"],
        }
    return {"diseases": diseases}
