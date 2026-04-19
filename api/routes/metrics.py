"""
GET /metrics/{disease} — Model performance metrics endpoint.
Returns stored evaluation metrics for trained models.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import ModelMetricsResponse, ErrorResponse
from api.dependencies import load_disease_metrics
from src.config import SUPPORTED_DISEASES

router = APIRouter(prefix="/metrics", tags=["Model Metrics"])


@router.get(
    "/{disease}",
    response_model=dict,
    responses={404: {"model": ErrorResponse}},
    summary="Get model performance metrics",
    description="Retrieve stored accuracy, F1, AUC, confusion matrix, and feature importance for a trained model.",
)
async def get_metrics(disease: str):
    disease = disease.lower()

    if disease not in SUPPORTED_DISEASES and disease != "chest_xray":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown disease '{disease}'. Options: {SUPPORTED_DISEASES + ['chest_xray']}",
        )

    metrics = load_disease_metrics(disease)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for '{disease}'. Train the model first.",
        )

    return {"disease": disease, "metrics": metrics}


@router.get(
    "/",
    summary="Get all available model metrics",
    description="List all diseases with trained models and their summary metrics.",
)
async def get_all_metrics():
    results = {}
    all_diseases = SUPPORTED_DISEASES + ["chest_xray"]

    for disease in all_diseases:
        metrics = load_disease_metrics(disease)
        if metrics:
            results[disease] = metrics

    return {
        "total_models": len(results),
        "diseases": results,
    }
