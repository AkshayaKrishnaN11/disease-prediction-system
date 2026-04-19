"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


# ──────────────────────────────────────────────
# Tabular Prediction
# ──────────────────────────────────────────────
class TabularPredictionRequest(BaseModel):
    """Request body for tabular disease prediction."""
    disease: str = Field(
        ...,
        description="Disease to predict. Options: diabetes, heart, kidney, liver, breast_cancer",
        examples=["diabetes"],
    )
    features: Dict[str, float] = Field(
        ...,
        description="Patient features as key-value pairs",
        examples=[{
            "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
            "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627, "Age": 50,
        }],
    )


class TabularPredictionResponse(BaseModel):
    """Response for tabular disease prediction."""
    disease: str
    prediction: int = Field(..., description="0 = Negative, 1 = Positive")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability: float = Field(..., description="Probability of positive class (0-1)")
    confidence: float = Field(..., description="Model confidence (max probability)")
    model_type: str = Field(default="xgboost", description="Model used for prediction")
    shap_explanation: Optional[Dict[str, float]] = Field(
        None, description="SHAP feature contributions (feature → impact)"
    )


# ──────────────────────────────────────────────
# X-Ray Prediction
# ──────────────────────────────────────────────
class XRayPredictionResponse(BaseModel):
    """Response for chest X-ray prediction."""
    prediction: int = Field(..., description="0 = Normal, 1 = Pneumonia")
    prediction_label: str
    probability: float = Field(..., description="Probability of pneumonia (0-1)")
    confidence: float
    gradcam_heatmap: Optional[str] = Field(
        None, description="Base64 encoded Grad-CAM heatmap overlay (PNG)"
    )


# ──────────────────────────────────────────────
# Model Metrics
# ──────────────────────────────────────────────
class ModelMetricsResponse(BaseModel):
    """Stored model performance metrics."""
    disease: str
    model_type: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    cv_auc_mean: Optional[float] = None
    cv_auc_std: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: List[str] = []
    gpu_available: bool = False


# ──────────────────────────────────────────────
# Error
# ──────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
