"""
FastAPI Application — Disease Prediction & Medical Diagnosis System.

Main entry point. Mounts all route modules and configures CORS.

Usage:
    python -m api.main
    uvicorn api.main:app --reload --port 8000
"""

import sys
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import API_HOST, API_PORT, SAVED_MODELS_DIR
from api.schemas import HealthResponse
from api.routes import predict, imaging, metrics
from api.dependencies import get_loaded_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    print("\n" + "=" * 60)
    print("🏥 Disease Prediction API — Starting")
    print("=" * 60)
    print(f"  Models dir: {SAVED_MODELS_DIR}")
    print(f"  GPU: {'✓ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ CPU only'}")
    print("=" * 60)
    yield
    print("\n🏥 Disease Prediction API — Shutting down")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="🏥 Disease Prediction & Medical Diagnosis API",
    description=(
        "ML-powered REST API for predicting diseases from patient data and chest X-rays.\n\n"
        "**Tabular Models**: Diabetes, Heart Disease, Kidney Disease, Liver Disease, Breast Cancer\n\n"
        "**Computer Vision**: Chest X-Ray Pneumonia Detection (ResNet-50)\n\n"
        "**Explainability**: SHAP (tabular) + Grad-CAM (X-ray)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
import os as _os
_cors_origins = _os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins + ["*"],  # Allow all in dev; restrict in prod via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(imaging.router)
app.include_router(metrics.router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=get_loaded_models(),
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/", tags=["System"], summary="API Root")
async def root():
    return {
        "message": "🏥 Disease Prediction & Medical Diagnosis API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict_tabular": "POST /predict/tabular",
            "predict_xray": "POST /predict/xray",
            "list_diseases": "GET /predict/diseases",
            "metrics": "GET /metrics/{disease}",
            "all_metrics": "GET /metrics/",
        },
    }


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
