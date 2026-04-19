"""
Shared pytest fixtures for the Disease Prediction test suite.
"""

import sys
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# ── Ensure project root is importable ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app
from src.config import SAVED_MODELS_DIR, DISEASE_CONFIGS, SUPPORTED_DISEASES


@pytest.fixture(scope="session")
def client():
    """FastAPI test client (shared for the entire test session)."""
    return TestClient(app)


@pytest.fixture(scope="session")
def supported_diseases():
    """List of supported tabular disease keys."""
    return SUPPORTED_DISEASES


@pytest.fixture(scope="session")
def disease_configs():
    """Full disease configuration dict."""
    return DISEASE_CONFIGS


@pytest.fixture(scope="session")
def models_dir():
    """Path to saved_models directory."""
    return SAVED_MODELS_DIR


@pytest.fixture
def sample_diabetes_features():
    """Realistic sample diabetes patient features."""
    return {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50,
    }


@pytest.fixture
def sample_heart_features():
    """Realistic sample heart disease patient features."""
    return {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }


@pytest.fixture
def sample_xray_path():
    """Path to a test X-ray image. Uses first available image from test data."""
    test_dirs = [
        SAVED_MODELS_DIR.parent / "data" / "raw" / "chest_xray" / "test" / "NORMAL",
        SAVED_MODELS_DIR.parent / "data" / "raw" / "chest_xray" / "test" / "PNEUMONIA",
    ]
    for d in test_dirs:
        if d.exists():
            images = list(d.glob("*.jpeg")) + list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if images:
                return images[0]
    return None
