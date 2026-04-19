"""
Dependency injection for model loading.
Lazy-loads and caches models as singletons.
"""

import sys
import joblib
import torch
from pathlib import Path
from functools import lru_cache
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SAVED_MODELS_DIR, DISEASE_CONFIGS, CNN_PARAMS


# ──────────────────────────────────────────────
# Model Cache
# ──────────────────────────────────────────────
_tabular_models = {}
_cnn_model = None
_device = None


def get_device():
    """Get the compute device (GPU if available)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_tabular_model(disease: str):
    """
    Load a trained tabular model (best of XGBoost/RF) from disk.
    Returns (model, model_type) or raises FileNotFoundError.
    """
    global _tabular_models

    if disease in _tabular_models:
        return _tabular_models[disease]

    disease_dir = SAVED_MODELS_DIR / disease

    # Try best model first, then XGBoost, then RF
    for model_file, model_type in [
        ("best_model.pkl", "best"),
        ("xgboost_model.pkl", "xgboost"),
        ("random_forest_model.pkl", "random_forest"),
    ]:
        model_path = disease_dir / model_file
        if model_path.exists():
            model = joblib.load(model_path)
            if model_type == "best":
                # Determine actual type
                model_type = "xgboost" if "XGB" in type(model).__name__ else "random_forest"
            _tabular_models[disease] = (model, model_type)
            print(f"  ✓ Loaded {model_type} model for {disease}")
            return model, model_type

    raise FileNotFoundError(
        f"No trained model found for '{disease}' in {disease_dir}. "
        f"Run: python -m train.train_tabular --disease {disease}"
    )


def load_cnn_model():
    """
    Load the trained ResNet-50 CNN model.
    Returns the model on the appropriate device.
    """
    global _cnn_model

    if _cnn_model is not None:
        return _cnn_model

    model_path = SAVED_MODELS_DIR / "chest_xray" / "resnet50_pneumonia.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model not found at {model_path}. "
            f"Run: python -m train.train_cnn"
        )

    device = get_device()

    # Build model architecture
    from src.models.cnn_trainer_utils import build_resnet50_for_inference
    model = build_resnet50_for_inference(model_path, device)

    _cnn_model = model
    print(f"  ✓ Loaded ResNet-50 CNN model on {device}")
    return model


def get_loaded_models() -> list:
    """Return list of currently loaded model names."""
    loaded = list(_tabular_models.keys())
    if _cnn_model is not None:
        loaded.append("chest_xray_cnn")
    return loaded


def load_disease_metrics(disease: str) -> Optional[dict]:
    """Load saved metrics.json for a disease."""
    import json
    metrics_path = SAVED_MODELS_DIR / disease / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)
