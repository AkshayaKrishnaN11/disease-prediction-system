"""
Tests for model loading and prediction logic.
Ensures saved models can be loaded and produce valid outputs.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SAVED_MODELS_DIR, DISEASE_CONFIGS, SUPPORTED_DISEASES


class TestModelFiles:
    """Verify all expected model files exist."""

    def is_ci(self):
        """Check if running in GitHub Actions."""
        import os
        return os.getenv("GITHUB_ACTIONS") == "true"

    @pytest.mark.parametrize("disease", SUPPORTED_DISEASES)
    def test_model_files_exist(self, disease):
        if self.is_ci():
            pytest.skip("Skipping file existence check in CI (models are gitignored)")
        
        disease_dir = SAVED_MODELS_DIR / disease
        assert disease_dir.exists(), f"Model directory missing: {disease_dir}"

        # At least one model file should exist
        model_files = list(disease_dir.glob("*.pkl"))
        assert len(model_files) > 0, f"No .pkl model files in {disease_dir}"

    @pytest.mark.parametrize("disease", SUPPORTED_DISEASES)
    def test_metrics_json_exists(self, disease):
        metrics_path = SAVED_MODELS_DIR / disease / "metrics.json"
        assert metrics_path.exists(), f"metrics.json missing for {disease}"

    def test_cnn_model_exists(self):
        cnn_path = SAVED_MODELS_DIR / "chest_xray" / "resnet50_pneumonia.pth"
        assert cnn_path.exists(), f"CNN model missing: {cnn_path}"

    def test_cnn_metrics_exist(self):
        metrics_path = SAVED_MODELS_DIR / "chest_xray" / "metrics.json"
        assert metrics_path.exists(), "chest_xray/metrics.json missing"


class TestTabularModelLoading:
    """Test that tabular models can be loaded and produce predictions."""

    def is_ci(self):
        import os
        return os.getenv("GITHUB_ACTIONS") == "true"

    @pytest.mark.parametrize("disease", SUPPORTED_DISEASES)
    def test_model_loads(self, disease):
        if self.is_ci():
            pytest.skip("Skipping model loading test in CI")
        from api.dependencies import load_tabular_model
        model, model_type = load_tabular_model(disease)
        assert model is not None
        assert model_type in ["xgboost", "random_forest"]

    @pytest.mark.parametrize("disease", SUPPORTED_DISEASES)
    def test_model_predicts(self, disease):
        if self.is_ci():
            pytest.skip("Skipping prediction test in CI")
        from api.dependencies import load_tabular_model
        model, _ = load_tabular_model(disease)

        cfg = DISEASE_CONFIGS[disease]
        n_features = len(cfg["feature_columns"])
        X = pd.DataFrame(
            [np.zeros(n_features)],
            columns=cfg["feature_columns"]
        )

        prediction = model.predict(X)
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]

    @pytest.mark.parametrize("disease", SUPPORTED_DISEASES)
    def test_model_predict_proba(self, disease):
        if self.is_ci():
            pytest.skip("Skipping probability test in CI")
        from api.dependencies import load_tabular_model
        model, _ = load_tabular_model(disease)

        cfg = DISEASE_CONFIGS[disease]
        n_features = len(cfg["feature_columns"])
        X = pd.DataFrame(
            [np.zeros(n_features)],
            columns=cfg["feature_columns"]
        )

        proba = model.predict_proba(X)
        assert proba.shape == (1, 2)
        assert abs(proba[0].sum() - 1.0) < 1e-5, "Probabilities should sum to 1"


class TestMetricsIntegrity:
    """Validate metrics.json files have correct structure."""

    @pytest.mark.parametrize("disease", SUPPORTED_DISEASES)
    def test_tabular_metrics_structure(self, disease):
        import json
        metrics_path = SAVED_MODELS_DIR / disease / "metrics.json"
        with open(metrics_path) as f:
            data = json.load(f)

        # Should have model keys (xgboost/random_forest) or flat structure
        has_models = "xgboost" in data or "random_forest" in data
        has_flat = "accuracy" in data
        assert has_models or has_flat, f"Invalid metrics structure for {disease}"

    def test_xray_metrics_structure(self):
        import json
        metrics_path = SAVED_MODELS_DIR / "chest_xray" / "metrics.json"
        with open(metrics_path) as f:
            data = json.load(f)

        assert "accuracy" in data
        assert "f1_score" in data
        assert "confusion_matrix" in data
        assert "roc_auc" in data
        assert 0 <= data["accuracy"] <= 1
        assert 0 <= data["roc_auc"] <= 1
