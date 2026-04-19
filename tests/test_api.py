"""
API Endpoint Tests — Health, Predict, Metrics.
Tests all FastAPI routes with realistic payloads.
"""

import pytest


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "gpu_available" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_endpoints_info(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        assert "version" in data


class TestDiseasesEndpoint:
    """Tests for GET /predict/diseases."""

    def test_list_diseases_returns_200(self, client):
        response = client.get("/predict/diseases")
        assert response.status_code == 200

    def test_list_diseases_has_all(self, client, supported_diseases):
        data = client.get("/predict/diseases").json()
        assert "diseases" in data
        for disease in supported_diseases:
            assert disease in data["diseases"]

    def test_each_disease_has_required_fields(self, client):
        data = client.get("/predict/diseases").json()
        for key, info in data["diseases"].items():
            assert "display_name" in info
            assert "required_features" in info
            assert "positive_label" in info
            assert "negative_label" in info


class TestTabularPrediction:
    """Tests for POST /predict/tabular."""

    def test_diabetes_prediction(self, client, sample_diabetes_features):
        response = client.post("/predict/tabular", json={
            "disease": "diabetes",
            "features": sample_diabetes_features,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["disease"] == "diabetes"
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        assert data["prediction_label"] in ["Diabetic", "Non-Diabetic"]

    def test_heart_prediction(self, client, sample_heart_features):
        response = client.post("/predict/tabular", json={
            "disease": "heart",
            "features": sample_heart_features,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["disease"] == "heart"
        assert data["prediction"] in [0, 1]

    def test_invalid_disease_returns_400(self, client):
        response = client.post("/predict/tabular", json={
            "disease": "unknown_disease",
            "features": {"a": 1},
        })
        assert response.status_code == 400

    def test_missing_features_still_works(self, client):
        """Missing features are filled with 0, should still return prediction."""
        response = client.post("/predict/tabular", json={
            "disease": "diabetes",
            "features": {"Glucose": 148, "BMI": 33.6},
        })
        assert response.status_code == 200

    def test_shap_explanation_included(self, client, sample_diabetes_features):
        data = client.post("/predict/tabular", json={
            "disease": "diabetes",
            "features": sample_diabetes_features,
        }).json()
        # SHAP may or may not be available, but the field should exist
        assert "shap_explanation" in data


class TestXRayPrediction:
    """Tests for POST /predict/xray."""

    def test_xray_prediction_with_image(self, client, sample_xray_path):
        if sample_xray_path is None:
            pytest.skip("No test X-ray images found")

        with open(sample_xray_path, "rb") as f:
            response = client.post(
                "/predict/xray",
                files={"file": (sample_xray_path.name, f, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["Normal", "Pneumonia"]
        assert 0 <= data["probability"] <= 1

    def test_invalid_file_type_returns_400(self, client):
        response = client.post(
            "/predict/xray",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400


class TestMetricsEndpoint:
    """Tests for GET /metrics/."""

    def test_all_metrics_returns_200(self, client):
        response = client.get("/metrics/")
        assert response.status_code == 200
        data = response.json()
        assert "total_models" in data
        assert "diseases" in data

    def test_single_disease_metrics(self, client, supported_diseases):
        for disease in supported_diseases:
            response = client.get(f"/metrics/{disease}")
            if response.status_code == 200:
                data = response.json()
                assert data["disease"] == disease
                assert "metrics" in data

    def test_invalid_disease_returns_400(self, client):
        response = client.get("/metrics/fake_disease")
        assert response.status_code == 400
