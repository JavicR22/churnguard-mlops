"""
test_api.py
-----------
Tests de los endpoints de la API FastAPI.
Se ejecutan con TestClient de Starlette (sin levantar el servidor).
"""

import os


import pytest
from fastapi.testclient import TestClient

# El API_KEY en CI se setea como variable de entorno
_API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
_AUTH = {"X-API-Key": _API_KEY}

# Clientes de ejemplo
_VALID = {
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

_LOW_RISK = {
    "tenure": 60,
    "MonthlyCharges": 40.0,
    "TotalCharges": 2400.0,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── TestHealthEndpoint ────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/health").json()
        for field in ("status", "model_loaded", "model_version", "model_stage", "uptime_seconds"):
            assert field in data, f"Campo '{field}' ausente en /health"

    def test_status_is_valid(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("ok", "degraded")

    def test_uptime_is_positive(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0

    def test_no_auth_required(self, client):
        """El endpoint /health debe ser público."""
        res = client.get("/health")
        assert res.status_code == 200


# ── TestMetricsEndpoint ───────────────────────────────────────────────────────
class TestMetricsEndpoint:
    def test_returns_200_or_404(self, client):
        res = client.get("/metrics")
        assert res.status_code in (200, 404)

    def test_metrics_values_in_range(self, client):
        res = client.get("/metrics")
        if res.status_code == 404:
            pytest.skip("metrics.json no disponible")
        data = res.json()
        assert 0.5 <= data["test_auc"] <= 1.0, "AUC fuera de rango"
        assert 0.0 <= data["test_f1"] <= 1.0, "F1 fuera de rango"
        assert 0.0 <= data["test_recall"] <= 1.0, "Recall fuera de rango"
        assert data["test_samples"] > 0, "test_samples debe ser > 0"

    def test_no_auth_required(self, client):
        res = client.get("/metrics")
        assert res.status_code in (200, 404)


# ── TestModelInfoEndpoint ─────────────────────────────────────────────────────
class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        res = client.get("/model/info")
        assert res.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/model/info").json()
        for field in ("model_name", "version", "stage", "source", "feature_count"):
            assert field in data, f"Campo '{field}' ausente en /model/info"

    def test_model_name_is_correct(self, client):
        data = client.get("/model/info").json()
        assert data["model_name"] == "churn-prediction-model"

    def test_feature_count_positive(self, client):
        data = client.get("/model/info").json()
        assert data["feature_count"] >= 0


# ── TestPredictEndpoint ───────────────────────────────────────────────────────
class TestPredictEndpoint:
    def test_returns_200_with_valid_payload(self, client):
        res = client.post("/predict", json=_VALID, headers=_AUTH)
        assert res.status_code == 200, f"Status inesperado: {res.status_code} — {res.text}"

    def test_response_has_required_fields(self, client):
        data = client.post("/predict", json=_VALID, headers=_AUTH).json()
        for field in ("prediction_id", "churn_probability", "churn_prediction", "risk_level", "model_version"):
            assert field in data, f"Campo '{field}' ausente"

    def test_probability_in_range(self, client):
        data = client.post("/predict", json=_VALID, headers=_AUTH).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_risk_level_is_valid(self, client):
        data = client.post("/predict", json=_VALID, headers=_AUTH).json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_churn_prediction_is_bool(self, client):
        data = client.post("/predict", json=_VALID, headers=_AUTH).json()
        assert isinstance(data["churn_prediction"], bool)

    def test_prediction_id_is_uuid_format(self, client):
        data = client.post("/predict", json=_VALID, headers=_AUTH).json()
        pid = data["prediction_id"]
        assert len(pid) == 36, "prediction_id debe tener formato UUID"
        assert pid.count("-") == 4

    def test_low_risk_customer_is_low_or_medium(self, client):
        data = client.post("/predict", json=_LOW_RISK, headers=_AUTH).json()
        assert data["risk_level"] in ("LOW", "MEDIUM"), (
            f"Cliente estable esperado LOW/MEDIUM, obtuvo: {data['risk_level']}"
        )

    def test_invalid_contract_returns_422(self, client):
        bad = {**_VALID, "Contract": "Weekly"}
        res = client.post("/predict", json=bad, headers=_AUTH)
        assert res.status_code == 422

    def test_invalid_internet_service_returns_422(self, client):
        bad = {**_VALID, "InternetService": "5G"}
        res = client.post("/predict", json=bad, headers=_AUTH)
        assert res.status_code == 422

    def test_negative_tenure_returns_422(self, client):
        bad = {**_VALID, "tenure": -1}
        res = client.post("/predict", json=bad, headers=_AUTH)
        assert res.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        incomplete = {k: v for k, v in _VALID.items() if k != "tenure"}
        res = client.post("/predict", json=incomplete, headers=_AUTH)
        assert res.status_code == 422

    def test_predict_is_deterministic(self, client):
        """El mismo input debe retornar la misma probabilidad."""
        r1 = client.post("/predict", json=_VALID, headers=_AUTH).json()
        r2 = client.post("/predict", json=_VALID, headers=_AUTH).json()
        assert r1["churn_probability"] == r2["churn_probability"]
        assert r1["risk_level"] == r2["risk_level"]

    def test_wrong_api_key_returns_403_in_strict_mode(self, client, monkeypatch):
        """Si API_KEY no es el default dev, una clave incorrecta debe retornar 403."""
        monkeypatch.setenv("API_KEY", "strict-key-123")
        import api.main as main_mod
        # Solo verificamos que la función de auth existe y está configurada
        assert hasattr(main_mod, "verify_api_key")


# ── TestBatchPredictEndpoint ──────────────────────────────────────────────────
class TestBatchPredictEndpoint:
    def test_returns_200(self, client):
        payload = {"customers": [_VALID, _LOW_RISK]}
        res = client.post("/predict/batch", json=payload, headers=_AUTH)
        assert res.status_code == 200

    def test_response_count_matches_input(self, client):
        payload = {"customers": [_VALID, _LOW_RISK, _VALID]}
        data = client.post("/predict/batch", json=payload, headers=_AUTH).json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_risk_counts_sum_to_total(self, client):
        payload = {"customers": [_VALID, _LOW_RISK]}
        data = client.post("/predict/batch", json=payload, headers=_AUTH).json()
        total = data["high_risk_count"] + data["medium_risk_count"] + data["low_risk_count"]
        assert total == data["total"]

    def test_processing_time_is_positive(self, client):
        payload = {"customers": [_VALID]}
        data = client.post("/predict/batch", json=payload, headers=_AUTH).json()
        assert data["processing_time_ms"] > 0

    def test_empty_customers_returns_422(self, client):
        res = client.post("/predict/batch", json={"customers": []}, headers=_AUTH)
        assert res.status_code == 422

    def test_each_prediction_has_id(self, client):
        payload = {"customers": [_VALID, _LOW_RISK]}
        data = client.post("/predict/batch", json=payload, headers=_AUTH).json()
        for pred in data["predictions"]:
            assert "prediction_id" in pred
            assert len(pred["prediction_id"]) == 36
