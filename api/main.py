"""
main.py
-------
FastAPI app de ChurnGuard.
Sirve predicciones de churn + dashboard de monitoreo con interfaz moderna.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftSummary,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskLevel,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MONITORING_DIR = ROOT / "monitoring" / "reports"

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-prediction-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
API_KEY_VALUE = os.getenv("API_KEY", "dev-key-change-in-production")

# ── Services const ────────────────────────────────────────────────────────────
_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
_NO_VALUES = {"No", "No internet service", "No phone service"}

# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {
    "pipeline": None,
    "version": "unknown",
    "stage": "unknown",
    "source": "none",
    "loaded_at": None,
    "mean_monthly": 64.76,
    "feature_info": None,
    "start_time": time.time(),
}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_feature_info()
    _load_mean_monthly()
    _load_model()
    log.info("✅ API lista.")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnGuard API",
    description="Predicción de churn de clientes de telecomunicaciones con MLOps.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Auth ──────────────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(_api_key_header)):
    """Valida el API key. En modo dev (key por defecto) no se exige."""
    if API_KEY_VALUE == "dev-key-change-in-production":
        return key or "dev"
    if not key or key != API_KEY_VALUE:
        raise HTTPException(status_code=403, detail="API key inválido o ausente.")
    return key


def _load_feature_info():
    path = DATA_PROCESSED / "feature_names.json"
    if path.exists():
        with open(path) as f:
            _state["feature_info"] = json.load(f)
        log.info("feature_names.json cargado.")
    else:
        log.warning("feature_names.json no encontrado — usando defaults.")


def _load_mean_monthly():
    path = DATA_PROCESSED / "train.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path, columns=["MonthlyCharges"])
            _state["mean_monthly"] = float(df["MonthlyCharges"].mean())
            log.info(f"mean_monthly cargado: {_state['mean_monthly']:.2f}")
        except Exception as e:
            log.warning(f"No se pudo calcular mean_monthly: {e}")
    else:
        log.warning(
            f"train.parquet no encontrado — usando mean_monthly={_state['mean_monthly']:.2f}"
        )


def _load_model():
    # 1. Intentar MLflow Registry
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        pipeline = mlflow.sklearn.load_model(model_uri)
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'", max_results=1, order_by=["version_number DESC"]
        )
        version = str(versions[0].version) if versions else "registry"
        _state.update(
            pipeline=pipeline, version=version, stage=MODEL_STAGE, source="mlflow"
        )
        log.info(f"Modelo cargado desde MLflow Registry: v{version} ({MODEL_STAGE})")
        return
    except Exception as e:
        log.warning(f"MLflow no disponible ({type(e).__name__}). Usando modelo local.")

    # 2. Fallback: archivo local
    local_path = MODELS_DIR / "preprocessor.joblib"
    if local_path.exists():
        pipeline = joblib.load(local_path)
        _state.update(
            pipeline=pipeline, version="local", stage="Production", source="local"
        )
        log.info("Modelo cargado desde models/preprocessor.joblib")
    else:
        log.error("❌ No se encontró ningún modelo. La API está degradada.")

    _state["loaded_at"] = datetime.utcnow().isoformat()


# ── Feature engineering ───────────────────────────────────────────────────────
def _build_features(customer: CustomerFeatures) -> pd.DataFrame:
    """Reproduce el feature engineering de data_prep.py para la inferencia."""
    d = customer.model_dump()
    mean_monthly = _state["mean_monthly"]

    d["avg_monthly_spend"] = d["TotalCharges"] / (d["tenure"] + 1)
    d["monthly_charge_ratio"] = d["MonthlyCharges"] / mean_monthly
    d["num_services"] = sum(1 for col in _SERVICE_COLS if d.get(col) not in _NO_VALUES)
    d["is_long_term"] = int(d["tenure"] > 24)

    return pd.DataFrame([d])


def _compute_risk(probability: float) -> RiskLevel:
    if probability < 0.35:
        return RiskLevel.LOW
    if probability < 0.60:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def _single_predict(customer: CustomerFeatures) -> PredictionResponse:
    pipeline = _state["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    df = _build_features(customer)
    proba = float(pipeline.predict_proba(df)[0, 1])
    pred = bool(pipeline.predict(df)[0])
    risk = _compute_risk(proba)

    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        risk_level=risk,
        model_version=_state["version"],
        model_stage=_state["stage"],
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Endpoints: público ────────────────────────────────────────────────────────
@app.get("/")
def dashboard():
    return FileResponse("api/templates/dashboard.html")


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    uptime = time.time() - _state["start_time"]
    return HealthResponse(
        status="ok" if _state["pipeline"] is not None else "degraded",
        model_loaded=_state["pipeline"] is not None,
        model_version=_state["version"],
        model_stage=_state["stage"],
        model_source=_state["source"],
        uptime_seconds=round(uptime, 1),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Status"])
async def get_metrics():
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json no encontrado. Ejecuta: dvc repro evaluate",
        )
    with open(path) as f:
        data = json.load(f)
    return MetricsResponse(**data)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Status"])
async def model_info():
    fi = _state.get("feature_info") or {}
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        version=_state["version"],
        stage=_state["stage"],
        source=_state["source"],
        feature_count=len(fi.get("all_features", [])),
        train_samples=fi.get("train_samples", 0),
        test_samples=fi.get("test_samples", 0),
        churn_rate_train=round(fi.get("churn_rate_train", 0.0), 4),
        loaded_at=_state.get("loaded_at") or datetime.utcnow().isoformat(),
    )


# ── Endpoints: predicción ─────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict(
    customer: CustomerFeatures,
    _key: str = Depends(verify_api_key),
):
    return _single_predict(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
async def predict_batch(
    request: BatchPredictionRequest,
    _key: str = Depends(verify_api_key),
):
    t0 = time.perf_counter()
    predictions = [_single_predict(c) for c in request.customers]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    high = sum(1 for p in predictions if p.risk_level == RiskLevel.HIGH)
    medium = sum(1 for p in predictions if p.risk_level == RiskLevel.MEDIUM)
    low = sum(1 for p in predictions if p.risk_level == RiskLevel.LOW)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ── Endpoints: monitoreo ──────────────────────────────────────────────────────
@app.get("/monitoring/drift", tags=["Monitoreo"])
async def get_drift_report():
    report_path = MONITORING_DIR / "drift_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Reporte de drift no disponible. POST /monitoring/run para generarlo.",
        )
    with open(report_path) as f:
        return JSONResponse(content=json.load(f))


@app.post("/monitoring/run", response_model=DriftSummary, tags=["Monitoreo"])
async def run_drift(
    _key: str = Depends(verify_api_key),
):
    try:
        from monitoring.drift_detector import run_drift_detection

        MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        summary = run_drift_detection(
            reference_path=DATA_PROCESSED / "train.parquet",
            current_path=DATA_PROCESSED / "test.parquet",
            output_dir=MONITORING_DIR,
        )
        return summary
    except Exception as e:
        log.error(f"Error en drift detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
