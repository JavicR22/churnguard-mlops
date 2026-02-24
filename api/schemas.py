"""
schemas.py
----------
Modelos Pydantic v2 para request/response de la API de ChurnGuard.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_VALID_CONTRACTS = ["Month-to-month", "One year", "Two year"]
_VALID_INTERNET = ["DSL", "Fiber optic", "No"]
_VALID_PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_VALID_BINARY = ["Yes", "No"]
_VALID_LINES = ["Yes", "No", "No phone service"]
_VALID_ADDONS = ["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    # ── Numeric ──────────────────────────────────────────────────────────────
    tenure: int = Field(..., ge=0, le=100, description="Meses como cliente")
    MonthlyCharges: float = Field(..., ge=0, description="Cargo mensual en USD")
    TotalCharges: float = Field(..., ge=0, description="Total cargos históricos en USD")

    # ── Demográficos ─────────────────────────────────────────────────────────
    gender: str = Field(default="Male", description="Male o Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="1 si es senior, 0 si no")
    Partner: str = Field(default="No", description="Yes o No")
    Dependents: str = Field(default="No", description="Yes o No")

    # ── Servicios ────────────────────────────────────────────────────────────
    PhoneService: str = Field(default="Yes", description="Yes o No")
    MultipleLines: str = Field(default="No", description="Yes, No, No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, No")
    OnlineSecurity: str = Field(default="No", description="Yes, No, No internet service")
    OnlineBackup: str = Field(default="No", description="Yes, No, No internet service")
    DeviceProtection: str = Field(default="No", description="Yes, No, No internet service")
    TechSupport: str = Field(default="No", description="Yes, No, No internet service")
    StreamingTV: str = Field(default="No", description="Yes, No, No internet service")
    StreamingMovies: str = Field(default="No", description="Yes, No, No internet service")

    # ── Contrato ─────────────────────────────────────────────────────────────
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(default="Yes", description="Yes o No")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)",
    )

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        if v not in _VALID_CONTRACTS:
            raise ValueError(f"Contract debe ser uno de: {_VALID_CONTRACTS}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v: str) -> str:
        if v not in _VALID_INTERNET:
            raise ValueError(f"InternetService debe ser uno de: {_VALID_INTERNET}")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        if v not in _VALID_PAYMENT:
            raise ValueError(f"PaymentMethod debe ser uno de: {_VALID_PAYMENT}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError("gender debe ser Male o Female")
        return v

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_binary(cls, v: str) -> str:
        if v not in _VALID_BINARY:
            raise ValueError(f"El campo debe ser Yes o No, recibido: {v!r}")
        return v

    @field_validator("MultipleLines")
    @classmethod
    def validate_lines(cls, v: str) -> str:
        if v not in _VALID_LINES:
            raise ValueError(f"MultipleLines debe ser uno de: {_VALID_LINES}")
        return v

    @field_validator(
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    @classmethod
    def validate_addons(cls, v: str) -> str:
        if v not in _VALID_ADDONS:
            raise ValueError(f"El campo debe ser uno de: {_VALID_ADDONS}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool
    risk_level: RiskLevel
    model_version: str
    model_stage: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_version: str
    model_stage: str
    model_source: str
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    version: str
    stage: str
    source: str
    feature_count: int
    train_samples: int
    test_samples: int
    churn_rate_train: float
    loaded_at: str


class MetricsResponse(BaseModel):
    test_auc: float
    test_f1: float
    test_precision: float
    test_recall: float
    test_samples: int
    churn_rate: float


class DriftSummary(BaseModel):
    status: str
    dataset_drift: bool
    drifted_features: int
    total_features: int
    drift_share: float
    report_path: Optional[str] = None
    timestamp: str
