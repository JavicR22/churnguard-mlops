# churnguard-mlops

> **Customer Churn Prediction Platform** — Pipeline MLOps end-to-end en producción.

![CI](https://github.com/tu-usuario/churnguard-mlops/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)

---

## ¿Qué hace este proyecto?

Predice si un cliente de telecomunicaciones va a cancelar su contrato (churn), demostrando el ciclo completo de MLOps:

```
RAW DATA → DVC → PREPROCESAMIENTO → ENTRENAMIENTO → MLFLOW REGISTRY
                                                          ↓
MONITOREO (Evidently) ← API FastAPI ← PRODUCCIÓN ← MODELO
                              ↑
                        CI/CD (GitHub Actions)
```

---

## Stack

| Capa | Tecnología |
|------|-----------|
| API Serving | FastAPI + Uvicorn |
| Experimentos | MLflow Tracking + Model Registry |
| Datos | DVC + Google Drive |
| Modelos | Scikit-learn · XGBoost |
| Monitoreo | Evidently AI |
| Infra | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| DB | PostgreSQL |

---

## Inicio rápido

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/tu-usuario/churnguard-mlops.git
cd churnguard-mlops

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tus credenciales
```

### 3. Descargar el dataset

Descarga el dataset de [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) y colócalo en:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 4. Inicializar DVC y correr el pipeline

```bash
dvc init
dvc repro          # Corre los 3 stages: data_prep → train → evaluate
```

### 5. Levantar todos los servicios con Docker

```bash
docker-compose up --build
```

Servicios disponibles:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

---

## Uso de la API

### Predecir churn de un cliente

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
  }'
```

**Respuesta:**
```json
{
  "churn_probability": 0.74,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "model_version": "v2.1.0"
}
```

---

## Estructura del proyecto

```
churnguard-mlops/
├── data/
│   ├── raw/              # Dataset original (versionado con DVC)
│   └── processed/        # Datos procesados (parquet)
├── src/
│   ├── data_prep.py      # Limpieza + feature engineering
│   ├── train.py          # Entrenamiento + MLflow tracking
│   └── evaluate.py       # Evaluación + promoción a Production
├── api/
│   ├── main.py           # FastAPI app
│   └── schemas.py        # Modelos Pydantic
├── monitoring/
│   └── drift_detector.py # Evidently AI
├── tests/
│   ├── test_data.py      # Tests del pipeline de datos
│   ├── test_model.py     # Tests de calidad del modelo
│   └── test_api.py       # Tests de endpoints
├── .github/workflows/
│   ├── ci.yml            # Tests en cada push
│   └── cd.yml            # Docker build en merge a main
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml              # Pipeline DVC
├── params.yaml           # Hiperparámetros versionados
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## Autor

Construido como portafolio de MLOps. Stack: FastAPI + MLflow + DVC + Docker + GitHub Actions.
