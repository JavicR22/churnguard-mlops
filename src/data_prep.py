"""
data_prep.py
------------
Stage 1 del pipeline DVC: limpieza, feature engineering y split de datos.
Lee el dataset crudo de Kaggle y genera los archivos train/test listos para entrenamiento.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Rutas del proyecto
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
PARAMS_FILE = ROOT / "params.yaml"


def load_params() -> dict:
    """Carga los hiperparámetros desde params.yaml."""
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_data(path: Path) -> pd.DataFrame:
    """Carga el dataset crudo."""
    log.info(f"Cargando dataset desde: {path}")
    df = pd.read_csv(path)
    log.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza del dataset:
    - Convierte TotalCharges a numérico (tiene espacios vacíos)
    - Elimina filas con nulos
    - Convierte la columna target a binario
    """
    log.info("Iniciando limpieza de datos...")

    # TotalCharges tiene espacios vacíos en lugar de NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Registrar cuántas filas tienen nulos antes de eliminarlas
    nulos = df.isnull().sum().sum()
    log.info(f"Valores nulos encontrados: {nulos}")

    df = df.dropna()
    log.info(f"Filas después de eliminar nulos: {df.shape[0]}")

    # Eliminar customerID — no es una feature útil
    df = df.drop(columns=["customerID"])

    # Convertir target a binario: Yes → 1, No → 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    churn_rate = df["Churn"].mean()
    log.info(f"Tasa de churn en el dataset: {churn_rate:.2%}")

    return df


def feature_engineering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Feature engineering:
    - Crea nuevas features derivadas
    - Estandariza valores categóricos
    """
    log.info("Aplicando feature engineering...")

    # Feature: costo promedio por mes de permanencia
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Feature: ratio de costo mensual vs promedio del dataset
    mean_monthly = df["MonthlyCharges"].mean()
    df["monthly_charge_ratio"] = df["MonthlyCharges"] / mean_monthly

    # Feature: número de servicios contratados
    servicios = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[servicios].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
        axis=1,
    )

    # Feature: cliente de largo plazo (más de 24 meses)
    df["is_long_term"] = (df["tenure"] > 24).astype(int)

    log.info(f"Features después de engineering: {df.shape[1]} columnas")
    return df


def split_and_save(df: pd.DataFrame, params: dict) -> None:
    """
    Divide el dataset en train/test y guarda como parquet.
    También guarda la lista de feature names para usarla en el API.
    """
    target = params["data"]["target_column"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Mantener proporción de churn en ambos splits
    )

    log.info(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")
    log.info(f"Churn en train: {y_train.mean():.2%} | Churn en test: {y_test.mean():.2%}")

    # Guardar como parquet (más eficiente que CSV para ML)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[target] = y_train
    train_df.to_parquet(DATA_PROCESSED / "train.parquet", index=False)

    test_df = X_test.copy()
    test_df[target] = y_test
    test_df.to_parquet(DATA_PROCESSED / "test.parquet", index=False)

    # Guardar la lista de features para que el API sepa qué esperar
    feature_info = {
        "numeric_features": params["features"]["numeric"] + [
            "avg_monthly_spend", "monthly_charge_ratio", "num_services", "is_long_term"
        ],
        "categorical_features": params["features"]["categorical"],
        "all_features": list(X_train.columns),
        "target": target,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "churn_rate_train": float(y_train.mean()),
    }

    with open(DATA_PROCESSED / "feature_names.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    log.info(f"Archivos guardados en: {DATA_PROCESSED}")
    log.info("✅ data_prep.py completado exitosamente")


def main():
    params = load_params()
    df = load_data(DATA_RAW)
    df = clean_data(df)
    df = feature_engineering(df, params)
    split_and_save(df, params)


if __name__ == "__main__":
    main()
