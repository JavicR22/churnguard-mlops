"""
train.py
--------
Stage 2 del pipeline DVC: entrena 3 modelos y loggea todo en MLflow.
Al final, promueve el mejor modelo al MLflow Model Registry.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PARAMS_FILE = ROOT / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_data(params: dict):
    """Carga los datos procesados y separa features del target."""
    target = params["data"]["target_column"]

    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    log.info(f"Train cargado: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
    return X_train, y_train


def build_preprocessor(params: dict) -> ColumnTransformer:
    """
    Construye el preprocesador como un ColumnTransformer de sklearn.
    Esto asegura que el mismo preprocesamiento se aplique en training y en el API.
    """
    with open(DATA_PROCESSED / "feature_names.json") as f:
        feature_info = json.load(f)

    numeric_features = [
        f for f in feature_info["numeric_features"] if f in feature_info["all_features"]
    ]
    categorical_features = [
        f for f in feature_info["categorical_features"] if f in feature_info["all_features"]
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_models(params: dict) -> dict:
    """Define los 3 modelos a entrenar con sus hiperparámetros desde params.yaml."""
    lr_params = params["models"]["logistic_regression"]
    rf_params = params["models"]["random_forest"]
    xgb_params = params["models"]["xgboost"]

    return {
        "logistic_regression": LogisticRegression(
            C=lr_params["C"],
            max_iter=lr_params["max_iter"],
            random_state=lr_params["random_state"],
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            random_state=rf_params["random_state"],
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            random_state=xgb_params["random_state"],
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def train_and_log(model_name: str, model, preprocessor, X_train, y_train, params: dict) -> tuple:
    """
    Entrena un modelo y loggea todo en MLflow:
    - Parámetros del modelo
    - Métricas de entrenamiento
    - El modelo completo como artifact
    Retorna el run_id y el AUC para comparar modelos.
    """
    experiment_name = "churnguard-churn-prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        log.info(f"Entrenando {model_name}...")

        # Aplicar SMOTE para balancear clases (el dataset tiene ~26% churn)
        preprocessor_only = Pipeline([("preprocessor", preprocessor)])
        X_preprocessed = preprocessor_only.fit_transform(X_train)

        smote = SMOTE(random_state=params["data"]["random_state"])
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_train)

        # Entrenar solo el clasificador con datos balanceados
        model.fit(X_resampled, y_resampled)

        # Para el pipeline final, usamos el preprocessor ya fitted
        final_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor_only.named_steps["preprocessor"]),
                ("classifier", model),
            ]
        )

        # Predicciones en training (para referencia, no para evaluar overfitting)
        y_pred = final_pipeline.predict(X_train)
        y_proba = final_pipeline.predict_proba(X_train)[:, 1]

        # Métricas
        metrics = {
            "train_auc": roc_auc_score(y_train, y_proba),
            "train_f1": f1_score(y_train, y_pred),
            "train_precision": precision_score(y_train, y_pred),
            "train_recall": recall_score(y_train, y_pred),
        }

        # Log parámetros del modelo
        mlflow.log_params(model.get_params())
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("train_samples", len(X_train))

        # Log métricas
        mlflow.log_metrics(metrics)

        # Log del modelo completo
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=None,  # Se registra solo el ganador
        )

        log.info(
            f"  AUC: {metrics['train_auc']:.4f} | F1: {metrics['train_f1']:.4f} | Recall: {metrics['train_recall']:.4f}"
        )

        return run.info.run_id, metrics["train_auc"], final_pipeline


def register_best_model(best_run_id: str, model_name: str, best_pipeline) -> None:
    """
    Registra el mejor modelo en el MLflow Model Registry
    y lo promueve al stage 'Staging'.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="churn-prediction-model",
    )

    # Guardar el preprocessor localmente para el API
    joblib.dump(best_pipeline, MODELS_DIR / "preprocessor.joblib")

    log.info(
        f"✅ Modelo registrado como 'churn-prediction-model' versión {registered_model.version}"
    )
    log.info(
        "   Stage: None → Staging (promover a Production desde la UI de MLflow o con evaluate.py)"
    )


def main():
    params = load_params()
    X_train, y_train = load_data(params)
    preprocessor = build_preprocessor(params)
    models = get_models(params)

    results = {}
    pipelines = {}

    for model_name, model in models.items():
        run_id, auc_score, pipeline = train_and_log(
            model_name, model, preprocessor, X_train, y_train, params
        )
        results[model_name] = {"run_id": run_id, "auc": auc_score}
        pipelines[model_name] = pipeline

    # Seleccionar el mejor modelo por AUC
    best_model_name = max(results, key=lambda k: results[k]["auc"])
    best_result = results[best_model_name]

    log.info("\n" + "=" * 50)
    log.info("RESULTADOS DE ENTRENAMIENTO:")
    for name, result in results.items():
        marker = "  ← GANADOR" if name == best_model_name else ""
        log.info(f"  {name}: AUC = {result['auc']:.4f}{marker}")
    log.info("=" * 50)

    # Verificar threshold mínimo
    min_auc = params["thresholds"]["min_auc"]
    if best_result["auc"] < min_auc:
        log.error(
            f"❌ El mejor modelo no supera el AUC mínimo de {min_auc}. Revisa los datos y hiperparámetros."
        )
        sys.exit(1)

    register_best_model(best_result["run_id"], best_model_name, pipelines[best_model_name])
    log.info("✅ train.py completado exitosamente")


if __name__ == "__main__":
    main()
