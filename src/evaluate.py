"""
evaluate.py
-----------
Stage 3 del pipeline DVC: evalúa el modelo en el conjunto de test.
Genera métricas finales y la confusion matrix.
Decide si el modelo pasa a Production en el Registry.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PARAMS_FILE = ROOT / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_test_data(params: dict):
    target = params["data"]["target_column"]
    test_df = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    log.info(f"Test cargado: {X_test.shape[0]} muestras")
    return X_test, y_test


def load_model():
    pipeline = joblib.load(MODELS_DIR / "preprocessor.joblib")
    log.info("Modelo cargado desde models/preprocessor.joblib")
    return pipeline


def evaluate(pipeline, X_test, y_test, params: dict) -> dict:
    """Evalúa el modelo en test set y retorna todas las métricas."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_samples": len(y_test),
        "churn_rate": float(y_test.mean()),
    }

    log.info("\n" + "=" * 50)
    log.info("MÉTRICAS EN TEST SET:")
    log.info(f"  AUC:       {metrics['test_auc']:.4f}")
    log.info(f"  F1:        {metrics['test_f1']:.4f}")
    log.info(f"  Precision: {metrics['test_precision']:.4f}")
    log.info(f"  Recall:    {metrics['test_recall']:.4f}")
    log.info("=" * 50)

    log.info("\nReporte detallado:")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics, y_pred, y_proba


def save_confusion_matrix(y_test, y_pred) -> None:
    """Genera y guarda la confusion matrix como imagen PNG."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — churnguard-mlops", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix guardada en: {output_path}")


def check_thresholds(metrics: dict, params: dict) -> bool:
    """Verifica si el modelo cumple los thresholds mínimos de calidad."""
    thresholds = params["thresholds"]
    passed = True
    checks = [
        ("AUC", metrics["test_auc"], thresholds["min_auc"]),
        ("F1", metrics["test_f1"], thresholds["min_f1"]),
        ("Recall", metrics["test_recall"], thresholds["min_recall"]),
    ]

    log.info("\nVERIFICACIÓN DE THRESHOLDS:")
    for name, value, threshold in checks:
        status = "✅ PASS" if value >= threshold else "❌ FAIL"
        log.info(f"  {name}: {value:.4f} >= {threshold} → {status}")
        if value < threshold:
            passed = False

    return passed


def promote_to_production(metrics: dict) -> None:
    """
    Promueve el modelo a Production en el MLflow Model Registry
    si supera todos los thresholds.
    """
    client = mlflow.MlflowClient()
    model_name = "churn-prediction-model"

    # Buscar el modelo en Staging
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        log.warning("No hay modelos en Staging para promover.")
        return

    latest_version = versions[0]

    # Promover a Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,  # Archiva la versión anterior
    )

    # Agregar descripción con métricas
    client.update_model_version(
        name=model_name,
        version=latest_version.version,
        description=(
            f"AUC: {metrics['test_auc']:.4f} | "
            f"F1: {metrics['test_f1']:.4f} | "
            f"Recall: {metrics['test_recall']:.4f}"
        ),
    )

    log.info(f"✅ Modelo v{latest_version.version} promovido a Production")


def main():
    params = load_params()
    X_test, y_test = load_test_data(params)
    pipeline = load_model()

    metrics, y_pred, y_proba = evaluate(pipeline, X_test, y_test, params)
    save_confusion_matrix(y_test, y_pred)

    # Guardar métricas en JSON (DVC las trackea como outputs)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Métricas guardadas en reports/metrics.json")

    # Verificar thresholds y promover si pasa
    passed = check_thresholds(metrics, params)

    if passed:
        try:
            promote_to_production(metrics)
        except Exception as e:
            log.warning(
                f"No se pudo promover en Registry (normal si MLflow no está corriendo): {e}"
            )
        log.info("✅ evaluate.py completado — modelo listo para producción")
    else:
        log.error("❌ El modelo no cumple los thresholds mínimos. No se promovió a Production.")
        sys.exit(1)


if __name__ == "__main__":
    main()
