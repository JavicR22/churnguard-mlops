"""
drift_detector.py
-----------------
Detección de data drift con Evidently AI.
Compara el dataset de referencia (train) con datos nuevos (test o producción).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MONITORING_REPORTS = ROOT / "monitoring" / "reports"

# Features por tipo (espejo de params.yaml)
_NUMERIC = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "monthly_charge_ratio",
    "num_services", "is_long_term",
]
_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def _load_parquet(path: Path, drop_target: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if drop_target and "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df


def run_drift_detection(
    reference_path: Optional[Path] = None,
    current_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_sample: int = 500,
) -> dict:
    """
    Ejecuta el análisis de drift entre referencia y datos actuales.

    Args:
        reference_path: Ruta al parquet de referencia (train). Default: data/processed/train.parquet
        current_path:   Ruta al parquet actual (test). Default: data/processed/test.parquet
        output_dir:     Directorio donde guardar los reportes.
        n_sample:       Máximo de filas del dataset actual a analizar.

    Returns:
        dict con el resumen del drift.
    """
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    reference_path = reference_path or (DATA_PROCESSED / "train.parquet")
    current_path = current_path or (DATA_PROCESSED / "test.parquet")
    output_dir = output_dir or MONITORING_REPORTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Cargando referencia: {reference_path}")
    ref_df = _load_parquet(reference_path)

    log.info(f"Cargando datos actuales: {current_path}")
    cur_df = _load_parquet(current_path)

    # Muestreo para no sobrecargar el análisis
    if len(cur_df) > n_sample:
        cur_df = cur_df.sample(n=n_sample, random_state=42)
        log.info(f"Datos actuales muestreados a {n_sample} filas.")

    # Columnas comunes disponibles
    common_cols = [c for c in ref_df.columns if c in cur_df.columns]
    ref_df = ref_df[common_cols]
    cur_df = cur_df[common_cols]

    numeric_features = [f for f in _NUMERIC if f in common_cols]
    categorical_features = [f for f in _CATEGORICAL if f in common_cols]

    column_mapping = ColumnMapping(
        target=None,
        numerical_features=numeric_features,
        categorical_features=categorical_features,
    )

    log.info("Ejecutando reporte de drift...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )

    # Guardar HTML
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))
    log.info(f"Reporte HTML guardado en: {html_path}")

    # Extraer resumen
    report_dict = report.as_dict()
    summary = _extract_summary(report_dict, str(html_path))

    # Guardar JSON
    json_path = output_dir / "drift_report.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Resumen JSON guardado en: {json_path}")

    return summary


def _extract_summary(report_dict: dict, html_path: str) -> dict:
    """Extrae las métricas clave del reporte de Evidently."""
    metrics = report_dict.get("metrics", [])
    dataset_drift = False
    drifted_features = 0
    total_features = 0
    drift_share = 0.0

    for metric in metrics:
        result = metric.get("result", {})
        if "dataset_drift" in result:
            dataset_drift = bool(result["dataset_drift"])
            drifted_features = int(result.get("number_of_drifted_columns", 0))
            total_features = int(result.get("number_of_columns", 0))
            drift_share = float(result.get("share_of_drifted_columns", 0.0))
            break

    return {
        "status": "drift_detected" if dataset_drift else "no_drift",
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "total_features": total_features,
        "drift_share": round(drift_share, 4),
        "report_path": html_path,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_drift_detection()
    print("\n" + "=" * 50)
    print("RESUMEN DE DRIFT:")
    print(f"  Dataset drift: {result['dataset_drift']}")
    print(f"  Features con drift: {result['drifted_features']}/{result['total_features']}")
    print(f"  Drift share: {result['drift_share']:.2%}")
    print(f"  Reporte: {result['report_path']}")
    print("=" * 50)
    sys.exit(0 if not result["dataset_drift"] else 1)
