"""
test_model.py
-------------
Tests de calidad del modelo entrenado.
Verifica que el modelo guardado cumple los thresholds mínimos en el test set.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
PARAMS_FILE = ROOT / "params.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def pipeline():
    path = MODELS_DIR / "preprocessor.joblib"
    if not path.exists():
        pytest.skip("Modelo no disponible. Ejecuta: dvc repro train")
    return joblib.load(path)


@pytest.fixture(scope="module")
def test_data(params):
    path = DATA_PROCESSED / "test.parquet"
    if not path.exists():
        pytest.skip("test.parquet no disponible. Ejecuta: dvc repro data_prep")
    target = params["data"]["target_column"]
    df = pd.read_parquet(path)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


@pytest.fixture(scope="module")
def saved_metrics():
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        pytest.skip("metrics.json no disponible. Ejecuta: dvc repro evaluate")
    with open(path) as f:
        return json.load(f)


# ── TestModelArtifacts ────────────────────────────────────────────────────────
class TestModelArtifacts:
    def test_model_file_exists(self):
        assert (MODELS_DIR / "preprocessor.joblib").exists(), (
            "El archivo del modelo no existe. Ejecuta: dvc repro train"
        )

    def test_model_loads_without_error(self, pipeline):
        assert pipeline is not None

    def test_model_has_predict_method(self, pipeline):
        assert hasattr(pipeline, "predict"), "El modelo no tiene método predict"
        assert hasattr(pipeline, "predict_proba"), "El modelo no tiene método predict_proba"

    def test_model_is_sklearn_pipeline(self, pipeline):
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline), f"Se esperaba Pipeline, se obtuvo {type(pipeline)}"

    def test_metrics_file_exists(self):
        assert (REPORTS_DIR / "metrics.json").exists(), (
            "metrics.json no existe. Ejecuta: dvc repro evaluate"
        )

    def test_metrics_has_required_keys(self, saved_metrics):
        required = ["test_auc", "test_f1", "test_precision", "test_recall", "test_samples"]
        for key in required:
            assert key in saved_metrics, f"Métrica '{key}' no encontrada en metrics.json"

    def test_feature_names_json_exists(self):
        assert (DATA_PROCESSED / "feature_names.json").exists()


# ── TestModelQuality ──────────────────────────────────────────────────────────
class TestModelQuality:
    def test_model_meets_auc_threshold(self, pipeline, test_data, params):
        X, y = test_data
        y_proba = pipeline.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        threshold = params["thresholds"]["min_auc"]
        assert auc >= threshold, f"AUC {auc:.4f} < threshold mínimo {threshold}"

    def test_model_meets_f1_threshold(self, pipeline, test_data, params):
        X, y = test_data
        y_pred = pipeline.predict(X)
        f1 = f1_score(y, y_pred)
        threshold = params["thresholds"]["min_f1"]
        assert f1 >= threshold, f"F1 {f1:.4f} < threshold mínimo {threshold}"

    def test_model_meets_recall_threshold(self, pipeline, test_data, params):
        X, y = test_data
        y_pred = pipeline.predict(X)
        recall = recall_score(y, y_pred)
        threshold = params["thresholds"]["min_recall"]
        assert recall >= threshold, f"Recall {recall:.4f} < threshold mínimo {threshold}"

    def test_model_precision_positive(self, pipeline, test_data):
        X, y = test_data
        y_pred = pipeline.predict(X)
        precision = precision_score(y, y_pred)
        assert precision > 0.0, "Precision debe ser mayor a 0"

    def test_predict_proba_in_valid_range(self, pipeline, test_data):
        X, _ = test_data
        y_proba = pipeline.predict_proba(X)[:, 1]
        assert float(y_proba.min()) >= 0.0, "Probabilidad mínima debe ser >= 0"
        assert float(y_proba.max()) <= 1.0, "Probabilidad máxima debe ser <= 1"

    def test_predict_outputs_are_binary(self, pipeline, test_data):
        X, _ = test_data
        y_pred = pipeline.predict(X)
        unique = set(np.unique(y_pred))
        assert unique.issubset({0, 1}), f"Predicciones deben ser binarias, encontrado: {unique}"

    def test_predict_proba_sums_to_one(self, pipeline, test_data):
        X, _ = test_data
        y_proba = pipeline.predict_proba(X)
        sums = y_proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), "Las probabilidades deben sumar 1"

    def test_model_handles_single_sample(self, pipeline, test_data):
        X, _ = test_data
        single = X.iloc[[0]]
        pred = pipeline.predict(single)
        proba = pipeline.predict_proba(single)
        assert len(pred) == 1
        assert proba.shape == (1, 2)

    def test_model_handles_small_batch(self, pipeline, test_data):
        X, _ = test_data
        batch = X.iloc[:10]
        pred = pipeline.predict(batch)
        proba = pipeline.predict_proba(batch)
        assert len(pred) == 10
        assert proba.shape == (10, 2)

    def test_model_is_deterministic(self, pipeline, test_data):
        """El mismo input siempre produce el mismo output."""
        X, _ = test_data
        sample = X.iloc[:5]
        pred_1 = pipeline.predict_proba(sample)
        pred_2 = pipeline.predict_proba(sample)
        assert np.array_equal(pred_1, pred_2), "El modelo no es determinista"

    def test_metrics_match_saved_report(self, pipeline, test_data, saved_metrics):
        """Las métricas calculadas deben coincidir con las del reporte guardado."""
        X, y = test_data
        y_proba = pipeline.predict_proba(X)[:, 1]
        computed_auc = roc_auc_score(y, y_proba)
        saved_auc = saved_metrics["test_auc"]
        assert abs(computed_auc - saved_auc) < 0.01, (
            f"AUC calculado ({computed_auc:.4f}) difiere del guardado ({saved_auc:.4f})"
        )
