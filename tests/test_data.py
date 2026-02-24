"""
test_data.py
------------
Tests de validación del pipeline de datos.
Verifica que data_prep.py produce los outputs correctos.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
PARAMS_FILE = ROOT / "params.yaml"


@pytest.fixture
def params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture
def train_df():
    path = DATA_PROCESSED / "train.parquet"
    if not path.exists():
        pytest.skip("Datos procesados no disponibles. Ejecuta: dvc repro data_prep")
    return pd.read_parquet(path)


@pytest.fixture
def test_df():
    path = DATA_PROCESSED / "test.parquet"
    if not path.exists():
        pytest.skip("Datos procesados no disponibles. Ejecuta: dvc repro data_prep")
    return pd.read_parquet(path)


@pytest.fixture
def feature_info():
    path = DATA_PROCESSED / "feature_names.json"
    if not path.exists():
        pytest.skip("feature_names.json no disponible.")
    with open(path) as f:
        return json.load(f)


class TestDataSchema:
    def test_train_has_target_column(self, train_df, params):
        target = params["data"]["target_column"]
        assert target in train_df.columns, f"Columna target '{target}' no encontrada en train"

    def test_test_has_target_column(self, test_df, params):
        target = params["data"]["target_column"]
        assert target in test_df.columns

    def test_target_is_binary(self, train_df, params):
        target = params["data"]["target_column"]
        unique_values = set(train_df[target].unique())
        assert unique_values == {
            0,
            1,
        }, f"Target debe ser binario, encontrado: {unique_values}"

    def test_no_null_values_in_train(self, train_df):
        null_counts = train_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        assert len(null_columns) == 0, f"Columnas con nulos: {null_columns.to_dict()}"

    def test_no_null_values_in_test(self, test_df):
        null_counts = test_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        assert len(null_columns) == 0, f"Columnas con nulos: {null_columns.to_dict()}"


class TestDataSplit:
    def test_train_test_ratio(self, train_df, test_df, params):
        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total
        expected = params["data"]["test_size"]
        assert (
            abs(test_ratio - expected) < 0.02
        ), f"Ratio test esperado: {expected}, obtenido: {test_ratio:.3f}"

    def test_train_has_enough_samples(self, train_df):
        assert len(train_df) > 100, "Train debe tener más de 100 muestras"

    def test_churn_rate_reasonable(self, train_df, params):
        target = params["data"]["target_column"]
        churn_rate = train_df[target].mean()
        assert 0.10 <= churn_rate <= 0.50, f"Churn rate inusual: {churn_rate:.2%}"


class TestFeatureEngineering:
    def test_engineered_features_exist(self, train_df):
        expected_features = [
            "avg_monthly_spend",
            "monthly_charge_ratio",
            "num_services",
            "is_long_term",
        ]
        for feat in expected_features:
            assert feat in train_df.columns, f"Feature '{feat}' no encontrada"

    def test_num_services_range(self, train_df):
        assert train_df["num_services"].min() >= 0
        assert train_df["num_services"].max() <= 9

    def test_is_long_term_binary(self, train_df):
        assert set(train_df["is_long_term"].unique()).issubset({0, 1})

    def test_feature_names_json_structure(self, feature_info):
        required_keys = [
            "numeric_features",
            "categorical_features",
            "all_features",
            "target",
        ]
        for key in required_keys:
            assert key in feature_info, f"Key '{key}' no encontrada en feature_names.json"
