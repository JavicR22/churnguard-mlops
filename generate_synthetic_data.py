"""
generate_synthetic_data.py
--------------------------
Genera clientes sintéticos con distribuciones SESGADAS respecto al dataset
de entrenamiento original (Telco Customer Churn).

El objetivo es forzar que Evidently detecte drift real en el endpoint
/monitoring/run, simulando un escenario donde el perfil de clientes
en producción cambió respecto a cuando se entrenó el modelo.

Cambios intencionales vs dataset original:
  - Más clientes con Fiber optic (40% → 70%)
  - Más contratos Month-to-month (55% → 80%)
  - Tenure más bajo (media 32 → 10 meses)
  - MonthlyCharges más alto (media 64 → 85 USD)
  - Menos clientes con servicios adicionales
  - Más SeniorCitizens (16% → 35%)

Uso:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --n 1000 --output data/processed/synthetic.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_PROCESSED = ROOT / "data" / "processed"

# ── Seed para reproducibilidad ────────────────────────────────────────────────
RNG = np.random.default_rng(seed=2026)


def generate_synthetic_customers(n: int = 500) -> pd.DataFrame:
    """
    Genera n clientes sintéticos con distribuciones sesgadas
    para simular drift en producción.
    """
    log.info(f"Generando {n} clientes sintéticos con drift intencional...")

    # ── Tenure: sesgado hacia clientes nuevos (churn más probable) ────────────
    # Original: media ~32 meses. Sintético: media ~10 meses
    tenure = RNG.exponential(scale=10, size=n).clip(0, 72).astype(int)

    # ── MonthlyCharges: más alto (clientes premium con más riesgo de churn) ───
    # Original: media ~64 USD. Sintético: media ~85 USD
    monthly_charges = RNG.normal(loc=85, scale=20, size=n).clip(20, 120)

    # ── TotalCharges: derivado de tenure * monthly (con algo de ruido) ────────
    total_charges = (tenure * monthly_charges * RNG.uniform(0.85, 1.15, size=n)).clip(0)

    # ── Género: balanceado igual que el original ──────────────────────────────
    gender = RNG.choice(["Male", "Female"], size=n, p=[0.505, 0.495])

    # ── SeniorCitizen: más seniors (16% → 35%) ────────────────────────────────
    senior_citizen = RNG.choice([0, 1], size=n, p=[0.65, 0.35])

    # ── Partner & Dependents: menos familias estables ─────────────────────────
    partner = RNG.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    dependents = RNG.choice(["Yes", "No"], size=n, p=[0.20, 0.80])

    # ── PhoneService: similar al original ────────────────────────────────────
    phone_service = RNG.choice(["Yes", "No"], size=n, p=[0.90, 0.10])

    # ── MultipleLines ─────────────────────────────────────────────────────────
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        RNG.choice(["Yes", "No"], size=n, p=[0.45, 0.55]),
    )

    # ── InternetService: mucho más Fiber optic (40% → 70%) ───────────────────
    # Este es el cambio más agresivo para forzar drift
    internet_service = RNG.choice(
        ["Fiber optic", "DSL", "No"],
        size=n,
        p=[0.70, 0.20, 0.10],
    )

    # ── Servicios adicionales: menos contratados (más riesgo) ─────────────────
    def internet_addon(prob_yes=0.15):
        return np.where(
            internet_service == "No",
            "No internet service",
            RNG.choice(["Yes", "No"], size=n, p=[prob_yes, 1 - prob_yes]),
        )

    online_security = internet_addon(0.15)
    online_backup = internet_addon(0.20)
    device_protection = internet_addon(0.18)
    tech_support = internet_addon(0.12)
    streaming_tv = internet_addon(0.40)
    streaming_movies = internet_addon(0.38)

    # ── Contract: casi todo Month-to-month (55% → 80%) ────────────────────────
    contract = RNG.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.80, 0.12, 0.08],
    )

    # ── PaperlessBilling: más digital ────────────────────────────────────────
    paperless_billing = RNG.choice(["Yes", "No"], size=n, p=[0.75, 0.25])

    # ── PaymentMethod: más Electronic check (correlación con churn) ──────────
    payment_method = RNG.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.50, 0.15, 0.20, 0.15],
    )

    # ── Feature engineering (igual que data_prep.py) ─────────────────────────
    mean_monthly = 64.76  # media del dataset de entrenamiento
    avg_monthly_spend = total_charges / (tenure + 1)
    monthly_charge_ratio = monthly_charges / mean_monthly
    num_services = (
        (phone_service == "Yes").astype(int)
        + (multiple_lines == "Yes").astype(int)
        + (internet_service != "No").astype(int)
        + (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    is_long_term = (tenure > 24).astype(int)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges.round(2),
            "TotalCharges": total_charges.round(2),
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "avg_monthly_spend": avg_monthly_spend.round(4),
            "monthly_charge_ratio": monthly_charge_ratio.round(4),
            "num_services": num_services,
            "is_long_term": is_long_term,
        }
    )

    log.info("Distribuciones del dataset sintético (vs original):")
    log.info(f"  tenure:           media={df['tenure'].mean():.1f}  (original ~32)")
    log.info(f"  MonthlyCharges:   media={df['MonthlyCharges'].mean():.1f}  (original ~64)")
    log.info(f"  Fiber optic:      {(df['InternetService']=='Fiber optic').mean():.1%}  (original ~40%)")
    log.info(f"  Month-to-month:   {(df['Contract']=='Month-to-month').mean():.1%}  (original ~55%)")
    log.info(f"  SeniorCitizen:    {df['SeniorCitizen'].mean():.1%}  (original ~16%)")
    log.info(f"  is_long_term:     {df['is_long_term'].mean():.1%}  (original ~45%)")

    return df


def main(n: int = 500, output: Path = None):
    output = output or (DATA_PROCESSED / "synthetic_production.parquet")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_customers(n)
    df.to_parquet(output, index=False)

    log.info(f"✅ Dataset sintético guardado en: {output}")
    log.info(f"   Filas: {len(df)} | Columnas: {len(df.columns)}")
    log.info("")
    log.info("Ahora corre el análisis de drift contra estos datos:")
    log.info("  python generate_synthetic_data.py  # ya lo hiciste")
    log.info("  curl -X POST http://localhost:8000/monitoring/run-synthetic")
    log.info("  O directamente:")
    log.info("  python -c \"")
    log.info("    from monitoring.drift_detector import run_drift_detection")
    log.info("    from pathlib import Path")
    log.info("    result = run_drift_detection(")
    log.info("        current_path=Path('data/processed/synthetic_production.parquet')")
    log.info("    )")
    log.info("    print(result)")
    log.info("  \"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera clientes sintéticos con drift intencional")
    parser.add_argument("--n", type=int, default=500, help="Número de clientes a generar")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida del parquet (default: data/processed/synthetic_production.parquet)",
    )
    args = parser.parse_args()
    main(n=args.n, output=args.output)