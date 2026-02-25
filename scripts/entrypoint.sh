#!/bin/bash
# entrypoint.sh
set -e

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "  ██████╗██╗  ██╗██╗   ██╗██████╗ ███╗   ██╗"
echo " ██╔════╝██║  ██║██║   ██║██╔══██╗████╗  ██║"
echo " ██║     ███████║██║   ██║██████╔╝██╔██╗ ██║"
echo " ██║     ██╔══██║██║   ██║██╔══██╗██║╚██╗██║"
echo " ╚██████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║"
echo "  ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝"
echo -e "${NC}"
echo " ChurnGuard API — Customer Churn Prediction Platform"
echo " ─────────────────────────────────────────────────────"

MODEL_PATH="/app/models/preprocessor.joblib"
DATA_PATH="/app/data/processed/train.parquet"
METRICS_PATH="/app/reports/metrics.json"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    echo -e " ${GREEN}✓${NC} Modelo encontrado (${MODEL_SIZE}): $MODEL_PATH"
else
    echo -e " ${YELLOW}⚠${NC}  Modelo NO encontrado: $MODEL_PATH"
    echo -e "    La API iniciará en modo degradado (sin predicciones)."
fi

if [ -f "$DATA_PATH" ]; then
    echo -e " ${GREEN}✓${NC} Datos procesados encontrados: $(dirname $DATA_PATH)/"
else
    echo -e " ${YELLOW}⚠${NC}  Datos procesados NO encontrados."
fi

if [ -f "$METRICS_PATH" ]; then
    if command -v python3 &> /dev/null; then
        AUC=$(python3 -c "import json; d=json.load(open('$METRICS_PATH')); print(f\"{d['test_auc']:.3f}\")" 2>/dev/null || echo "N/A")
        F1=$(python3 -c "import json; d=json.load(open('$METRICS_PATH')); print(f\"{d['test_f1']:.3f}\")" 2>/dev/null || echo "N/A")
        echo -e " ${GREEN}✓${NC} Métricas del modelo — AUC: ${GREEN}${AUC}${NC} | F1: ${GREEN}${F1}${NC}"
    fi
fi

echo " ─────────────────────────────────────────────────────"
echo -e " ${BLUE}→${NC} MLflow URI: ${MLFLOW_TRACKING_URI:-http://localhost:5000}"
echo -e " ${BLUE}→${NC} Model:      ${MODEL_NAME:-churn-prediction-model} (${MODEL_STAGE:-Production})"
echo -e " ${BLUE}→${NC} Workers:    ${WORKERS:-2}"
echo " ─────────────────────────────────────────────────────"
echo ""

exec "$@"