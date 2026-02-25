# ══════════════════════════════════════════════════════════════════════════════
# ChurnGuard API — Dockerfile
# Multi-stage build: builder + production
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Dependencias del sistema para compilar extensiones C (psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python en un directorio aislado
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: production ───────────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL org.opencontainers.image.title="ChurnGuard API" \
      org.opencontainers.image.description="Customer Churn Prediction Platform" \
      org.opencontainers.image.source="https://github.com/JavicR22/churnguard-mlops"

WORKDIR /app

# Librerías de sistema necesarias en runtime + git para DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes Python instalados desde el builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Código de la aplicación ───────────────────────────────────────────────────
COPY api/          ./api/
COPY src/          ./src/
COPY monitoring/   ./monitoring/
COPY params.yaml   .
COPY .dvc/         ./.dvc/

# ── Inicializar git y descargar artefactos con DVC ────────────────────────────
ARG GDRIVE_CREDENTIALS_DATA
RUN mkdir -p models data/processed reports monitoring/reports \
    && git init \
    && git config user.email "ci@churnguard.com" \
    && git config user.name "ChurnGuard CI" \
    && if [ -n "$GDRIVE_CREDENTIALS_DATA" ]; then \
        echo "$GDRIVE_CREDENTIALS_DATA" > /tmp/gdrive_creds.json \
        && dvc remote modify gdrive gdrive_service_account_json_file_path /tmp/gdrive_creds.json \
        && dvc pull --no-run-cache \
        && rm /tmp/gdrive_creds.json \
        && echo "✅ Artefactos descargados desde Google Drive" ; \
    else \
        echo "⚠ GDRIVE_CREDENTIALS_DATA no configurado — modo degradado" ; \
    fi

# ── Configuración final ───────────────────────────────────────────────────────
COPY scripts/entrypoint.sh /entrypoint.sh

# Usuario no-root (seguridad)
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
    && chmod +x /entrypoint.sh \
    && chown -R appuser:appuser /app /entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]