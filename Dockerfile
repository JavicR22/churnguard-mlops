# ══════════════════════════════════════════════════════════════════════════════
# ChurnGuard API — Dockerfile
# Multi-stage build: builder + production
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: production ───────────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL org.opencontainers.image.title="ChurnGuard API" \
      org.opencontainers.image.description="Customer Churn Prediction Platform" \
      org.opencontainers.image.source="https://github.com/JavicR22/churnguard-mlops"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Código de la aplicación ───────────────────────────────────────────────────
COPY api/              ./api/
COPY src/              ./src/
COPY monitoring/       ./monitoring/
COPY params.yaml       .

# ── Artefactos del modelo ─────────────────────────────────────────────────────
COPY models/           ./models/
COPY data/processed/   ./data/processed/
COPY reports/          ./reports/

# ── Configuración final ───────────────────────────────────────────────────────
COPY scripts/entrypoint.sh /entrypoint.sh

RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
    && mkdir -p monitoring/reports \
    && chmod +x /entrypoint.sh \
    && chown -R appuser:appuser /app /entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]