## ═══════════════════════════════════════════════════════════════════════════
## ChurnGuard MLOps — Makefile
## Uso: make <target>
## ═══════════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Colores
BOLD  := \033[1m
GREEN := \033[32m
CYAN  := \033[36m
RESET := \033[0m

# Configuración
PYTHON   := python
PYTEST   := pytest
UVICORN  := uvicorn
DC       := docker compose
DC_PROD  := docker compose -f docker-compose.yml
DC_DEV   := docker compose -f docker-compose.yml -f docker-compose.override.yml

# ── Ayuda ──────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Muestra esta ayuda
	@echo ""
	@echo "$(BOLD)ChurnGuard MLOps$(RESET) — Comandos disponibles"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup inicial ──────────────────────────────────────────────────────────────
.PHONY: install
install: ## Crea el entorno virtual e instala todas las dependencias
	$(PYTHON) -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencias instaladas. Activa el entorno: source venv/bin/activate$(RESET)"

.PHONY: setup
setup: install pipeline ## Setup completo: instala deps + genera todos los artefactos ML
	@echo "$(GREEN)✓ Setup completo. Para iniciar: make up$(RESET)"

# ── Pipeline DVC ───────────────────────────────────────────────────────────────
.PHONY: pipeline
pipeline: ## Ejecuta el pipeline DVC completo (data_prep → train → evaluate)
	@echo "Ejecutando pipeline DVC..."
	dvc repro
	@echo "$(GREEN)✓ Pipeline completado. Artefactos generados en models/ data/processed/ reports/$(RESET)"

.PHONY: pipeline-force
pipeline-force: ## Fuerza re-ejecución de todo el pipeline aunque nada haya cambiado
	dvc repro --force

.PHONY: pipeline-status
pipeline-status: ## Muestra el estado del pipeline DVC
	dvc status
	dvc dag

# ── Tests ──────────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Ejecuta todos los tests con cobertura
	$(PYTEST) tests/ -v --tb=short \
		--cov=src --cov=api \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

.PHONY: test-fast
test-fast: ## Ejecuta tests sin cobertura (más rápido)
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-api
test-api: ## Solo tests de la API
	$(PYTEST) tests/test_api.py -v --tb=short

.PHONY: test-model
test-model: ## Solo tests del modelo
	$(PYTEST) tests/test_model.py -v --tb=short

.PHONY: test-data
test-data: ## Solo tests del pipeline de datos
	$(PYTEST) tests/test_data.py -v --tb=short

# ── Linting ────────────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Verifica estilo (flake8) y formato (black)
	flake8 src/ api/ tests/ monitoring/ --max-line-length=100 --ignore=E203,W503,E501
	black --check --line-length=100 src/ api/ tests/ monitoring/
	@echo "$(GREEN)✓ Lint OK$(RESET)"

.PHONY: format
format: ## Formatea el código automáticamente con black
	black --line-length=100 src/ api/ tests/ monitoring/
	@echo "$(GREEN)✓ Código formateado$(RESET)"

# ── API local ──────────────────────────────────────────────────────────────────
.PHONY: run
run: ## Inicia la API localmente con hot-reload (sin Docker)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: run-prod
run-prod: ## Inicia la API en modo producción (sin Docker, 2 workers)
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Docker ─────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build: ## Construye las imágenes Docker (API + MLflow)
	$(DC) build --no-cache
	@echo "$(GREEN)✓ Imágenes Docker construidas$(RESET)"

.PHONY: docker-build-api
docker-build-api: ## Construye solo la imagen de la API
	docker build -t churnguard-api:local .

.PHONY: up
up: ## Inicia el stack completo en background (postgres + mlflow + api)
	$(DC_PROD) up -d
	@echo ""
	@echo "$(GREEN)✓ Stack iniciado:$(RESET)"
	@echo "   API:    http://localhost:8000"
	@echo "   MLflow: http://localhost:5000"
	@echo "   Logs:   make logs"

.PHONY: up-dev
up-dev: ## Inicia la API con hot-reload (modo desarrollo)
	$(DC_DEV) up api
	@echo "$(GREEN)✓ API en modo dev: http://localhost:8000$(RESET)"

.PHONY: up-full
up-full: ## Inicia el stack completo en foreground (ver logs directamente)
	$(DC_PROD) up

.PHONY: down
down: ## Detiene todos los servicios
	$(DC) down
	@echo "$(GREEN)✓ Servicios detenidos$(RESET)"

.PHONY: down-volumes
down-volumes: ## Detiene servicios y ELIMINA los volúmenes (borra datos de postgres + mlflow)
	$(DC) down -v
	@echo "$(GREEN)✓ Servicios detenidos y volúmenes eliminados$(RESET)"

.PHONY: restart
restart: ## Reinicia todos los servicios
	$(DC) restart

.PHONY: restart-api
restart-api: ## Reinicia solo la API
	$(DC) restart api

.PHONY: logs
logs: ## Sigue los logs de todos los servicios
	$(DC) logs -f

.PHONY: logs-api
logs-api: ## Sigue los logs de la API
	$(DC) logs -f api

.PHONY: ps
ps: ## Muestra el estado de los contenedores
	$(DC) ps

# ── Monitoreo ──────────────────────────────────────────────────────────────────
.PHONY: drift
drift: ## Ejecuta el análisis de drift (requiere API corriendo)
	curl -s -X POST http://localhost:8000/monitoring/run \
		-H "X-API-Key: $${API_KEY:-dev-key-change-in-production}" \
		| python3 -m json.tool

# ── Limpieza ───────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Limpia archivos temporales y cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	@echo "$(GREEN)✓ Limpieza completada$(RESET)"

.PHONY: clean-all
clean-all: clean down-volumes ## Limpieza total (incluyendo artefactos ML y volúmenes Docker)
	rm -rf models/preprocessor.joblib
	rm -rf data/processed/*.parquet data/processed/feature_names.json
	rm -rf reports/metrics.json reports/confusion_matrix.png
	rm -rf monitoring/reports/
	@echo "$(GREEN)✓ Limpieza total completada. Ejecuta 'make setup' para reconfigurar$(RESET)"

# ── Utilidades ─────────────────────────────────────────────────────────────────
.PHONY: health
health: ## Verifica el estado de la API
	@curl -s http://localhost:8000/health | python3 -m json.tool

.PHONY: metrics
metrics: ## Muestra las métricas del modelo
	@curl -s http://localhost:8000/metrics | python3 -m json.tool

.PHONY: predict-example
predict-example: ## Envía una predicción de ejemplo
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-H "X-API-Key: $${API_KEY:-dev-key-change-in-production}" \
		-d '{"tenure":24,"MonthlyCharges":65.5,"TotalCharges":1572.0,"Contract":"Month-to-month","InternetService":"Fiber optic","PaymentMethod":"Electronic check"}' \
		| python3 -m json.tool
