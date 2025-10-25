# ---------------------------------------------
# Variables
# ---------------------------------------------
DOCKER_IMAGE = credit-scoring-api:latest
MODEL_PATH   = models/latest_model.pkl
MLFLOW_URI   = http://localhost:5000

# ---------------------------------------------
# Commandes par défaut
# ---------------------------------------------
.PHONY: all
all: build test

# ---------------------------------------------
# 1. Installation dépendances
# ---------------------------------------------
.PHONY: install
install:
	@echo "💠 Installation des dépendances..."
	pip install --upgrade pip
	pip install -r requirements.txt

# ---------------------------------------------
# 2. Build Docker
# ---------------------------------------------
.PHONY: build
build:
	@echo "💠 Build de l'image Docker..."
	docker build -t $(DOCKER_IMAGE) .

# ---------------------------------------------
# 3. Tests unitaires
# ---------------------------------------------
.PHONY: test
test:
	@echo "💠 Lancement des tests unitaires..."
	pytest -v tests/

# ---------------------------------------------
# 4. Export du modèle
# ---------------------------------------------
.PHONY: export_model
export_model:
	@echo "💠 Export du modèle..."
	python src/export_model.py

# ---------------------------------------------
# 5. Enregistrement MLflow
# ---------------------------------------------
.PHONY: register_mlflow
register_mlflow:
	@echo "💠 Enregistrement du modèle dans MLflow..."
	python src/register_mlflow.py --model-path $(MODEL_PATH) --mlflow-uri $(MLFLOW_URI)

# ---------------------------------------------
# 6. Lancement API
# ---------------------------------------------
.PHONY: run_api
run_api:
	@echo "💠 Démarrage de l'API FastAPI..."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------
# 7. Lancement monitoring/dashboards
# ---------------------------------------------
.PHONY: monitor
monitor:
	@echo "💠 Génération du dashboard Evidently..."
	python src/generate_report.py

# ---------------------------------------------
# 8. Déploiement Docker (serveur local)
# ---------------------------------------------
.PHONY: deploy
deploy:
	@echo "💠 Déploiement de l'image Docker..."
	docker stop credit_scoring || true
	docker rm credit_scoring || true
	docker run -d --name credit_scoring -p 8000:8000 $(DOCKER_IMAGE)

# ---------------------------------------------
# 9. Nettoyage
# ---------------------------------------------
.PHONY: clean
clean:
	@echo "💠 Nettoyage des fichiers temporaires..."
	rm -rf __pycache__ .pytest_cache logs dashboards/*.html models/*.pkl
