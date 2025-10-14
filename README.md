# Industrialisation-du-modele-de-scoring-credit => Scoring MLOps Platform

## 🎯 Objectif
Cette plateforme permet de **préparer, entraîner, déployer et monitorer des modèles de scoring crédit**. Elle inclut l’ingestion de données en streaming, l’entraînement multi-modèles, le suivi MLflow, l’industrialisation via API FastAPI/Docker et la surveillance de dérive via Evidently AI.

---

## 🗂️ Structure du projet

```bash
credit_scoring_mlops/
│
├── data/
│   ├── raw/                     # Données brutes (ingérées depuis Kafka / S3)
│   ├── processed/               # Données nettoyées et prêtes pour entraînement
│   ├── reference/               # Jeu de référence pour détection de dérive
│   └── new_data/                # Données récentes pour recalibrage périodique
│
├── notebooks/
│   ├── exploration.ipynb        # Analyse exploratoire initiale
│   ├── model_comparison.ipynb   # Tests de différents modèles (LR, XGBoost, NN)
│   └── monitoring_drift.ipynb   # Détection et visualisation des dérives Evidently
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_preprocessing/
│   │   ├── preprocessing.py     # Nettoyage, encodage, imputation, scaling
│   │   ├── feature_selection.py # Sélection des variables pertinentes
│   │   └── split_data.py        # Split train/validation/test
│   │
│   ├── training/
│   │   ├── train_lr.py          # Entraînement du modèle Logistic Regression
│   │   ├── train_xgb.py         # Entraînement XGBoost
│   │   ├── train_nn.py          # Entraînement du modèle neuronal
│   │   └── evaluate_models.py   # Comparaison AUC/F1 et sélection du meilleur modèle
│   │
│   ├── export/
│   │   ├── export_model.py      # Conversion modèle en ONNX / PMML
│   │   └── register_mlflow.py   # Enregistrement modèle final dans MLflow
│   │
│   ├── deployment/
│   │   ├── app/
│   │   │   ├── main.py          # API FastAPI : endpoint `/predict`
│   │   │   ├── schemas.py       # Schémas Pydantic pour les entrées utilisateur
│   │   │   └── utils.py         # Fonctions auxiliaires (chargement modèle, logs)
│   │   │
│   │   ├── Dockerfile           # Dockerfile pour containeriser l’API
│   │   ├── requirements.txt     # Dépendances API (FastAPI, ONNXRuntime…)
│   │   ├── entrypoint.sh        # Script de démarrage Docker (optionnel)
│   │   └── test_api.py          # Tests unitaires de l’API avec pytest
│   │
│   ├── monitoring/
│   │   ├── drift_detection.py   # Vérification automatique des dérives (Evidently)
│   │   ├── retrain_trigger.py   # Script de recalibrage périodique
│   │   └── generate_report.py   # Génération du dashboard HTML Evidently
│   │
│   └── utils/
│       ├── spark_session.py     # Initialisation Spark pour preprocessing
│       ├── config.yaml          # Paramètres globaux (chemins, hyperparams, seuils)
│       └── logger.py            # Configuration centralisée des logs
│
├── ci_cd/
│   ├── github_actions.yml       # Workflow GitHub Actions (build, test, deploy)
│   ├── docker-compose.yml       # Lancement multi-services (API + MLflow + monitoring)
│   └── Jenkinsfile              # (Optionnel) Pipeline Jenkins équivalent
│
├── models/
│   ├── xgb_credit_model.pkl     # Modèle XGBoost sauvegardé (Pickle)
│   ├── xgb_credit_model.onnx    # Modèle exporté en ONNX
│   ├── model_metadata.json      # Métadonnées du modèle (version, date, métriques)
│   └── mlruns/                  # Dossier de tracking MLflow (expériences)
│
├── tests/
│   ├── test_preprocessing.py    # Tests unitaires sur le preprocessing
│   ├── test_training.py         # Tests unitaires sur l'entraînement
│   ├── test_export.py           # Vérifie l’intégrité du modèle exporté
│   ├── test_api.py              # Tests sur les endpoints FastAPI
│   └── test_monitoring.py       # Tests de détection de dérive
│
├── dashboards/
│   ├── credit_model_monitoring.html  # Dashboard Evidently généré
│   └── drift_report.html             # Rapport de dérive complet
│
├── docker-compose.yml           # Orchestration (API + MLflow + PostgreSQL)
├── README.md                    # Documentation du projet complet
├── requirements.txt             # Dépendances Python globales
└── Makefile                     # Commandes automatiques (build, test, deploy)
```
---


**Description rapide :**

- **data/** : jeux de données bruts et transformés.  
- **notebooks/** : analyses exploratoires et tests de modèles.  
- **src/** : code source (préprocessing, entraînement, export, déploiement, monitoring).  
- **ci_cd/** : pipelines CI/CD pour build et déploiement.  
- **models/** : modèles sauvegardés (Pickle, ONNX) + historiques MLflow.  
- **tests/** : tests unitaires pour chaque étape du pipeline.  
- **dashboards/** : dashboards de suivi de performance et dérive.  

---

## ⚙️ Fonctionnalités principales

1. **Préparation des données** : nettoyage, imputation, encodage, scaling via PySpark / Pandas.  
2. **Entraînement multi-modèles** : Logistic Regression, XGBoost, Neural Network.  
3. **Tracking et versioning** : MLflow pour enregistrer paramètres, métriques et modèles.  
4. **Export industriel** : modèles exportés en ONNX/PMML pour déploiement.  
5. **API REST FastAPI** : prédictions en temps réel.  
6. **Containerisation Docker** : déploiement reproductible.  
7. **CI/CD** : build, test et déploiement automatisés (GitHub Actions / Jenkins).  
8. **Monitoring & recalibrage** : Evidently AI pour dérive de données et déclenchement automatique de réentraînement.

---

## 🚀 Pipeline global

```mermaid
graph TD
A[Kafka / S3 Data] --> B[PySpark Preprocessing]
B --> C[Model Training (LR / XGBoost / NN)]
C --> D[Tracking MLflow]
D --> E[Export ONNX / PMML]
E --> F[FastAPI + Docker Deployment]
F --> G[CI/CD Pipeline (GitHub Actions)]
F --> H[Monitoring & Drift Detection (Evidently AI)]
H --> I[Retrain Trigger (Automatic Recalibration)]
```
---

## 🛠️ Installation

```bash
# Cloner le projet
git clone https://github.com/votre_repo/credit_scoring_mlops.git
cd credit_scoring_mlops

# Installer les dépendances Python
pip install -r requirements.txt

# Construire l'image Docker
docker build -t credit_scoring_api:latest ./src/deployment/

# Lancer l'API
docker run -p 8000:8000 credit_scoring_api:latest
```

## 📊 Monitoring

- Le dashboard Evidently AI est généré à chaque batch ou via triggers de recalibrage automatique :
``dashboards/credit_model_monitoring.html``
- Permet de suivre la dérive de données et la performance du modèle.

## ⚡ CI/CD

- GitHub Actions ou Jenkins déclenchent :
    - Build Docker
    - Tests unitaires
    - Déploiement de l’API sur serveur ou cluster

## 🔄 Recalibrage périodique

- Basé sur seuils de dérive ou métriques (AUC / F1) via Evidently / MLflow.
- Permet un réentraînement automatique pour maintenir la performance en production.

## 📌 Remarques
- Compatible avec Spark, MLflow, FastAPI, ONNX, Docker, Evidently AI.
- Prêt pour un déploiement local ou cloud (AWS, GCP, Azure).
