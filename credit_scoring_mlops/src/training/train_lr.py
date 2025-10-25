"""
Entraînement d'un modèle Logistic Regression pour le scoring crédit.

Fonctionnalités :
- Chargement des datasets pré-traités
- Entraînement du modèle Logistic Regression avec class_weight
- Évaluation AUC et F1
- Tracking complet avec MLflow (metrics, params, modèle)
- Sauvegarde du modèle
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Charger les datasets
X_train = np.load('../data/processed/X_train.npy')
X_val = np.load('../data/processed/X_val.npy')
X_test = np.load('../data/processed/X_test.npy')
y_train = np.load('../data/processed/y_train.npy')
y_val = np.load('../data/processed/y_val.npy')
y_test = np.load('../data/processed/y_test.npy')

# Initialiser le modèle
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# MLflow tracking
mlflow.set_experiment("Credit_Scoring_LogisticRegression")

with mlflow.start_run(run_name="LogisticRegression") as run:
    # Entraînement
    lr_model.fit(X_train, y_train)

    # Prédiction
    y_pred = lr_model.predict(X_test)

    # Évaluation
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Logistic Regression - Test AUC: {auc:.4f}, F1: {f1:.4f}")

    # Logging MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1", f1)

    # Sauvegarde du modèle
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/logistic_regression_model.pkl'
    joblib.dump(lr_model, model_path)
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    print(f"Modèle sauvegardé dans {model_path}")
