"""
Entraînement d'un modèle XGBoost pour le scoring crédit.

Fonctionnalités :
- Chargement des datasets pré-traités
- Entraînement XGBoost avec gestion des classes déséquilibrées
- Évaluation AUC et F1
- Tracking complet avec MLflow (metrics, params, modèle)
- Sauvegarde du modèle
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score
import mlflow
import mlflow.sklearn
import os
import joblib

# Charger les datasets
X_train = np.load('../data/processed/X_train.npy')
X_val = np.load('../data/processed/X_val.npy')
X_test = np.load('../data/processed/X_test.npy')
y_train = np.load('../data/processed/y_train.npy')
y_val = np.load('../data/processed/y_val.npy')
y_test = np.load('../data/processed/y_test.npy')

# Calcul du scale_pos_weight pour classes déséquilibrées
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# Initialiser le modèle XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# MLflow tracking
mlflow.set_experiment("Credit_Scoring_XGBoost")

with mlflow.start_run(run_name="XGBoost"):
    # Entraînement
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    # Prédiction
    y_pred = xgb_model.predict(X_test)

    # Évaluation
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"XGBoost - Test AUC: {auc:.4f}, F1: {f1:.4f}")

    # Logging MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1", f1)

    # Sauvegarde du modèle
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/xgboost_model.pkl'
    joblib.dump(xgb_model, model_path)
    mlflow.sklearn.log_model(xgb_model, "xgboost_model")
    print(f"Modèle sauvegardé dans {model_path}")
