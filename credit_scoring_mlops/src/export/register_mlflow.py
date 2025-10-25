"""
Enregistrement du meilleur modèle dans MLflow Model Registry.

Fonctionnalités :
- Chargement des modèles sauvegardés
- Sélection du meilleur modèle basé sur les métriques (AUC prioritaire)
- Enregistrement dans le MLflow Model Registry avec versioning
"""

import joblib
import os
from tensorflow.keras.models import load_model
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# Charger le test set
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

# Chemins des modèles
lr_model_path = '../models/logistic_regression_model.pkl'
xgb_model_path = '../models/xgboost_model.pkl'
nn_model_path = '../models/nn_model'

# Charger les modèles existants
models = {}

if os.path.exists(lr_model_path):
    lr_model = joblib.load(lr_model_path)
    y_pred = lr_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    models['LogisticRegression'] = {'model': lr_model, 'auc': auc, 'type': 'sklearn'}

if os.path.exists(xgb_model_path):
    xgb_model = joblib.load(xgb_model_path)
    y_pred = xgb_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    models['XGBoost'] = {'model': xgb_model, 'auc': auc, 'type': 'sklearn'}

if os.path.exists(nn_model_path):
    nn_model = load_model(nn_model_path)
    y_pred_prob = nn_model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred)
    models['NeuralNetwork'] = {'model': nn_model, 'auc': auc, 'type': 'keras'}

# Sélection du meilleur modèle (AUC prioritaire)
best_model_name = max(models, key=lambda k: models[k]['auc'])
best_model_info = models[best_model_name]
print(f"✅ Meilleur modèle : {best_model_name} avec AUC={best_model_info['auc']:.4f}")

# MLflow registry
mlflow.set_experiment("Credit_Scoring_Final_Model")
with mlflow.start_run(run_name=f"Register_{best_model_name}"):
    if best_model_info['type'] == 'sklearn':
        mlflow.sklearn.log_model(best_model_info['model'], artifact_path=best_model_name, registered_model_name="CreditScoringModel")
    elif best_model_info['type'] == 'keras':
        mlflow.keras.log_model(best_model_info['model'], artifact_path=best_model_name, registered_model_name="CreditScoringModel")
    
    mlflow.log_param("best_model_name", best_model_name)
    mlflow.log_metric("best_model_auc", best_model_info['auc'])

print(f"Le modèle {best_model_name} est enregistré dans le MLflow Model Registry sous 'CreditScoringModel'")
