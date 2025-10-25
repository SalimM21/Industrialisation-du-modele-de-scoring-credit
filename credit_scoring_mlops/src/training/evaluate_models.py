"""
Comparaison des modèles de scoring crédit et sélection du meilleur.

Fonctionnalités :
- Chargement des modèles entraînés
- Évaluation sur le test set (AUC et F1)
- Comparaison et sélection du meilleur modèle
- Tracking MLflow et logging des métriques comparatives
"""

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.metrics import roc_auc_score, f1_score
import os
from tensorflow.keras.models import load_model

# Charger le test set
X_test = np.load('../data/processed/X_test.npy')
y_test = np.load('../data/processed/y_test.npy')

# Dictionnaire pour stocker les modèles et leurs types
models = {}

# Charger Logistic Regression
lr_model_path = '../models/logistic_regression_model.pkl'
if os.path.exists(lr_model_path):
    lr_model = joblib.load(lr_model_path)
    models['LogisticRegression'] = ('sklearn', lr_model)

# Charger XGBoost
xgb_model_path = '../models/xgboost_model.pkl'
if os.path.exists(xgb_model_path):
    xgb_model = joblib.load(xgb_model_path)
    models['XGBoost'] = ('sklearn', xgb_model)

# Charger Neural Network
nn_model_path = '../models/nn_model'
if os.path.exists(nn_model_path):
    nn_model = load_model(nn_model_path)
    models['NeuralNetwork'] = ('keras', nn_model)

# Comparaison des modèles
results = {}
for name, (model_type, model) in models.items():
    if model_type == 'sklearn':
        y_pred = model.predict(X_test)
    elif model_type == 'keras':
        y_pred_prob = model.predict(X_test).ravel()
        y_pred = (y_pred_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'AUC': auc, 'F1': f1}
    print(f"{name} - Test AUC: {auc:.4f}, F1: {f1:.4f}")

# Sélection du meilleur modèle (AUC prioritaire)
best_model_name = max(results, key=lambda x: results[x]['AUC'])
print(f"\n✅ Meilleur modèle : {best_model_name} avec AUC={results[best_model_name]['AUC']:.4f} et F1={results[best_model_name]['F1']:.4f}")

# Logging MLflow
mlflow.set_experiment("Credit_Scoring_Model_Comparison")
with mlflow.start_run(run_name="Model_Comparison"):
    for name, metrics in results.items():
        mlflow.log_metric(f"{name}_AUC", metrics['AUC'])
        mlflow.log_metric(f"{name}_F1", metrics['F1'])
    mlflow.log_param("best_model", best_model_name)
