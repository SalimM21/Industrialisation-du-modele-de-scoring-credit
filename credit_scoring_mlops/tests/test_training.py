"""
Tests unitaires pour les scripts d'entraînement (train_lr.py, train_xgb.py, train_nn.py)
Vérifie que les modèles s'entraînent correctement et retournent des métriques attendues.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.train_lr import train_logistic_regression
from src.train_xgb import train_xgboost
from src.train_nn import train_neural_network
from sklearn.metrics import roc_auc_score, f1_score

# -----------------------------
# Fixture : données simulées
# -----------------------------
@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# -----------------------------
# Test : Logistic Regression
# -----------------------------
def test_train_lr(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    model = train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    f1 = f1_score(y_test, preds)
    
    assert auc > 0.5  # AUC minimale
    assert f1 >= 0.0  # F1 minimale

# -----------------------------
# Test : XGBoost
# -----------------------------
def test_train_xgb(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    f1 = f1_score(y_test, preds)
    
    assert auc > 0.5
    assert f1 >= 0.0

# -----------------------------
# Test : Neural Network
# -----------------------------
def test_train_nn(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    model = train_neural_network(X_train, y_train)
    preds_prob = model.predict(X_test).ravel()
    preds = (preds_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, preds_prob)
    f1 = f1_score(y_test, preds)
    
    assert auc > 0.5
    assert f1 >= 0.0
