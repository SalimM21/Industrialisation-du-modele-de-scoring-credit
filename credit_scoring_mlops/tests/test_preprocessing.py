"""
Tests unitaires pour le module preprocessing.py
Utilise pytest pour valider le nettoyage, encodage, imputation et scaling.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_data

# -----------------------------
# Fixtures pour les tests
# -----------------------------
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40],
        "income": [50000, 60000, 55000, np.nan],
        "gender": ["male", "female", "female", "male"],
        "loan_amount": [10000, 15000, 12000, 13000]
    })

# -----------------------------
# Test : imputation des valeurs manquantes
# -----------------------------
def test_imputation(sample_data):
    df_processed = preprocess_data(sample_data)
    # Pas de valeurs manquantes après preprocessing
    assert df_processed.isnull().sum().sum() == 0

# -----------------------------
# Test : encodage des variables catégorielles
# -----------------------------
def test_encoding(sample_data):
    df_processed = preprocess_data(sample_data)
    # Les colonnes catégorielles doivent être converties en numériques
    assert np.issubdtype(df_processed["gender_male"].dtype, np.number)

# -----------------------------
# Test : scaling des features
# -----------------------------
def test_scaling(sample_data):
    df_processed = preprocess_data(sample_data)
    # Les colonnes numériques doivent être centrées autour de 0
    for col in ["age", "income", "loan_amount"]:
        mean_val = df_processed[col].mean()
        assert abs(mean_val) < 1e-6  # tolérance pour arrondis

# -----------------------------
# Test : output type
# -----------------------------
def test_output_type(sample_data):
    df_processed = preprocess_data(sample_data)
    assert isinstance(df_processed, pd.DataFrame)
