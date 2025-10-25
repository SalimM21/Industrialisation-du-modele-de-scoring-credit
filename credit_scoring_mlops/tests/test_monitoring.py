"""
Tests unitaires pour la détection de dérive (Evidently)
Vérifie que le script monitoring_drift.ipynb / drift_detection.py fonctionne.
"""

import pytest
import pandas as pd
import numpy as np
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from src.drift_detection import compute_drift_report

# -----------------------------
# Fixture : données de référence et nouvelles données
# -----------------------------
@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 60, 100),
        "income": np.random.randint(30000, 100000, 100),
        "loan_amount": np.random.randint(5000, 20000, 100),
        "credit_history": np.random.randint(0, 2, 100)
    })

@pytest.fixture
def new_data(reference_data):
    # Introduire une dérive artificielle sur 'income'
    data = reference_data.copy()
    data["income"] = data["income"] * np.random.uniform(0.8, 1.2, len(data))
    return data

# -----------------------------
# Test : génération du rapport de dérive
# -----------------------------
def test_drift_report(reference_data, new_data):
    report_path = "tests/test_drift_report.html"
    compute_drift_report(reference_data, new_data, output_path=report_path)
    
    # Vérifier que le fichier HTML a été créé
    import os
    assert os.path.exists(report_path)
    
    # Nettoyage (optionnel)
    os.remove(report_path)

# -----------------------------
# Test : calcul des metrics de drift
# -----------------------------
def test_drift_metrics(reference_data, new_data):
    profile = Profile(sections=[DataDriftProfileSection()])
    profile.calculate(reference_data, new_data)
    metrics = profile.json()
    
    # Vérifier que certaines clés existent
    assert "data_drift" in metrics
    assert "metrics" in metrics
