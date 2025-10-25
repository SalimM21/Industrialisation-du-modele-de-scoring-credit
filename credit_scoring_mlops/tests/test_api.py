"""
Tests unitaires pour les endpoints FastAPI.
Vérifie /predict et réponses JSON.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app  # Assurez-vous que main.py contient l'objet FastAPI `app`

client = TestClient(app)

# -----------------------------
# Exemple de payload utilisateur
# -----------------------------
@pytest.fixture
def sample_input():
    return {
        "age": 30,
        "income": 50000,
        "gender": "male",
        "loan_amount": 10000,
        "credit_history": 1,
        "dependents": 0
    }

# -----------------------------
# Test endpoint /predict
# -----------------------------
def test_predict_endpoint(sample_input):
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    data = response.json()
    # Vérifie la présence des clés attendues
    assert "risk_score" in data
    assert "risk_class" in data
    # Vérifie que le score est compris entre 0 et 1
    assert 0.0 <= data["risk_score"] <= 1.0
    # Vérifie que la classe est 0 ou 1
    assert data["risk_class"] in [0, 1]

# -----------------------------
# Test endpoint racine ou health
# -----------------------------
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
