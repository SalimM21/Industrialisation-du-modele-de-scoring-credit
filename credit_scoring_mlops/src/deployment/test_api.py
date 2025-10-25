"""
Tests unitaires pour l'API FastAPI de scoring crédit.
Utilise pytest et FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# -----------------------------
# Test endpoint /predict
# -----------------------------
def test_predict_low_risk():
    payload = {
        "age": 30,
        "income": 50000,
        "loan_amount": 10000,
        "gender": "female",
        "occupation": "engineer",
        "marital_status": "single"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "score" in json_data
    assert "risk_class" in json_data
    assert json_data["risk_class"] in ["low", "high"]

def test_predict_high_risk():
    payload = {
        "age": 22,
        "income": 15000,
        "loan_amount": 12000,
        "gender": "male",
        "occupation": "student",
        "marital_status": "single"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "score" in json_data
    assert "risk_class" in json_data
    assert json_data["risk_class"] in ["low", "high"]

def test_predict_missing_field():
    payload = {
        "age": 25,
        "income": 30000,
        # "loan_amount" manquant
        "gender": "male",
        "occupation": "engineer",
        "marital_status": "single"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error attendu
