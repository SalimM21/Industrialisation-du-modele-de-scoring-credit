"""
API FastAPI pour le scoring crédit.

Fonctionnalités :
- Endpoint `/predict` pour prédire le score de risque
- Chargement du meilleur modèle MLflow
- Préprocessing des données entrantes
- Retour JSON avec score et classification
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
import uvicorn
from preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from tensorflow.keras.models import load_model

# Définir l'application FastAPI
app = FastAPI(title="Credit Scoring API", version="1.0")

# Charger le pipeline de préprocessing et les features sélectionnées
preprocessor = DataPreprocessor()
preprocessor.load()
selected_features = FeatureSelector(pd.DataFrame()).load_selected_features()

# Charger le meilleur modèle depuis MLflow ou local
BEST_MODEL_PATH = "../models/nn_model"  # Exemple : Neural Network
model_type = "keras"  # "sklearn" ou "keras"

if model_type == "sklearn":
    model = joblib.load(BEST_MODEL_PATH + ".pkl")
elif model_type == "keras":
    model = load_model(BEST_MODEL_PATH)

# Définir le schéma des données d'entrée
class InputData(BaseModel):
    age: float
    income: float
    loan_amount: float
    gender: str
    occupation: str
    marital_status: str

# Endpoint `/predict`
@app.post("/predict")
def predict(data: InputData):
    # Convertir en DataFrame
    import pandas as pd
    input_df = pd.DataFrame([data.dict()])

    # Préprocessing
    X_scaled = preprocessor.transform(input_df)

    # Sélection des features
    X_final = pd.DataFrame(X_scaled, columns=selected_features)[selected_features].values

    # Prédiction
    if model_type == "sklearn":
        pred = model.predict_proba(X_final)[:, 1][0]
    elif model_type == "keras":
        pred = model.predict(X_final).ravel()[0]

    # Classification simple
    threshold = 0.5
    risk_class = "high" if pred >= threshold else "low"

    return {"score": float(pred), "risk_class": risk_class}

# Lancer le serveur
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
