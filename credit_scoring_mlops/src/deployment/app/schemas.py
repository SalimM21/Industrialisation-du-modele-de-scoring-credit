"""
Définition des schémas Pydantic pour les entrées et sorties de l'API Credit Scoring.
"""

from pydantic import BaseModel, Field
from typing import Optional

# Schéma d'entrée pour un client
class ClientData(BaseModel):
    age: float = Field(..., example=35, description="Âge du client")
    income: float = Field(..., example=45000, description="Revenu annuel du client")
    loan_amount: float = Field(..., example=15000, description="Montant du prêt demandé")
    gender: str = Field(..., example="male", description="Genre du client")
    occupation: str = Field(..., example="engineer", description="Profession du client")
    marital_status: str = Field(..., example="single", description="Situation matrimoniale du client")

# Schéma de sortie de prédiction
class PredictionResult(BaseModel):
    score: float = Field(..., example=0.78, description="Score de risque de défaut (0-1)")
    risk_class: str = Field(..., example="high", description="Classe de risque ('high' ou 'low')")
