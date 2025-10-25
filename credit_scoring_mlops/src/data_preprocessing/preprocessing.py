"""
----------------
Module de préparation des données pour le projet Credit Scoring.

Fonctionnalités :
- Nettoyage des valeurs manquantes et aberrantes
- Encodage des variables catégorielles
- Imputation des valeurs manquantes
- Scaling des variables numériques
- Préparation pour l'entraînement des modèles ML
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class DataPreprocessor:
    def __init__(self, numeric_features, categorical_features):
        """
        Args:
            numeric_features (list): Liste des colonnes numériques
            categorical_features (list): Liste des colonnes catégorielles
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = None

    def build_pipeline(self):
        """Construit le pipeline de préprocessing"""
        # Pipeline pour les features numériques
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Imputation médiane
            ('scaler', StandardScaler())  # Normalisation
        ])

        # Pipeline pour les features catégorielles
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation mode
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encodage one-hot
        ])

        # Transformer complet
        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, self.numeric_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])

    def fit_transform(self, df):
        """
        Fit et transforme les données
        Args:
            df (pd.DataFrame): Dataframe à transformer
        Returns:
            np.ndarray: données transformées
        """
        if self.preprocessor is None:
            self.build_pipeline()
        return self.preprocessor.fit_transform(df)

    def transform(self, df):
        """
        Transforme les nouvelles données
        Args:
            df (pd.DataFrame): Dataframe à transformer
        Returns:
            np.ndarray: données transformées
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline non initialisé. Utiliser fit_transform avant.")
        return self.preprocessor.transform(df)

    def save(self, filepath='../models/preprocessor.joblib'):
        """
        Sauvegarde le pipeline de preprocessing
        Args:
            filepath (str): chemin de sauvegarde
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
        print(f"Pipeline de preprocessing sauvegardé dans {filepath}")

    def load(self, filepath='../models/preprocessor.joblib'):
        """
        Charge le pipeline de preprocessing
        Args:
            filepath (str): chemin du pipeline sauvegardé
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier {filepath} non trouvé.")
        self.preprocessor = joblib.load(filepath)
        print(f"Pipeline de preprocessing chargé depuis {filepath}")

# Exemple d'utilisation
if __name__ == "__main__":
    df = pd.read_csv('../data/processed/processed_data.csv')
    numeric_features = ['age', 'income', 'loan_amount']  # adapter selon ton dataset
    categorical_features = ['gender', 'occupation', 'marital_status']  # adapter selon ton dataset

    preprocessor = DataPreprocessor(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(df.drop('target', axis=1))
    preprocessor.save()
