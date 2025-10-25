"""
-------------------
Module pour la sélection des variables pertinentes pour le projet Credit Scoring.

Fonctionnalités :
- Sélection univariée (test chi² ou ANOVA pour variables catégorielles / numériques)
- Sélection basée sur l'importance des features (Random Forest / XGBoost)
- Sortie des features retenues pour l'entraînement des modèles
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import joblib

class FeatureSelector:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Args:
            X (pd.DataFrame): features
            y (pd.Series): target
        """
        self.X = X
        self.y = y
        self.selected_features = None

    def univariate_selection(self, k=10):
        """
        Sélection univariée des features (ANOVA pour numériques)
        Args:
            k (int): nombre de features à retenir
        Returns:
            list: features retenues
        """
        numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_features)))
        selector.fit(self.X[numeric_features], self.y)
        self.selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
        print(f"Features sélectionnées (univariée) : {self.selected_features}")
        return self.selected_features

    def importance_selection(self, threshold=0.01):
        """
        Sélection des features basée sur l'importance d'un RandomForest
        Args:
            threshold (float): seuil d'importance minimal
        Returns:
            list: features retenues
        """
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(self.X, self.y)
        importances = model.feature_importances_
        self.selected_features = self.X.columns[importances > threshold].tolist()
        print(f"Features sélectionnées (importance) : {self.selected_features}")

        # Visualisation
        feat_importances = pd.Series(importances, index=self.X.columns)
        feat_importances.sort_values(ascending=False).plot(kind='bar')
        plt.title("Importance des features (RandomForest)")
        plt.show()

        return self.selected_features

    def save_selected_features(self, filepath='../models/selected_features.pkl'):
        """
        Sauvegarde la liste des features sélectionnées
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.selected_features, filepath)
        print(f"Features sélectionnées sauvegardées dans {filepath}")

    def load_selected_features(self, filepath='../models/selected_features.pkl'):
        """
        Charge les features sélectionnées
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier {filepath} non trouvé.")
        self.selected_features = joblib.load(filepath)
        print(f"Features sélectionnées chargées depuis {filepath}")
        return self.selected_features

# Exemple d'utilisation
if __name__ == "__main__":
    df = pd.read_csv('../data/processed/processed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    selector = FeatureSelector(X, y)
    selector.univariate_selection(k=10)
    selector.importance_selection(threshold=0.02)
    selector.save_selected_features()
