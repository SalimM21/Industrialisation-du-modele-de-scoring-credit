"""
-------------
Module pour séparer les données en train, validation et test pour le projet Credit Scoring.

Fonctionnalités :
- Split stratifié pour conserver la proportion de la cible
- Application du pipeline de préprocessing
- Sélection des features pertinentes
- Sauvegarde des datasets transformés
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
import joblib

def split_and_prepare(df, target_column='target', test_size=0.3, val_size=0.5, numeric_features=None, categorical_features=None):
    """
    Split et préparation des données
    
    Args:
        df (pd.DataFrame): dataset complet
        target_column (str): nom de la colonne cible
        test_size (float): proportion du split test+validation
        val_size (float): proportion du split validation dans test+validation
        numeric_features (list): colonnes numériques
        categorical_features (list): colonnes catégorielles
        
    Returns:
        dict: datasets transformés {X_train, X_val, X_test, y_train, y_val, y_test}
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split train / temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Split validation / test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    # Préprocessing
    preprocessor = DataPreprocessor(numeric_features=numeric_features, categorical_features=categorical_features)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    preprocessor.save()

    # Feature selection
    selector = FeatureSelector(pd.DataFrame(X_train_scaled), y_train)
    selector.importance_selection(threshold=0.01)
    selected_features = selector.selected_features
    selector.save_selected_features()

    # Réduction des datasets aux features sélectionnées
    X_train_final = pd.DataFrame(X_train_scaled, columns=numeric_features + categorical_features)[selected_features].values
    X_val_final = pd.DataFrame(X_val_scaled, columns=numeric_features + categorical_features)[selected_features].values
    X_test_final = pd.DataFrame(X_test_scaled, columns=numeric_features + categorical_features)[selected_features].values

    # Sauvegarde
    os.makedirs('../data/processed', exist_ok=True)
    np.save('../data/processed/X_train.npy', X_train_final)
    np.save('../data/processed/X_val.npy', X_val_final)
    np.save('../data/processed/X_test.npy', X_test_final)
    np.save('../data/processed/y_train.npy', y_train.values)
    np.save('../data/processed/y_val.npy', y_val.values)
    np.save('../data/processed/y_test.npy', y_test.values)

    print("Split train/validation/test et préprocessing terminés, datasets sauvegardés dans '../data/processed/'")

    return {
        'X_train': X_train_final,
        'X_val': X_val_final,
        'X_test': X_test_final,
        'y_train': y_train.values,
        'y_val': y_val.values,
        'y_test': y_test.values
    }

# Exemple d'utilisation
if __name__ == "__main__":
    df = pd.read_csv('../data/processed/processed_data.csv')
    numeric_features = ['age', 'income', 'loan_amount']  # adapter selon ton dataset
    categorical_features = ['gender', 'occupation', 'marital_status']  # adapter selon ton dataset

    datasets = split_and_prepare(df, target_column='target', numeric_features=numeric_features, categorical_features=categorical_features)
