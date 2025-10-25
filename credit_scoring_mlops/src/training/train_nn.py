"""
Entraînement d'un modèle Neural Network pour le scoring crédit.

Fonctionnalités :
- Chargement des datasets pré-traités
- Construction d'un réseau de neurones Keras
- Entraînement avec early stopping
- Évaluation AUC et F1
- Tracking complet avec MLflow (metrics, params, modèle)
- Sauvegarde du modèle
"""

import numpy as np
import os
import mlflow
import mlflow.keras
from sklearn.metrics import roc_auc_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Charger les datasets
X_train = np.load('../data/processed/X_train.npy')
X_val = np.load('../data/processed/X_val.npy')
X_test = np.load('../data/processed/X_test.npy')
y_train = np.load('../data/processed/y_train.npy')
y_val = np.load('../data/processed/y_val.npy')
y_test = np.load('../data/processed/y_test.npy')

# Définir le modèle Neural Network
def build_nn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[])
    return model

nn_model = build_nn(X_train.shape[1])

# MLflow tracking
mlflow.set_experiment("Credit_Scoring_NeuralNetwork")

with mlflow.start_run(run_name="NeuralNetwork"):
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraînement
    nn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Prédiction
    y_pred_prob = nn_model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Évaluation
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Neural Network - Test AUC: {auc:.4f}, F1: {f1:.4f}")

    # Logging MLflow
    mlflow.log_param("model_type", "NeuralNetwork")
    mlflow.log_param("input_dim", X_train.shape[1])
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1", f1)

    # Sauvegarde du modèle
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/nn_model'
    nn_model.save(model_path, save_format='tf')
    mlflow.keras.log_model(nn_model, "nn_model")
    print(f"Modèle Neural Network sauvegardé dans {model_path}")
