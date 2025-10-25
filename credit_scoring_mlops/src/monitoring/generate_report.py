"""
Génération automatique d'un dashboard HTML Evidently
pour la surveillance de la qualité des données et la détection de dérive.
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from utils import load_numpy_array, logger

# -----------------------------
# Chargement des données
# -----------------------------
reference_data_path = "../data/processed/X_train.npy"
current_data_path   = "../data/processed/X_test.npy"
y_test_path         = "../data/processed/y_test.npy"

X_ref = load_numpy_array(reference_data_path)
X_curr = load_numpy_array(current_data_path)
y_curr = load_numpy_array(y_test_path)

# Convertir en DataFrame
feature_names = [f"feature_{i}" for i in range(X_ref.shape[1])]
df_ref  = pd.DataFrame(X_ref, columns=feature_names)
df_curr = pd.DataFrame(X_curr, columns=feature_names)

# -----------------------------
# Ajout des prédictions (si disponibles)
# -----------------------------
# Exemple : charger modèle Keras pour prédictions
from tensorflow.keras.models import load_model
model_path = "../models/nn_model"
model = load_model(model_path)
df_ref["prediction"]  = model.predict(X_ref).ravel()
df_curr["prediction"] = model.predict(X_curr).ravel()

# -----------------------------
# Génération du rapport Evidently
# -----------------------------
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), ClassificationPreset()])
report.run(reference_data=df_ref, current_data=df_curr)

# Export HTML
report_path = "../dashboards/evidently_dashboard.html"
report.save_html(report_path)
logger.info(f"Dashboard Evidently généré : {report_path}")
