"""
Détection automatique de dérive des données pour le scoring crédit.
Utilise Evidently pour analyser les features et les prédictions.
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
import joblib
from tensorflow.keras.models import load_model
from utils import load_numpy_array, logger

# -----------------------------
# Charger les données de référence et nouvelles données
# -----------------------------
reference_data_path = "../data/processed/X_train.npy"
new_data_path = "../data/processed/X_test.npy"
y_test_path = "../data/processed/y_test.npy"

X_ref = load_numpy_array(reference_data_path)
X_new = load_numpy_array(new_data_path)
y_new = load_numpy_array(y_test_path)

# Convertir en DataFrame (Evidently nécessite noms de colonnes)
feature_names = [f"feature_{i}" for i in range(X_ref.shape[1])]
df_ref = pd.DataFrame(X_ref, columns=feature_names)
df_new = pd.DataFrame(X_new, columns=feature_names)

# Charger modèle pour prédictions
model_path = "../models/nn_model"
model = load_model(model_path)
y_pred_new = model.predict(X_new).ravel()
df_new["prediction"] = y_pred_new
df_ref["prediction"] = model.predict(X_ref).ravel()

# -----------------------------
# Détection de drift
# -----------------------------
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), ClassificationPreset()])
report.run(reference_data=df_ref, current_data=df_new, column_mapping=None)

# Générer un rapport HTML
report_path = "../dashboards/drift_report.html"
report.save_html(report_path)
logger.info(f"Rapport de dérive généré : {report_path}")

# Optionnel : print résumé des dérives
metrics_result = report.as_dict()
data_drift_status = metrics_result['metrics'][0]['result']['dataset_drift']
if data_drift_status:
    logger.warning("⚠️ Dérive détectée dans les données !")
else:
    logger.info("✅ Pas de dérive détectée.")
