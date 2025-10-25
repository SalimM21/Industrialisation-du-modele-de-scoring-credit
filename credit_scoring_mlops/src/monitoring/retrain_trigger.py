"""
Déclenche le recalibrage périodique du modèle de scoring crédit
lorsqu'une dérive significative est détectée dans les données.
"""

import os
import logging
from drift_detection import report
from utils import logger
import subprocess
from datetime import datetime

# Seuil de dérive pour déclencher le recalibrage (True si dérive détectée)
DRIFT_THRESHOLD = True  # basé sur la métrique DataDriftPreset d'Evidently

# Fonction de déclenchement du réentraînement
def trigger_retrain():
    """
    Vérifie le rapport de dérive et déclenche le script d'entraînement
    si une dérive significative est détectée.
    """
    metrics_result = report.as_dict()
    data_drift_status = metrics_result['metrics'][0]['result']['dataset_drift']

    if data_drift_status == DRIFT_THRESHOLD:
        logger.warning("⚠️ Dérive détectée ! Déclenchement du réentraînement...")
        
        # Exemple : appeler un script de réentraînement complet
        # Le script peut être train_all_models.py ou pipeline complet
        retrain_script = "train_all_models.py"
        if os.path.exists(retrain_script):
            subprocess.run(["python", retrain_script], check=True)
            logger.info(f"Réentraînement déclenché à {datetime.now()}")
        else:
            logger.error(f"Script de réentraînement introuvable : {retrain_script}")
    else:
        logger.info("✅ Pas de dérive détectée. Réentraînement non nécessaire.")

# Exécution principale
if __name__ == "__main__":
    trigger_retrain()
