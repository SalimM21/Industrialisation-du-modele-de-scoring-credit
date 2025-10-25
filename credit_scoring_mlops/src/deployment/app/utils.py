"""
Fonctions utilitaires pour le projet de scoring crédit :
- Chargement de modèles ML
- Gestion des logs
- Fonctions auxiliaires diverses
"""

import os
import joblib
from tensorflow.keras.models import load_model
import logging

# ----------------------------
# Gestion des logs centralisés
# ----------------------------
def setup_logger(name="credit_scoring", log_file="../logs/app.log", level=logging.INFO):
    """Configure un logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    
    # Stream handler (console)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    # Ajouter handlers
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# ----------------------------
# Chargement des modèles
# ----------------------------
def load_sklearn_model(model_path):
    """Charger un modèle scikit-learn"""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Modèle sklearn chargé depuis {model_path}")
        return model
    else:
        logger.error(f"Modèle sklearn introuvable : {model_path}")
        return None

def load_keras_model(model_path):
    """Charger un modèle Keras/TensorFlow"""
    if os.path.exists(model_path):
        model = load_model(model_path)
        logger.info(f"Modèle Keras chargé depuis {model_path}")
        return model
    else:
        logger.error(f"Modèle Keras introuvable : {model_path}")
        return None

# ----------------------------
# Fonctions auxiliaires diverses
# ----------------------------
def ensure_dir(directory):
    """Créer un dossier s'il n'existe pas"""
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Dossier créé ou déjà existant : {directory}")

def save_numpy_array(array, path):
    """Sauvegarder un tableau NumPy"""
    ensure_dir(os.path.dirname(path))
    import numpy as np
    np.save(path, array)
    logger.info(f"Tableau NumPy sauvegardé : {path}")

def load_numpy_array(path):
    """Charger un tableau NumPy"""
    import numpy as np
    if os.path.exists(path):
        array = np.load(path)
        logger.info(f"Tableau NumPy chargé : {path}")
        return array
    else:
        logger.error(f"Fichier NumPy introuvable : {path}")
        return None
