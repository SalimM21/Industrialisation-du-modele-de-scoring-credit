"""
Configuration centralisée des logs pour le projet Credit Scoring.
Permet d'avoir des logs cohérents dans tous les modules.
"""

import logging
import os

# -----------------------------
# Fonction de création du logger
# -----------------------------
def setup_logger(name="credit_scoring", log_file="../logs/app.log", level=logging.INFO):
    """
    Crée et retourne un logger configuré pour le projet.
    
    Args:
        name (str): nom du logger.
        log_file (str): chemin du fichier de log.
        level: niveau de log (DEBUG, INFO, WARNING, ERROR).
        
    Returns:
        logging.Logger: logger configuré.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter standard
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler pour fichier
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    
    # Handler pour console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    # Ajouter handlers si non existants
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

# -----------------------------
# Logger global pour tout le projet
# -----------------------------
logger = setup_logger()

# -----------------------------
# Exemple d'utilisation
# -----------------------------
if __name__ == "__main__":
    logger.info("✅ Logger initialisé avec succès.")
    logger.warning("⚠️ Ceci est un avertissement de test.")
    logger.error("❌ Ceci est un message d'erreur de test.")
