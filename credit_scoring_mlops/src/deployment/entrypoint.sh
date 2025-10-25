#!/bin/bash
# -----------------------------
# entrypoint.sh
# Script de démarrage pour le container Docker
# -----------------------------

# Sortir si une commande échoue
set -e

# Activer les logs
echo "🚀 Démarrage de l'API Credit Scoring..."

# Vérifier que le dossier logs existe
mkdir -p /app/logs

# Exécuter l'API FastAPI avec uvicorn
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
