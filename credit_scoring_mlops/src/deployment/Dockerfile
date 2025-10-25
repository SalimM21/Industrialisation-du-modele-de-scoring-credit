# ---------------------------
# Étape 1 : Base image
# ---------------------------
FROM python:3.11-slim

# Labels et métadonnées
LABEL maintainer="salim.majide@example.com"
LABEL description="Container API FastAPI pour scoring crédit"

# ---------------------------
# Étape 2 : Environnement
# ---------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ---------------------------
# Étape 3 : Installer les dépendances
# ---------------------------
# Copier les fichiers requirements
COPY requirements.txt .

# Installer pip et dépendances
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Étape 4 : Copier le code source
# ---------------------------
COPY . .

# Créer dossier logs
RUN mkdir -p /app/logs

# ---------------------------
# Étape 5 : Exposer le port
# ---------------------------
EXPOSE 8000

# ---------------------------
# Étape 6 : Commande de démarrage
# ---------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# point d’entrée Docker
# COPY entrypoint.sh /app/entrypoint.sh
# RUN chmod +x /app/entrypoint.sh
# ENTRYPOINT ["/app/entrypoint.sh"]
