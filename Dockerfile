# Utilisez une image de base avec Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Installer les dépendances supplémentaires nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Exposer le port pour Streamlit
EXPOSE 8501

# Commande pour lancer l'application
CMD ["streamlit", "run", "ui/app.py"]