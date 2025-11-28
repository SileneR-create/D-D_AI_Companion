FROM python:3.9-slim

# Dossier de travail propre
WORKDIR /app

COPY . .

#projet_dnd/D-D_AI_Companion

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# Installer dépendances Python
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/HEALTHCHECK || exit 1

ENTRYPOINT ["streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0"]
