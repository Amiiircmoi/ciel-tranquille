# Image unique servant l'ingestion (poller) ET le dashboard.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dépendances d'abord (cache des couches Docker).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Code applicatif + données échantillons + modèle packagé (dashboard autonome).
COPY dashboard ./dashboard
COPY data/samples ./data/samples
COPY models ./models
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

EXPOSE 8501

# Par défaut : dashboard. Le service "pipeline" surcharge la commande (compose).
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
