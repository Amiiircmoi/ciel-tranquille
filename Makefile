.PHONY: install demo poll build train test lint dashboard docker clean

PY ?= .venv/bin/python
export PYTHONPATH := src

install:           ## Crée le venv et installe le projet (dev)
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dev]"

demo:              ## Jeu de démo complet : synth -> curated -> modèle
	./scripts/bootstrap_demo.sh

poll:              ## Un cycle de micro-batches (ingestion) en mode replay
	$(PY) -m ciel_tranquille.ingest.poller --batches 5 --no-sleep

build:             ## Reconstruit la couche curated (données réelles)
	$(PY) -c "from ciel_tranquille.storage.build_curated import build; build()"

train:             ## Entraîne et compare les modèles
	$(PY) -m ciel_tranquille.ml.train

eval-real:         ## Évalue le modèle sur l'échantillon réel (honnêteté)
	$(PY) -m ciel_tranquille.ml.evaluate_real

test:              ## Tests unitaires
	$(PY) -m pytest -q

lint:              ## Lint ruff
	$(PY) -m ruff check src tests dashboard

dashboard:         ## Lance le dashboard Streamlit
	$(PY) -m streamlit run dashboard/app.py

docker:            ## Build + run via docker-compose
	docker compose up --build

clean:             ## Nettoie les artefacts régénérables
	rm -rf data/raw data/curated models/*.joblib data/samples/bruit_synth.csv
