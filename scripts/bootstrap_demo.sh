#!/usr/bin/env bash
# Prépare un jeu de démonstration complet et reproductible :
# génération synthétique -> couche curated -> entraînement du modèle.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
PY="${PYTHON:-.venv/bin/python}"

echo "[1/3] Génération des données synthétiques (seed=42, 10 jours)…"
"$PY" -m ciel_tranquille.ml.synth --seed 42 --days 10

echo "[2/3] Construction de la couche curated (DuckDB)…"
"$PY" -c "from ciel_tranquille.storage.build_curated import build; build(noise_csv='bruit_synth.csv')"

echo "[3/3] Entraînement et comparaison des modèles…"
"$PY" -m ciel_tranquille.ml.train

echo "✅ Démo prête. Lancez le dashboard : streamlit run dashboard/app.py"
