#!/usr/bin/env bash
# Collecteur journalier d'événements de survol Bruitparif (source bruit).
# Récupère par défaut les derniers jours pour les 3 stations, puis recharge DuckDB.
# Utilisé par le timer systemd ciel-tranquille-noise.timer (ou cron / en local).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
# Par défaut on (re)collecte les 2 derniers jours : couvre la latence de publication
# Bruitparif sans re-télécharger tout l'historique. Surchargable : collect_noise.sh --days 14
if [ "$#" -eq 0 ]; then
  set -- --days 2
fi
exec .venv/bin/python -m ciel_tranquille.ingest.noise_collector "$@"
