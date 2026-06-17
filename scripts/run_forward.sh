#!/usr/bin/env bash
# Poller OpenSky *forward* continu (collecte co-localisée, cadence CT_POLL_INTERVAL_S).
# S'arrête proprement sur SIGTERM/SIGINT ou au plancher de crédits (CT_CREDIT_FLOOR).
# Utilisé par le service systemd ciel-tranquille-forward.service (ou en local).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
exec .venv/bin/python -m ciel_tranquille.ingest.poller --forward "$@"
