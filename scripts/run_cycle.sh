#!/usr/bin/env bash
# Exécute un cycle de pipeline micro-batch (ingestion -> curated).
# Utilisé par cron / systemd timer (cf. deploy/).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
exec .venv/bin/python -m ciel_tranquille.pipeline --batches "${CT_BATCHES:-5}" "$@"
