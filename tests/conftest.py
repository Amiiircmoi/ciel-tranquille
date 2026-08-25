"""Fixtures de test : settings isolés sur un répertoire temporaire."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SNAPSHOT = REPO_ROOT / "data" / "samples" / "opensky_snapshot.csv"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings pointant vers un data_dir temporaire (aucun effet de bord réel)."""
    from ciel_tranquille.config import get_settings

    # Neutralise le `.env` du poste : sans cela, un CIEL_DATA_DIR=/data destiné au
    # conteneur ferait écrire la suite de tests hors du répertoire temporaire.
    monkeypatch.setenv("CIEL_DATA_DIR", "")
    monkeypatch.setenv("CIEL_STATIONS_FILE", "")
    monkeypatch.setenv("CT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CT_INGEST_MODE", "replay")
    monkeypatch.setenv("CT_POLL_INTERVAL_S", "12")
    get_settings.cache_clear()
    s = get_settings()
    s.ensure_dirs()
    if REAL_SNAPSHOT.exists():
        shutil.copy(REAL_SNAPSHOT, s.samples_dir / "opensky_snapshot.csv")
    yield s
    get_settings.cache_clear()
