"""Couche d'accès données du dashboard (lecture DuckDB, cache Streamlit).

Robuste : si la couche curated ou le modèle n'existent pas encore, les pages
affichent un message guidant l'utilisateur vers les commandes à lancer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Bootstrap : rendre le package importable même si lancé depuis dashboard/.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ciel_tranquille.config import get_settings  # noqa: E402
from ciel_tranquille.pipeline import monitoring_summary  # noqa: E402
from ciel_tranquille.storage.duck import connect  # noqa: E402

SETTINGS = get_settings()


def curated_ready() -> bool:
    return SETTINGS.duckdb_path.exists()


@st.cache_data(ttl=30, show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """Charge une table curated (cache 30 s -> rafraîchissement quasi temps réel)."""
    with connect(read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if name not in tables:
            return pd.DataFrame()
        return con.execute(f"SELECT * FROM {name}").fetch_df()


@st.cache_data(ttl=30, show_spinner=False)
def query(sql: str) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(sql).fetch_df()


@st.cache_data(ttl=15, show_spinner=False)
def monitoring() -> dict:
    return monitoring_summary(SETTINGS)


@st.cache_resource(show_spinner=False)
def model_payload():
    from ciel_tranquille.ml.predict import load_payload
    from ciel_tranquille.ml.train import MODEL_PATH

    if not Path(MODEL_PATH).exists():
        return None
    return load_payload()


def color_for_db(db: float) -> list[int]:
    """Échelle de couleur (vert -> jaune -> rouge) selon le niveau sonore."""
    if db < 50:
        return [34, 197, 94, 170]
    if db < 60:
        return [234, 179, 8, 190]
    if db < 68:
        return [249, 115, 22, 210]
    return [239, 68, 68, 230]


def require_curated() -> bool:
    """Affiche un guide si la couche curated est absente. Retourne True si prête."""
    if curated_ready():
        return True
    st.warning("⚠️ Couche *curated* introuvable. Lancez d'abord le pipeline :")
    st.code(
        "PYTHONPATH=src .venv/bin/python -m ciel_tranquille.ml.synth --days 10\n"
        "PYTHONPATH=src .venv/bin/python -c \"from ciel_tranquille.storage."
        "build_curated import build; build(noise_csv='bruit_synth.csv')\"\n"
        "PYTHONPATH=src .venv/bin/python -m ciel_tranquille.ml.train",
        language="bash",
    )
    return False
