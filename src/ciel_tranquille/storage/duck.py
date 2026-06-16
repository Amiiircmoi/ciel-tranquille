"""Accès DuckDB — moteur analytique in-process du projet.

Pourquoi DuckDB : colonnaire, vectorisé, **sans serveur** (un fichier), lit
Parquet/CSV nativement, SQL analytique complet (window functions, `CUBE`).
Parfait pour une vitrine data sur VPS sans coût d'exploitation, tout en offrant
un vrai langage de requête (contrairement à du Parquet seul).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb

from ciel_tranquille.config import Settings, get_settings


@contextmanager
def connect(settings: Settings | None = None, read_only: bool = False):
    """Connexion DuckDB sur le fichier curated du projet."""
    settings = settings or get_settings()
    settings.curated_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.duckdb_path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def raw_states_glob(settings: Settings | None = None) -> str:
    """Motif glob des Parquet de la landing zone (partition Hive par date)."""
    settings = settings or get_settings()
    return str(settings.raw_dir / "states" / "**" / "*.parquet")


def table_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def list_tables(settings: Settings | None = None) -> list[str]:
    with connect(settings, read_only=True) as con:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]


def has_raw_data(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    states_dir = settings.raw_dir / "states"
    return states_dir.exists() and any(states_dir.rglob("*.parquet"))


def sample_csv(name: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.samples_dir / name
