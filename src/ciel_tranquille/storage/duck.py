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


def states_roots(settings: Settings | None = None) -> list[Path]:
    """Racines contenant des partitions d'états, de la plus fraîche à la plus ancienne.

    Trois emplacements coexistent volontairement :
    - `landing/` : ce que le poller écrit aujourd'hui ;
    - `curated/states_hourly/` : les mêmes snapshots après compaction horaire ;
    - `raw/states/` : les collectes antérieures au découpage `landing/`.

    Les lire toutes évite qu'une compaction ou un changement de disposition ne
    rende invisible une partie de l'historique déjà collecté.
    """
    settings = settings or get_settings()
    return [settings.landing_dir, settings.compacted_states_dir, settings.raw_dir / "states"]


def states_globs(settings: Settings | None = None) -> list[str]:
    """Motifs glob des racines qui contiennent effectivement des Parquet."""
    return [
        str(root / "**" / "*.parquet")
        for root in states_roots(settings)
        if root.exists() and any(root.rglob("*.parquet"))
    ]


def raw_states_glob(settings: Settings | None = None) -> str:
    """Motif glob principal (landing courante) — compatibilité ascendante."""
    settings = settings or get_settings()
    globs = states_globs(settings)
    return globs[0] if globs else str(settings.landing_dir / "**" / "*.parquet")


def table_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def list_tables(settings: Settings | None = None) -> list[str]:
    with connect(settings, read_only=True) as con:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]


def has_raw_data(settings: Settings | None = None) -> bool:
    return bool(states_globs(settings))


def sample_csv(name: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.samples_dir / name
