"""Ciel Tranquille — vitrine analytique du bruit aérien urbain.

Pipeline de bout en bout :
ingestion micro-batch OpenSky -> stockage DuckDB/Parquet -> features ->
modèle ML supervisé -> dashboard.
"""

__version__ = "0.1.0"
