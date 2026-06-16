"""Définition des features et de la cible.

Garde-fou **anti-fuite de cible** :
- la cible est `laeq_db` ;
- on EXCLUT `lmax_db` (autre mesure du même phénomène acoustique = fuite
  directe) et toute colonne dérivée du bruit mesuré ;
- les features avions proviennent de la **jointure spatio-temporelle** calculée
  par le pipeline (rayon 20 km / ±5 min) — un proxy bruité des inputs ayant servi
  à générer le bruit (rayon 25 km / instant exact). Le modèle apprend donc une
  relation réelle, pas une recopie de la cible.
"""

from __future__ import annotations

import pandas as pd

from ciel_tranquille.storage.duck import connect

TARGET = "laeq_db"

# Features numériques temporelles + trafic aérien (issues de la jointure).
NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_night",
    "is_rush_hour",
    "is_weekend",
    "num_aircraft",
    "num_close_aircraft",
    "avg_altitude_m",
    "min_altitude_m",
    "avg_velocity_m_s",
    "min_distance_km",
    "avg_distance_km",
]
CATEGORICAL_FEATURES = ["airport"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Colonnes interdites en entrée (fuite de cible ou identifiants).
LEAKAGE_COLS = ["lmax_db", "laeq_db", "station_id", "station_name"]


def load_training_frame(settings=None) -> pd.DataFrame:
    """Charge `noise_enriched` depuis DuckDB et garde features + cible."""
    with connect(settings, read_only=True) as con:
        df = con.execute("SELECT * FROM noise_enriched").fetch_df()
    cols = [c for c in FEATURES if c in df.columns] + [TARGET]
    df = df[cols].dropna(subset=[TARGET]).reset_index(drop=True)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()
