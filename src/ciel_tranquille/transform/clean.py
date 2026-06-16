"""Nettoyage et normalisation des sources.

Chaque fonction est **pure** (DataFrame -> DataFrame), documentée et testable.
Choix de nettoyage explicités :
- déduplication (les snapshots/exports contiennent des doublons),
- coercition de types et parsing des horodatages en UTC,
- bornage des valeurs physiquement impossibles (plutôt que suppression
  aveugle) : un LAeq < 20 dB ou > 130 dB est aberrant pour une station urbaine,
- features temporelles (heure, nuit, heure de pointe, week-end) utiles à
  l'analyse comme au modèle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bornes physiques plausibles (dB) pour une station de mesure urbaine.
LAEQ_MIN_DB, LAEQ_MAX_DB = 20.0, 130.0
NIGHT_START, NIGHT_END = 22, 6  # arrêté bruit : période nocturne 22 h–6 h
RUSH_HOURS = {7, 8, 9, 17, 18, 19}


def _add_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    h = df[ts_col].dt.hour
    df["hour"] = h
    df["day_of_week"] = df[ts_col].dt.dayofweek
    df["is_night"] = ((h >= NIGHT_START) | (h < NIGHT_END)).astype(int)
    df["is_rush_hour"] = h.isin(RUSH_HOURS).astype(int)
    df["is_weekend"] = (df[ts_col].dt.dayofweek >= 5).astype(int)
    return df


def clean_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les mesures de bruit (`bruit_survol.csv`).

    Normalise les noms de colonnes en snake_case (`laeq_db`, `lmax_db`).
    """
    df = df.copy()
    df = df.rename(columns={"LAeq_dB": "laeq_db", "Lmax_dB": "lmax_db"})
    df["timestamp"] = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce")
    for col in ("laeq_db", "lmax_db", "latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp", "laeq_db", "latitude", "longitude"])
    # Bornage des aberrations physiques (conserve la ligne, borne la valeur).
    df["laeq_db"] = df["laeq_db"].clip(LAEQ_MIN_DB, LAEQ_MAX_DB)
    df = _add_time_features(df, "timestamp")
    return df.reset_index(drop=True)


def clean_states(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les états avions ingérés (landing Parquet)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["time_position_unix"], unit="s", utc=True, errors="coerce")
    for col in ("longitude", "latitude", "baro_altitude_m", "velocity_m_s", "heading_deg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["icao24", "latitude", "longitude", "snapshot_ts"])
    df = df.drop_duplicates(subset=["icao24", "snapshot_ts"])
    # Avions au sol = pas de contribution au bruit de survol.
    if "on_ground" in df.columns:
        df = df[~df["on_ground"].fillna(False)]
    # Altitude négative (bruit capteur) -> 0.
    df["baro_altitude_m"] = df["baro_altitude_m"].clip(lower=0)
    df["velocity_m_s"] = df["velocity_m_s"].clip(lower=0)
    return df.reset_index(drop=True)


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie l'historique de vols (`flights_history.csv`)."""
    df = df.copy()
    df["first_seen"] = pd.to_datetime(df["first_seen_iso"], utc=True, errors="coerce")
    df["last_seen"] = pd.to_datetime(df["last_seen_iso"], utc=True, errors="coerce")
    for col in ("avg_altitude_m", "avg_speed_knots", "approx_distance_km"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=["flight_id", "first_seen", "last_seen"])
    df["duration_min"] = (df["last_seen"] - df["first_seen"]).dt.total_seconds() / 60.0
    df = df[df["duration_min"].between(0, 24 * 60)]  # vols < 24 h
    return df.reset_index(drop=True)


def quality_report(df: pd.DataFrame, name: str) -> dict:
    """Petit rapport qualité : complétude et cardinalité."""
    return {
        "source": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "null_ratio": float(np.round(df.isna().mean().mean(), 4)),
        "duplicated_rows": int(df.duplicated().sum()),
    }
