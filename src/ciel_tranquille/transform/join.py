"""Jointure spatio-temporelle bruit ↔ trafic aérien (multi-sources).

Pour chaque mesure de bruit, on agrège les aéronefs « proches » dans une fenêtre
spatio-temporelle (±`time_window_min`, rayon `radius_km`). Cela produit la table
de features qui relie les deux sources hétérogènes (mesures sol + positions
avions) — base de l'analyse et du modèle.

Implémentation vectorisée (numpy) plutôt que `DataFrame.iterrows()` (comme le
prototype d'origine) : O(n_bruit × n_states) mais en opérations numpy, donc
exploitable et testable. Le calcul de distance est une **haversine** explicite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

# Features agrégées produites par mesure de bruit.
AIRCRAFT_FEATURES = [
    "num_aircraft",
    "num_close_aircraft",
    "avg_altitude_m",
    "min_altitude_m",
    "avg_velocity_m_s",
    "min_distance_km",
    "avg_distance_km",
]


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Distance grand-cercle (km) entre deux points (vectorisable)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _empty_features() -> dict:
    return {f: 0.0 for f in AIRCRAFT_FEATURES}


def spatio_temporal_join(
    noise: pd.DataFrame,
    states: pd.DataFrame,
    time_window_min: int = 5,
    radius_km: float = 20.0,
    close_radius_km: float = 5.0,
) -> pd.DataFrame:
    """Enrichit chaque mesure de bruit avec les avions proches.

    `noise` doit contenir `timestamp`, `latitude`, `longitude`.
    `states` doit contenir `timestamp`, `latitude`, `longitude`,
    `baro_altitude_m`, `velocity_m_s`.
    """
    noise = noise.reset_index(drop=True).copy()
    if states.empty:
        feats = pd.DataFrame([_empty_features()] * len(noise))
        return pd.concat([noise, feats], axis=1)

    # Comparaison sur epoch en **nanosecondes** -> insensible au fuseau ET à la
    # résolution datetime (les sources peuvent être en [s], [us] ou [ns]).
    window_ns = int(time_window_min * 60 * 1e9)
    s_ts = (
        states["timestamp"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy("datetime64[ns]")
        .astype("int64")
    )
    s_lat = states["latitude"].to_numpy(dtype=float)
    s_lon = states["longitude"].to_numpy(dtype=float)
    s_alt = states["baro_altitude_m"].to_numpy(dtype=float)
    s_vel = states["velocity_m_s"].to_numpy(dtype=float)

    rows: list[dict] = []
    for _, m in noise.iterrows():
        t_ns = int(m["timestamp"].value)
        time_mask = (s_ts >= t_ns - window_ns) & (s_ts <= t_ns + window_ns)
        if not time_mask.any():
            rows.append(_empty_features())
            continue
        idx = np.flatnonzero(time_mask)
        dist = haversine_km(m["latitude"], m["longitude"], s_lat[idx], s_lon[idx])
        near = dist <= radius_km
        if not near.any():
            rows.append(_empty_features())
            continue
        d = dist[near]
        rows.append(
            {
                "num_aircraft": int(near.sum()),
                "num_close_aircraft": int((d <= close_radius_km).sum()),
                "avg_altitude_m": float(np.nanmean(s_alt[idx][near])),
                "min_altitude_m": float(np.nanmin(s_alt[idx][near])),
                "avg_velocity_m_s": float(np.nanmean(s_vel[idx][near])),
                "min_distance_km": float(np.min(d)),
                "avg_distance_km": float(np.mean(d)),
            }
        )
    feats = pd.DataFrame(rows).fillna(0.0)
    return pd.concat([noise, feats], axis=1)
