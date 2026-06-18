"""Co-localisation événement de bruit ↔ aéronef (cœur de la v2).

Pour chaque événement de survol (cible `max_laeq` à l'instant `max_ts`), on
estime la position de **chaque aéronef** présent dans la fenêtre à l'instant
exact `max_ts` — par **interpolation linéaire** entre les deux snapshots qui
l'encadrent (cadence 30 s), ou à défaut par **dead-reckoning** (vitesse / cap /
taux de montée) — puis on retient l'aéronef **le plus proche en distance
oblique** (slant range) station↔avion. C'est l'aéronef associé au survol.

Pourquoi la distance oblique : le bruit perçu au sol dépend de la distance 3D à
la source (horizontale + altitude), pas de la seule distance au sol. C'est la
variable physique de premier ordre du niveau sonore d'un survol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ciel_tranquille.config import Station
from ciel_tranquille.transform.join import haversine_km

EARTH_RADIUS_M = 6_371_000.0

# Colonnes d'états attendues (issues de la landing OpenSky normalisée).
_STATE_COLS = ("icao24", "snapshot_ts", "latitude", "longitude", "baro_altitude_m",
               "velocity_m_s", "heading_deg", "vertical_rate_m_s")


def _advance(lat: float, lon: float, heading_deg: float, distance_m: float) -> tuple[float, float]:
    """Dead-reckoning : position après `distance_m` au cap `heading_deg`."""
    bearing = math.radians(heading_deg)
    ang = distance_m / EARTH_RADIUS_M
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def estimate_position(states_icao: pd.DataFrame, t: float) -> tuple[float, float, float] | None:
    """Position (lat, lon, alt_m) d'UN aéronef à l'instant `t` (epoch s).

    `states_icao` : snapshots d'un seul `icao24` (triés ou non). Interpolation
    linéaire si `t` est encadré, sinon dead-reckoning depuis le snapshot le plus
    proche. Retourne None si position/altitude inexploitables.
    """
    g = states_icao.sort_values("snapshot_ts")
    ts = g["snapshot_ts"].to_numpy(dtype=float)
    before = g[ts <= t]
    after = g[ts >= t]
    if len(before) and len(after):
        b, a = before.iloc[-1], after.iloc[0]
        span = a["snapshot_ts"] - b["snapshot_ts"]
        f = 0.0 if span == 0 else (t - b["snapshot_ts"]) / span
        lat = b["latitude"] + f * (a["latitude"] - b["latitude"])
        lon = b["longitude"] + f * (a["longitude"] - b["longitude"])
        alt = _lerp_alt(b, a, f)
    else:
        row = before.iloc[-1] if len(before) else after.iloc[0]
        lat, lon = row["latitude"], row["longitude"]
        vel = row.get("velocity_m_s") or 0.0
        hdg = row.get("heading_deg")
        dt = t - row["snapshot_ts"]
        if hdg is not None and not pd.isna(hdg) and vel:
            lat, lon = _advance(lat, lon, float(hdg), float(vel) * dt)
        alt = row.get("baro_altitude_m")
        vr = row.get("vertical_rate_m_s")
        if alt is not None and not pd.isna(alt) and vr is not None and not pd.isna(vr):
            alt = alt + float(vr) * dt
    if pd.isna(lat) or pd.isna(lon):
        return None
    return float(lat), float(lon), float(alt) if alt is not None and not pd.isna(alt) else 0.0


def _lerp_alt(b, a, f: float) -> float:
    ba, aa = b.get("baro_altitude_m"), a.get("baro_altitude_m")
    if ba is None or pd.isna(ba):
        return aa if aa is not None and not pd.isna(aa) else 0.0
    if aa is None or pd.isna(aa):
        return ba
    return ba + f * (aa - ba)


def slant_distance_km(station: Station, lat: float, lon: float, alt_m: float) -> float:
    """Distance oblique 3D station↔avion (km) = √(horizontale² + altitude²)."""
    horiz = float(haversine_km(station.latitude, station.longitude, lat, lon))
    vert = max(alt_m, 0.0) / 1000.0
    return math.sqrt(horiz * horiz + vert * vert)


# --- Primitives géométriques partagées -------------------------------------
# Maison unique des fonctions de géométrie réutilisées par les scripts de
# validation (GATE strict `validate_join.py` et cross-check `validate_crosscheck.py`)
# afin d'éviter la copie du même calcul trigonométrique dans plusieurs fichiers.
# Chaque script garde en revanche SA méthode d'association (anti-ambiguïté
# « unique dominant » vs plus-proche-en-oblique) : ce sont deux validations
# indépendantes, pas un seul code dupliqué.

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance grand-cercle en mètres (wrapper de `haversine_km`)."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def bearing_deg(lat1, lon1, lat2, lon2):
    """Relèvement initial station(1)→avion(2), degrés depuis le Nord (sens horaire)."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def ang_diff(a, b):
    """Écart angulaire minimal (degrés) entre deux azimuts, dans [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@dataclass
class Association:
    event_id: int
    station: str
    max_laeq: float
    max_ts_unix: float
    icao24: str | None
    callsign: str | None
    slant_km: float | None
    horiz_km: float | None
    altitude_m: float | None
    velocity_m_s: float | None
    heading_deg: float | None
    aircraft_lat: float | None
    aircraft_lon: float | None
    n_candidates: int       # aéronefs présents dans la fenêtre temporelle
    dt_nearest_s: float | None  # écart au snapshot le plus proche de l'aéronef associé


def associate_event(
    event: pd.Series,
    states: pd.DataFrame,
    station: Station,
    time_tol_s: float = 45.0,
) -> Association:
    """Associe un événement à l'aéronef le plus proche en distance oblique à `max_ts`.

    `states` : snapshots OpenSky (déjà restreints à la station/fenêtre si voulu).
    `time_tol_s` : on ne considère que les aéronefs ayant au moins un snapshot à
    ±`time_tol_s` de `max_ts` (cadence 30 s → 45 s encadre un tick).
    """
    t = float(event["max_ts_unix"])
    win = states[(states["snapshot_ts"] >= t - time_tol_s) & (states["snapshot_ts"] <= t + time_tol_s)]
    base = Association(
        event_id=int(event["event_id"]), station=station.measurement_id,
        max_laeq=float(event["max_laeq"]), max_ts_unix=t,
        icao24=None, callsign=None, slant_km=None, horiz_km=None, altitude_m=None,
        velocity_m_s=None, heading_deg=None, aircraft_lat=None, aircraft_lon=None,
        n_candidates=int(win["icao24"].nunique()), dt_nearest_s=None,
    )
    if win.empty:
        return base

    best = None
    for icao, g in win.groupby("icao24"):
        pos = estimate_position(g, t)
        if pos is None:
            continue
        lat, lon, alt = pos
        slant = slant_distance_km(station, lat, lon, alt)
        if best is None or slant < best["slant"]:
            nearest = g.iloc[(g["snapshot_ts"] - t).abs().argmin()]
            best = {
                "icao": icao, "slant": slant,
                "horiz": float(haversine_km(station.latitude, station.longitude, lat, lon)),
                "alt": alt, "lat": lat, "lon": lon,
                "vel": _val(nearest, "velocity_m_s"), "hdg": _val(nearest, "heading_deg"),
                "callsign": nearest.get("callsign"), "dt": float(nearest["snapshot_ts"] - t),
            }
    if best is None:
        return base
    base.icao24 = best["icao"]
    base.callsign = best["callsign"]
    base.slant_km = round(best["slant"], 3)
    base.horiz_km = round(best["horiz"], 3)
    base.altitude_m = round(best["alt"], 1)
    base.velocity_m_s = best["vel"]
    base.heading_deg = best["hdg"]
    base.aircraft_lat = round(best["lat"], 5)
    base.aircraft_lon = round(best["lon"], 5)
    base.dt_nearest_s = round(best["dt"], 1)
    return base


def _val(row, col):
    v = row.get(col)
    return None if v is None or pd.isna(v) else float(v)


def build_pairs(
    events: pd.DataFrame,
    states: pd.DataFrame,
    stations: dict[str, Station],
    time_tol_s: float = 45.0,
) -> pd.DataFrame:
    """Construit la table de paires événement↔aéronef pour un lot d'événements.

    `events` doit contenir : event_id, station, max_laeq, max_ts_unix.
    `states` doit contenir les colonnes `_STATE_COLS`. Le filtrage par station
    n'est pas appliqué (l'avion peut survoler depuis la bbox commune) : on associe
    sur l'ensemble des états, la distance oblique fait le tri.
    """
    rows = []
    for _, ev in events.iterrows():
        st = stations[ev["station"]]
        rows.append(associate_event(ev, states, st, time_tol_s=time_tol_s).__dict__)
    return pd.DataFrame(rows)
