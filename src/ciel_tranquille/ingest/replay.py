"""Mode *replay* déterministe — micro-batches hors-ligne sans appel réseau.

Objectif : pouvoir développer, **tester** et démontrer le pipeline micro-batch
sans dépendre de l'API OpenSky (ni de ses identifiants ni de son rate-limit).

Honnêteté : la base est un snapshot **synthétique** de schéma identique à
`/states/all` (`opensky_snapshot.csv`, généré par `scripts/gen_sample_snapshot.py`
— aucune donnée OpenSky réelle n'est redistribuée). Le replay synthétise des
snapshots successifs par *dead-reckoning* : chaque
aéronef avance le long de son cap (`heading_deg`) à sa vitesse (`velocity_m_s`)
pendant l'intervalle de polling. C'est **déterministe** (aucune source
aléatoire) → mêmes entrées ⇒ mêmes sorties, idéal pour la CI.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from pathlib import Path

from ciel_tranquille.config import get_settings

EARTH_RADIUS_M = 6_371_000.0

# Colonnes numériques du snapshot de référence à convertir.
_FLOAT_COLS = (
    "longitude",
    "latitude",
    "baro_altitude_m",
    "velocity_m_s",
    "heading_deg",
)


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def load_base_snapshot(path: Path | None = None) -> list[dict]:
    """Charge le snapshot OpenSky de référence depuis `data/samples/`."""
    path = path or (get_settings().samples_dir / "opensky_snapshot.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for col in _FLOAT_COLS:
            r[col] = _to_float(r.get(col))
        r["time_position_unix"] = int(_to_float(r.get("time_position_unix")) or 0)
    return rows


def _advance(lat: float, lon: float, heading_deg: float, distance_m: float) -> tuple[float, float]:
    """Dead-reckoning : nouvelle position après `distance_m` au cap `heading_deg`."""
    bearing = math.radians(heading_deg)
    ang = distance_m / EARTH_RADIUS_M
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def replay_snapshots(
    n_batches: int,
    interval_s: int | None = None,
    base: list[dict] | None = None,
) -> Iterator[list[dict]]:
    """Génère `n_batches` snapshots successifs (déterministes).

    Yields une liste de records au schéma interne (cf. `states_to_records`).
    """
    settings = get_settings()
    interval_s = interval_s or settings.poll_interval_s
    base = base if base is not None else load_base_snapshot()
    base_ts = max((r["time_position_unix"] for r in base), default=0)

    for batch_idx in range(n_batches):
        elapsed = batch_idx * interval_s
        snapshot_ts = base_ts + elapsed
        records: list[dict] = []
        for r in base:
            lat, lon = r.get("latitude"), r.get("longitude")
            vel = r.get("velocity_m_s") or 0.0
            hdg = r.get("heading_deg")
            if lat is not None and lon is not None and hdg is not None and elapsed:
                lat, lon = _advance(lat, lon, hdg, vel * elapsed)
            records.append(
                {
                    "icao24": r.get("icao24"),
                    "callsign": (r.get("callsign") or "").strip() or None,
                    "origin_country": r.get("origin_country"),
                    "time_position_unix": snapshot_ts,
                    "longitude": lon,
                    "latitude": lat,
                    "baro_altitude_m": r.get("baro_altitude_m"),
                    "velocity_m_s": vel,
                    "heading_deg": hdg,
                    "squawk": r.get("squawk"),
                    "last_contact_unix": snapshot_ts,
                    "on_ground": False,
                    "snapshot_ts": snapshot_ts,
                }
            )
        yield records
