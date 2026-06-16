"""Générateur de données synthétiques **reproductible** et documenté.

Pourquoi des données synthétiques : le tier gratuit d'OpenSky ne fournit qu'un
*snapshot* instantané — impossible d'obtenir un historique horaire aligné sur
des mesures de bruit pour entraîner un modèle. On génère donc un « monde »
physique cohérent et **documenté** : du trafic aérien autour des 3 aéroports
franciliens (CDG, ORY, LBG) et le bruit qu'il produit à des stations au sol,
via un modèle acoustique simplifié mais explicite.

Garde-fous :
- **Reproductible** : tout l'aléatoire vient d'un `numpy.Generator` seedé.
- **Agrégation spatiale** : stations fixes documentées (pas de géolocalisation
  d'un domicile réel), aéronefs synthétiques (icao24 fictifs).
- Le générateur écrit (1) des états avions dans la *landing zone* (le pipeline
  les ingère comme n'importe quel snapshot) et (2) des mesures de bruit
  `bruit_synth.csv`. La jointure spatio-temporelle RECONSTRUIT ensuite les
  features avions : le modèle apprend donc une relation que le pipeline a
  réellement calculée (pas une fuite directe de la cible).
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.ingest.poller import write_batch
from ciel_tranquille.ingest.replay import _advance
from ciel_tranquille.transform.join import haversine_km

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Station:
    station_id: str
    station_name: str
    airport: str
    lat: float
    lon: float


# Points de référence des aéroports (coordonnées réelles des plateformes).
AIRPORTS = {
    "CDG": (49.0097, 2.5479, 32.0),  # (lat, lon, trafic max/h dans la fenêtre)
    "ORY": (48.7233, 2.3794, 22.0),
    "LBG": (48.9694, 2.4414, 9.0),  # Le Bourget : aviation d'affaires
}

# Stations de mesure (fixes, documentées — pas de domicile réel ciblé).
STATIONS = [
    Station("CT_CDG_N", "CDG Nord — Le Mesnil-Amelot", "CDG", 49.0300, 2.5600),
    Station("CT_CDG_S", "CDG Sud — Tremblay-en-France", "CDG", 48.9850, 2.5300),
    Station("CT_ORY_N", "Orly Nord — Villeneuve-le-Roi", "ORY", 48.7450, 2.3780),
    Station("CT_ORY_S", "Orly Sud — Paray-Vieille-Poste", "ORY", 48.7050, 2.3800),
    Station("CT_LBG_E", "Le Bourget Est — Drancy", "LBG", 48.9600, 2.4600),
    Station("CT_LBG_W", "Le Bourget Ouest — Dugny", "LBG", 48.9550, 2.4200),
]

# Profil diurne relatif (intensité de trafic par heure locale, 0..1).
_HOURLY_PROFILE = np.array(
    [
        0.05, 0.03, 0.02, 0.02, 0.04, 0.10,  # 0-5 h
        0.35, 0.70, 0.95, 0.90, 0.80, 0.78,  # 6-11 h
        0.80, 0.82, 0.80, 0.78, 0.82, 0.95,  # 12-17 h
        0.98, 0.88, 0.70, 0.50, 0.25, 0.10,  # 18-23 h
    ]
)

# Modèle acoustique (paramètres documentés et calibrés).
AMBIENT_DAY_DB = 44.0
AMBIENT_NIGHT_DB = 39.0
SWL_LIGHT_DB = 132.0  # niveau de puissance source (avion léger)
SWL_HEAVY_DB = 139.0  # gros porteur
RADIUS_KM = 25.0  # rayon d'influence acoustique pris en compte


def _curfew_factor(airport: str, hour: int) -> float:
    """Couvre-feu nocturne : ORY fermé ~23h30-6h ; CDG/LBG réduits la nuit."""
    if airport == "ORY":
        return 0.0 if (hour >= 23 or hour < 6) else 1.0
    return 0.25 if (hour >= 23 or hour < 6) else 1.0


def _energetic_sum_db(levels_db: np.ndarray, ambient_db: float) -> float:
    energies = np.power(10.0, levels_db / 10.0)
    total = energies.sum() + 10.0 ** (ambient_db / 10.0)
    return float(10.0 * np.log10(total))


def generate(
    seed: int = 42,
    start: str = "2025-09-01",
    days: int = 10,
    settings: Settings | None = None,
    reset_landing: bool = True,
) -> dict:
    """Génère le monde synthétique. Retourne un résumé (volumes, plages)."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    rng = np.random.default_rng(seed)

    if reset_landing:
        states_dir = settings.raw_dir / "states"
        if states_dir.exists():
            shutil.rmtree(states_dir)

    start_ts = pd.Timestamp(start, tz="UTC")
    noise_rows: list[dict] = []
    n_states_total = 0
    aircraft_counter = 0

    for day in range(days):
        for hour in range(24):
            ts = start_ts + pd.Timedelta(days=day, hours=hour)
            snapshot_ts = int(ts.timestamp())
            records: list[dict] = []
            # Pool d'aéronefs de l'heure (tous aéroports confondus).
            pool: list[dict] = []
            for airport, (alat, alon, lam_max) in AIRPORTS.items():
                lam = lam_max * _HOURLY_PROFILE[hour] * _curfew_factor(airport, hour)
                n_ac = int(rng.poisson(lam))
                for _ in range(n_ac):
                    dist_km = float(RADIUS_KM * np.sqrt(rng.uniform(0, 1)))  # densité ~ proche
                    bearing = float(rng.uniform(0, 360))
                    altitude = float(np.clip(dist_km * 150.0 + rng.normal(0, 200), 0, 4200))
                    velocity = float(rng.uniform(70, 135))
                    is_heavy = rng.uniform() < (0.55 if airport == "CDG" else 0.15)
                    lat, lon = _advance(alat, alon, bearing, dist_km * 1000.0)
                    aircraft_counter += 1
                    icao = f"{aircraft_counter & 0xFFFFFF:06x}"
                    rec = {
                        "icao24": icao,
                        "callsign": f"SYN{aircraft_counter % 10000:04d}",
                        "origin_country": "France",
                        "time_position_unix": snapshot_ts,
                        "longitude": lon,
                        "latitude": lat,
                        "baro_altitude_m": altitude,
                        "velocity_m_s": velocity,
                        "heading_deg": bearing,
                        "squawk": None,
                        "last_contact_unix": snapshot_ts,
                        "on_ground": False,
                        "snapshot_ts": snapshot_ts,
                    }
                    records.append(rec)
                    pool.append({**rec, "is_heavy": is_heavy, "airport": airport})

            if records:
                write_batch(records, settings.raw_dir)
                n_states_total += len(records)

            # Calcul du bruit à chaque station depuis les aéronefs proches.
            ambient = AMBIENT_DAY_DB if 6 <= hour < 22 else AMBIENT_NIGHT_DB
            for st in STATIONS:
                if not pool:
                    laeq = ambient + float(rng.normal(0, 0.8))
                    lmax = laeq + float(abs(rng.normal(2, 1)))
                else:
                    p_lat = np.array([p["latitude"] for p in pool])
                    p_lon = np.array([p["longitude"] for p in pool])
                    p_alt = np.array([p["baro_altitude_m"] for p in pool])
                    p_heavy = np.array([p["is_heavy"] for p in pool])
                    h_dist = haversine_km(st.lat, st.lon, p_lat, p_lon)
                    near = h_dist <= RADIUS_KM
                    if near.any():
                        slant_m = np.sqrt((h_dist[near] * 1000.0) ** 2 + p_alt[near] ** 2)
                        slant_m = np.clip(slant_m, 50.0, None)
                        swl = np.where(p_heavy[near], SWL_HEAVY_DB, SWL_LIGHT_DB)
                        li = swl - 20.0 * np.log10(slant_m) - 8.0
                        laeq = _energetic_sum_db(li, ambient) + float(rng.normal(0, 1.0))
                        lmax = max(ambient, float(li.max())) + float(abs(rng.normal(2.5, 1.0)))
                    else:
                        laeq = ambient + float(rng.normal(0, 0.8))
                        lmax = laeq + float(abs(rng.normal(2, 1)))
                noise_rows.append(
                    {
                        "station_id": st.station_id,
                        "station_name": st.station_name,
                        "timestamp_iso": ts.tz_convert(None).isoformat(),
                        "LAeq_dB": round(float(np.clip(laeq, 30, 95)), 1),
                        "Lmax_dB": round(float(np.clip(lmax, 30, 110)), 1),
                        "latitude": st.lat,
                        "longitude": st.lon,
                        "airport": st.airport,
                    }
                )

    noise_df = pd.DataFrame(noise_rows)
    out_csv = settings.samples_dir / "bruit_synth.csv"
    noise_df.to_csv(out_csv, index=False)

    summary = {
        "seed": seed,
        "days": days,
        "noise_rows": len(noise_df),
        "states_rows": n_states_total,
        "stations": len(STATIONS),
        "laeq_min": round(float(noise_df["LAeq_dB"].min()), 1),
        "laeq_max": round(float(noise_df["LAeq_dB"].max()), 1),
        "laeq_mean": round(float(noise_df["LAeq_dB"].mean()), 1),
        "csv": str(out_csv),
    }
    logger.info("Génération synthétique : %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Génère le jeu de données synthétique.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2025-09-01")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--keep-landing", action="store_true", help="Ne pas vider data/raw.")
    args = parser.parse_args(argv)
    summary = generate(
        seed=args.seed, start=args.start, days=args.days, reset_landing=not args.keep_landing
    )
    print("Jeu synthétique généré :")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
