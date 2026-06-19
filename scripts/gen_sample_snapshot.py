#!/usr/bin/env python3
"""Génère le snapshot d'états avions *synthétique* utilisé par le mode replay.

Le tier gratuit d'OpenSky n'autorise pas la redistribution de ses vecteurs
d'état bruts. Le dépôt n'embarque donc **aucune donnée OpenSky réelle** : ce
script fabrique un snapshot de **schéma identique** à `/states/all` (colonnes
OpenSky), avec des aéronefs entièrement fictifs (ICAO24 hexadécimaux tirés
aléatoirement, callsigns synthétiques), positionnés dans la bounding box
francilienne couverte par le projet.

Le résultat (`data/samples/opensky_snapshot.csv`) sert de base déterministe au
mode replay (`ciel_tranquille.ingest.replay`) et aux tests hors-ligne : on peut
développer et démontrer le pipeline micro-batch sans réseau ni identifiants.

Reproductible : tout l'aléatoire vient d'un `random.Random` seedé.
Usage : ``python scripts/gen_sample_snapshot.py [--seed 7] [--rows 100]``
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Bounding box combinée (WGS84) couvrant les 3 stations de mesure.
LAT_MIN, LAT_MAX = 48.65, 49.09
LON_MIN, LON_MAX = 2.18, 2.53

# Instant de référence du snapshot (UNIX, UTC) — fixe pour la reproductibilité.
BASE_TS = 1_758_540_000  # 2025-09-22 ~12:00 UTC

# Préfixes de compagnies (format OACI) pour des callsigns synthétiques crédibles.
_AIRLINE_PREFIXES = ("AFR", "RYR", "BAW", "DLH", "EZY", "TVF", "VLG", "KLM")
_COUNTRIES = ("France", "United Kingdom", "Germany", "Spain", "Netherlands", "Ireland")

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "opensky_snapshot.csv"

COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position_unix",
    "longitude",
    "latitude",
    "baro_altitude_m",
    "velocity_m_s",
    "heading_deg",
    "squawk",
    "last_contact_unix",
]


def generate(seed: int = 7, rows: int = 100) -> list[dict]:
    """Construit `rows` vecteurs d'état synthétiques, déterministes pour `seed`."""
    rng = random.Random(seed)
    records: list[dict] = []
    for _ in range(rows):
        ts = BASE_TS - rng.randint(0, 900)  # jusqu'à 15 min avant la référence
        records.append(
            {
                "icao24": f"{rng.randrange(0x1000000):06x}",
                "callsign": f"{rng.choice(_AIRLINE_PREFIXES)}{rng.randint(1, 9999)}",
                "origin_country": rng.choice(_COUNTRIES),
                "time_position_unix": ts,
                "longitude": round(rng.uniform(LON_MIN, LON_MAX), 6),
                "latitude": round(rng.uniform(LAT_MIN, LAT_MAX), 6),
                "baro_altitude_m": round(rng.uniform(300, 13000), 1),
                "velocity_m_s": round(rng.uniform(70, 270), 1),
                "heading_deg": round(rng.uniform(0, 360), 1),
                "squawk": f"{rng.randint(1000, 7777)}",
                "last_contact_unix": ts + rng.randint(0, 3),
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Génère un snapshot d'états avions synthétique.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rows", type=int, default=100)
    args = parser.parse_args(argv)

    records = generate(seed=args.seed, rows=args.rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Snapshot synthétique écrit : {OUT_PATH} ({len(records)} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
