"""Construction de la couche *curated* (architecture medallion : raw -> curated).

Étapes :
1. Lecture des sources : états avions (Parquet landing) + mesures de bruit et
   historique de vols (CSV samples).
2. Nettoyage (module `transform.clean`).
3. Jointure spatio-temporelle bruit ↔ avions (`transform.join`) -> table de
   features `noise_enriched`.
4. Matérialisation dans DuckDB + **cube OLAP** multidimensionnel `cube_noise`
   via `GROUP BY CUBE`.

Idempotent : `CREATE OR REPLACE TABLE`. Réexécutable à volonté.
"""

from __future__ import annotations

import argparse
import logging

import duckdb
import pandas as pd

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.storage.duck import connect, has_raw_data, raw_states_glob, sample_csv
from ciel_tranquille.transform.clean import (
    clean_flights,
    clean_noise,
    clean_states,
    quality_report,
)
from ciel_tranquille.transform.join import spatio_temporal_join

logger = logging.getLogger(__name__)

# Cube OLAP : agrégats multidimensionnels du bruit par aéroport × période.
_CUBE_SQL = """
CREATE OR REPLACE TABLE cube_noise AS
SELECT
    COALESCE(airport, 'TOUS')          AS airport,
    CASE WHEN is_night = 1 THEN 'nuit' ELSE 'jour' END   AS periode,
    CASE WHEN is_weekend = 1 THEN 'we' ELSE 'semaine' END AS type_jour,
    GROUPING(airport, is_night, is_weekend)              AS niveau_agregation,
    count(*)                           AS n_mesures,
    round(avg(laeq_db), 2)             AS laeq_moyen_db,
    round(max(laeq_db), 2)             AS laeq_max_db,
    round(avg(num_aircraft), 2)        AS avions_moyen
FROM noise_enriched
GROUP BY CUBE(airport, is_night, is_weekend)
ORDER BY niveau_agregation, airport, periode, type_jour
"""


def _load_states(con: duckdb.DuckDBPyConnection, settings: Settings) -> pd.DataFrame:
    if not has_raw_data(settings):
        logger.warning("Aucun Parquet en landing zone : table states vide.")
        return pd.DataFrame()
    glob = raw_states_glob(settings)
    return con.execute(
        f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
    ).fetch_df()


def build(settings: Settings | None = None, noise_csv: str = "bruit_survol.csv") -> dict:
    """Construit la couche curated. Retourne un rapport (comptes + qualité).

    `noise_csv` : source des mesures de bruit dans `data/samples/`
    (`bruit_survol.csv` = réel, `bruit_synth.csv` = synthétique reproductible).
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    noise_raw = pd.read_csv(sample_csv(noise_csv, settings))
    flights_raw = pd.read_csv(sample_csv("flights_history.csv", settings))

    noise = clean_noise(noise_raw)
    flights = clean_flights(flights_raw)

    with connect(settings) as con:
        states_raw = _load_states(con, settings)
        states = clean_states(states_raw) if not states_raw.empty else states_raw

        enriched = spatio_temporal_join(noise, states if not states.empty else pd.DataFrame())

        con.register("noise_df", noise)
        con.register("flights_df", flights)
        con.register("enriched_df", enriched)
        if not states.empty:
            con.register("states_df", states)
            con.execute("CREATE OR REPLACE TABLE states AS SELECT * FROM states_df")
        else:
            con.execute(
                "CREATE OR REPLACE TABLE states AS SELECT * FROM noise_df WHERE 1=0"
            )

        con.execute("CREATE OR REPLACE TABLE noise_measurements AS SELECT * FROM noise_df")
        con.execute("CREATE OR REPLACE TABLE flights AS SELECT * FROM flights_df")
        con.execute("CREATE OR REPLACE TABLE noise_enriched AS SELECT * FROM enriched_df")
        con.execute(_CUBE_SQL)

        report = {
            "tables": {
                "noise_measurements": con.execute(
                    "SELECT count(*) FROM noise_measurements"
                ).fetchone()[0],
                "states": con.execute("SELECT count(*) FROM states").fetchone()[0],
                "flights": con.execute("SELECT count(*) FROM flights").fetchone()[0],
                "noise_enriched": con.execute(
                    "SELECT count(*) FROM noise_enriched"
                ).fetchone()[0],
                "cube_noise": con.execute("SELECT count(*) FROM cube_noise").fetchone()[0],
            },
            "quality": [
                quality_report(noise, "noise_measurements"),
                quality_report(flights, "flights"),
                quality_report(enriched, "noise_enriched"),
            ],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description="Construit la couche curated DuckDB.").parse_args(argv)
    report = build()
    print("Tables curated :")
    for name, n in report["tables"].items():
        print(f"  - {name:20s} {n:>6d} lignes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
