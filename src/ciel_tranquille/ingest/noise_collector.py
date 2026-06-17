"""Collecteur d'événements de survol Bruitparif (source bruit).

Rôle : pour chaque station retenue et chaque jour, récupérer les événements de
catégorie ``air`` (un survol = un pic sonore attribuable à un avion), les
**normaliser**, les **dédupliquer** (clé = ``id`` d'événement) et les écrire en
**Parquet partitionné** ``raw/noise_events/station=…/date=…/`` — symétrique de la
landing zone OpenSky. La cible prédictive est ``max_laeq`` (le LAmax du survol) ;
``max_ts`` (instant du pic) est la **clé de jointure** avec les positions avions.

Honnêteté / licences : la donnée brute (JSON) est conservée en local
(`data/real_survol/`, gitignoré) ; le dépôt ne versionne que du code et des
échantillons dérivés. Source : Bruitparif,
Licence Ouverte Etalab (attribution).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ciel_tranquille.config import STATIONS, Settings, Station, get_settings
from ciel_tranquille.ingest.bruitparif_client import BruitparifClient
from ciel_tranquille.storage.duck import connect

logger = logging.getLogger(__name__)

_NOISE_SCHEMA = pa.schema(
    [
        ("station", pa.string()),
        ("airport", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("event_id", pa.int64()),
        ("category", pa.string()),
        ("start_ts_unix", pa.float64()),
        ("end_ts_unix", pa.float64()),
        ("max_ts_unix", pa.float64()),  # clé de jointure avec les états OpenSky
        ("max_ts_iso", pa.string()),
        ("duration_s", pa.float64()),
        ("laeq", pa.float64()),
        ("max_laeq", pa.float64()),  # CIBLE prédictive (LAmax du survol)
        ("sel", pa.float64()),
        ("nrj_laeq", pa.float64()),
        ("valid", pa.bool_()),
        ("collect_day", pa.string()),
    ]
)


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse un horodatage Bruitparif ('…T..:..:..Z' / '+0000' / '+00:00')."""
    if not ts:
        return None
    raw = ts.strip().replace("Z", "+00:00")
    # offset sans deux-points ('+0000') -> ('+00:00') pour fromisoformat < 3.11 strict
    if len(raw) >= 5 and raw[-5] in "+-" and raw[-3] != ":":
        raw = raw[:-2] + ":" + raw[-2:]
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _epoch(dt: datetime | None) -> float | None:
    return dt.timestamp() if dt else None


def normalize_event(ev: dict, station: Station, collect_day: date) -> dict | None:
    """Transforme un événement brut 'air' en enregistrement normalisé.

    Retourne None si le pic (`max_ts`) ou la cible (`max_laeq`) est absent
    (événement inexploitable pour la jointure / l'apprentissage).
    """
    max_dt = _parse_iso(ev.get("max_ts"))
    start_dt = _parse_iso(ev.get("start"))
    end_dt = _parse_iso(ev.get("end"))
    if max_dt is None or ev.get("max_laeq") is None or ev.get("id") is None:
        return None
    duration = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None
    return {
        "station": station.measurement_id,
        "airport": station.airport,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "event_id": int(ev["id"]),
        "category": ev.get("category"),
        "start_ts_unix": _epoch(start_dt),
        "end_ts_unix": _epoch(end_dt),
        "max_ts_unix": _epoch(max_dt),
        "max_ts_iso": ev.get("max_ts"),
        "duration_s": duration,
        "laeq": ev.get("laeq"),
        "max_laeq": float(ev["max_laeq"]),
        "sel": ev.get("sel"),
        "nrj_laeq": ev.get("nrj_laeq"),
        "valid": bool(ev.get("valid", True)),
        "collect_day": collect_day.isoformat(),
    }


def _day_windows(
    day: date, window_minutes: int, now: datetime
) -> list[tuple[datetime, datetime]]:
    """Découpe un jour J en fenêtres [début, fin) (UTC), bornées à `now`.

    Les couloirs CDG/Orly dépassent 50 survols/jour → le plafond de ~50/appel
    tronque une requête journalière. Des fenêtres infra-journalières (60 min ≈
    27 évts max au pic, vérifié) restent sous le plafond et évitent la troncature.
    """
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    day_end = start + timedelta(days=1)
    end_cap = min(day_end, now)  # ne pas interroger le futur
    windows: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end_cap:
        nxt = min(cur + timedelta(minutes=window_minutes), day_end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def _partition_dir(settings: Settings, station: str, day: date) -> Path:
    return settings.noise_events_dir / f"station={station}" / f"date={day.isoformat()}"


def write_events(records: list[dict], settings: Settings, station: str, day: date) -> Path | None:
    """Écrit (idempotent) les événements normalisés d'une (station, jour)."""
    if not records:
        return None
    part = _partition_dir(settings, station, day)
    part.mkdir(parents=True, exist_ok=True)
    out = part / f"events_{day.isoformat()}.parquet"
    table = pa.Table.from_pylist(records, schema=_NOISE_SCHEMA)
    pq.write_table(table, out, compression="snappy")
    return out


@dataclass
class StationDayResult:
    station: str
    day: str
    air_events: int
    written: int
    ok: bool
    error: str | None = None


def collect(
    settings: Settings | None = None,
    days_back: int = 14,
    end_day: date | None = None,
    stations: tuple[Station, ...] = STATIONS,
    client: BruitparifClient | None = None,
    collected_at: datetime | None = None,
    request_pause_s: float = 0.5,
    window_minutes: int = 60,
) -> dict:
    """Collecte les survols des `stations` sur les `days_back` derniers jours.

    `end_day` = dernier jour inclus (défaut : aujourd'hui UTC). Chaque jour est
    découpé en **fenêtres infra-journalières** (`window_minutes`, 60 par défaut)
    pour rester sous le plafond de ~50 événements/appel (anti-troncature). Écrit
    un Parquet par (station, jour) + JSON brut (provenance) et journalise une
    métrique par (station, jour). `request_pause_s` espace les appels.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    collected_at = collected_at or datetime.now(timezone.utc)
    end_day = end_day or collected_at.date()
    days = [end_day - timedelta(days=i) for i in range(days_back)][::-1]

    own_client = client is None
    client = client or BruitparifClient(settings)
    results: list[StationDayResult] = []
    raw_by_station: dict[str, list[dict]] = {st.measurement_id: [] for st in stations}

    try:
        for st in stations:
            for day in days:
                day_air: list[dict] = []
                ok = True
                error: str | None = None
                for start, end in _day_windows(day, window_minutes, collected_at):
                    if request_pause_s:
                        time.sleep(request_pause_s)
                    try:
                        air = client.fetch_window_air_events(st.measurement_id, start, end)
                        day_air.extend(air)
                    except Exception as exc:  # noqa: BLE001 — fenêtre en échec : on continue
                        ok = False
                        error = str(exc)
                        logger.exception(
                            "Échec fenêtre %s %s..%s", st.measurement_id, start, end
                        )
                raw_by_station[st.measurement_id].extend(day_air)
                # déduplication par id d'événement (fenêtres jointives + relances)
                seen: set[int] = set()
                uniq = [
                    r
                    for e in day_air
                    if (r := normalize_event(e, st, day)) is not None
                    and not (r["event_id"] in seen or seen.add(r["event_id"]))
                ]
                write_events(uniq, settings, st.measurement_id, day)
                results.append(
                    StationDayResult(st.measurement_id, day.isoformat(), len(day_air), len(uniq), ok, error)
                )
                logger.info(
                    "bruit %s %s : %d air (fenêtres) -> %d écrits%s",
                    st.measurement_id, day, len(day_air), len(uniq), "" if ok else " [fenêtre(s) KO]"
                )
    finally:
        if own_client:
            client.close()

    # Sauvegarde brute par station (provenance, gitignoré) : {station}_{from}_{to}.json
    if days:
        span = f"{days[0].isoformat()}_{days[-1].isoformat()}"
        for station_id, raw in raw_by_station.items():
            if raw:
                (settings.real_survol_dir / f"{station_id}_{span}.json").write_text(
                    json.dumps(raw, ensure_ascii=False), encoding="utf-8"
                )

    _log_metrics(settings, results, collected_at)
    report = {
        "days": [d.isoformat() for d in days],
        "stations": [st.measurement_id for st in stations],
        "station_days_ok": sum(1 for r in results if r.ok),
        "station_days": len(results),
        "air_events_total": sum(r.air_events for r in results),
        "written_total": sum(r.written for r in results),
    }
    return report


def _log_metrics(settings: Settings, results: list[StationDayResult], collected_at: datetime) -> None:
    path = settings.curated_dir / "noise_collect_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "collected_at": collected_at.isoformat(),
                        "station": r.station,
                        "day": r.day,
                        "air_events": r.air_events,
                        "written": r.written,
                        "ok": r.ok,
                        "error": r.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def noise_events_glob(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return str(settings.noise_events_dir / "**" / "*.parquet")


def load_to_duckdb(settings: Settings | None = None) -> int:
    """Matérialise tous les Parquet d'événements dans la table `noise_events`."""
    settings = settings or get_settings()
    glob = noise_events_glob(settings)
    if not any(settings.noise_events_dir.rglob("*.parquet")):
        logger.warning("Aucun Parquet d'événements à charger.")
        return 0
    with connect(settings) as con:
        con.execute(
            f"CREATE OR REPLACE TABLE noise_events AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
        )
        return con.execute("SELECT count(*) FROM noise_events").fetchone()[0]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Collecteur d'événements de survol Bruitparif.")
    parser.add_argument("--days", type=int, default=14, help="Nombre de jours à remonter.")
    parser.add_argument("--end-day", type=str, default=None, help="Dernier jour inclus (YYYY-MM-DD).")
    parser.add_argument("--no-duckdb", action="store_true", help="Ne pas (re)charger DuckDB.")
    args = parser.parse_args(argv)

    end_day = date.fromisoformat(args.end_day) if args.end_day else None
    report = collect(days_back=args.days, end_day=end_day)
    print("Collecte bruit terminée :")
    for k, v in report.items():
        if k not in ("days", "stations"):
            print(f"  {k}: {v}")
    if not args.no_duckdb:
        n = load_to_duckdb()
        print(f"  noise_events (DuckDB): {n} lignes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
