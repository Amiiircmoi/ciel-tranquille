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

from ciel_tranquille.config import Settings, Station, StationConfigError, get_settings
from ciel_tranquille.ingest.bruitparif_client import BruitparifClient, BruitparifRateLimited
from ciel_tranquille.monitoring.heartbeat import write_json_atomic
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


def _fetch_window_with_backoff(
    client: BruitparifClient,
    station: str,
    start: datetime,
    end: datetime,
    max_retries: int,
    backoff_base_s: float,
    sleep=time.sleep,
) -> list[dict]:
    """Récupère une fenêtre, avec back-off exponentiel entre tentatives.

    Le client porte déjà un retry réseau (tenacity). Cette couche traite le cas
    au-dessus : une fenêtre qui échoue malgré les retries (coupure prolongée,
    503 en rafale). On patiente **de plus en plus longtemps** (5 s, 10 s, 20 s…)
    plutôt que de repartir aussitôt : sur une IP partagée avec une production
    tierce, insister vite est le meilleur moyen de se faire bloquer. Un 429
    remonte immédiatement, sans nouvelle tentative.
    """
    last_exc: Exception | None = None
    for attempt in range(max(1, max_retries)):
        try:
            return client.fetch_window_air_events(station, start, end)
        except BruitparifRateLimited:
            raise
        except Exception as exc:  # noqa: BLE001 — on retente puis on abandonne la fenêtre
            last_exc = exc
            if attempt + 1 >= max(1, max_retries):
                break
            wait_s = backoff_base_s * (2**attempt)
            logger.warning(
                "Fenêtre %s %s..%s en échec (tentative %d/%d) — back-off %.0fs : %s",
                station, start, end, attempt + 1, max_retries, wait_s, exc,
            )
            sleep(wait_s)
    raise last_exc if last_exc else RuntimeError("Échec de fenêtre sans exception.")


def _collect_window(
    client: BruitparifClient,
    station: str,
    start: datetime,
    end: datetime,
    settings: Settings,
    pause_s: float,
    stats: dict,
    sleep=time.sleep,
) -> list[dict]:
    """Collecte une fenêtre, en la **redécoupant** si la réponse est saturée.

    `/events` plafonne à ~50 réponses **biaisées vers le début de l'intervalle** :
    une fenêtre saturée ne rend donc pas « 50 événements sur 60 minutes » mais
    « les 50 premiers », et la fin de la fenêtre est perdue **en silence**. Dès
    que le compte approche du plafond (`CIEL_EVENTS_PAGE_CAP × ratio`), on coupe
    la fenêtre en deux et on rappelle chaque moitié, jusqu'à
    `CIEL_EVENTS_MIN_WINDOW_MIN`. Chaque redécoupage est compté dans les métriques :
    une hausse durable signale qu'il faut réduire la fenêtre nominale.
    """
    events = _fetch_window_with_backoff(
        client, station, start, end,
        settings.bruitparif_max_retries, settings.bruitparif_backoff_base_s, sleep,
    )
    stats["calls"] = stats.get("calls", 0) + 1

    span_minutes = (end - start).total_seconds() / 60.0
    if (
        len(events) < settings.events_saturation_threshold
        or span_minutes <= settings.events_min_window_minutes
    ):
        return events

    midpoint = start + (end - start) / 2
    stats["resplits"] = stats.get("resplits", 0) + 1
    logger.info(
        "Fenêtre saturée %s %s..%s (%d événements >= seuil %d) — redécoupage en 2 × %.0f min.",
        station, start, end, len(events), settings.events_saturation_threshold, span_minutes / 2,
    )
    merged: list[dict] = []
    for sub_start, sub_end in ((start, midpoint), (midpoint, end)):
        if pause_s:
            sleep(pause_s)
        merged.extend(
            _collect_window(client, station, sub_start, sub_end, settings, pause_s, stats, sleep)
        )
    return merged


def collect(
    settings: Settings | None = None,
    days_back: int = 14,
    end_day: date | None = None,
    stations: tuple[Station, ...] | None = None,
    client: BruitparifClient | None = None,
    collected_at: datetime | None = None,
    request_pause_s: float | None = None,
    window_minutes: int = 60,
) -> dict:
    """Collecte les survols des `stations` sur les `days_back` derniers jours.

    `stations` par défaut = la liste externalisée (`CIEL_STATIONS_FILE` /
    `$CIEL_DATA_DIR/stations.json` / `config/stations.json`), pas une constante :
    on augmente le rendement en ajoutant des stations **sans toucher au code** —
    et sans consommer un seul crédit OpenSky de plus, la bbox étant commune.

    `end_day` = dernier jour inclus (défaut : aujourd'hui UTC). Chaque jour est
    découpé en **fenêtres infra-journalières** (`window_minutes`, 60 par défaut)
    pour rester sous le plafond de ~50 événements/appel (anti-troncature). Écrit
    un Parquet par (station, jour) + JSON brut (provenance) et journalise une
    métrique par (station, jour).

    `request_pause_s` (défaut `CIEL_BRUITPARIF_PAUSE_S`) espace **chaque** appel ;
    une fenêtre en échec est retentée avec back-off exponentiel avant d'être
    abandonnée. Avec 9 stations × 24 fenêtres × 2 jours, l'espacement est ce qui
    sépare une collecte polie d'une rafale de 400 requêtes.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    stations = stations if stations is not None else settings.stations
    if request_pause_s is None:
        request_pause_s = settings.bruitparif_pause_s
    collected_at = collected_at or datetime.now(timezone.utc)
    end_day = end_day or collected_at.date()
    days = [end_day - timedelta(days=i) for i in range(days_back)][::-1]

    own_client = client is None
    client = client or BruitparifClient(settings)
    results: list[StationDayResult] = []
    raw_by_station: dict[str, list[dict]] = {st.measurement_id: [] for st in stations}
    stats: dict[str, int] = {"calls": 0, "resplits": 0}
    rate_limited = False

    try:
        for st in stations:
            if rate_limited:
                break
            for day in days:
                if rate_limited:
                    break
                day_air: list[dict] = []
                ok = True
                error: str | None = None
                for start, end in _day_windows(day, window_minutes, collected_at):
                    if request_pause_s:
                        time.sleep(request_pause_s)
                    try:
                        air = _collect_window(
                            client,
                            st.measurement_id,
                            start,
                            end,
                            settings,
                            request_pause_s,
                            stats,
                        )
                        day_air.extend(air)
                    except BruitparifRateLimited as exc:
                        # Arrêt propre et immédiat : on garde ce qui est déjà collecté.
                        ok = False
                        error = str(exc)
                        rate_limited = True
                        logger.error("429 Bruitparif — arrêt de la collecte. %s", exc)
                        break
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
        "api_calls": stats["calls"],
        "window_resplits": stats["resplits"],
        "rate_limited": rate_limited,
    }
    _write_noise_health(settings, client, report, collected_at)
    return report


def _write_noise_health(
    settings: Settings,
    client: BruitparifClient,
    report: dict,
    collected_at: datetime,
) -> None:
    """Santé de la source bruit, lue par la supervision (`status.json`).

    Contrôle **distinct** du poller avion : si le motif d'extraction du token
    casse (redéploiement de la SPA Bruitparif), la collecte de bruit s'arrête
    alors que le poller continue de tourner normalement. Sans ce fichier, la
    panne resterait invisible jusqu'à l'analyse finale.
    """
    # `getattr` : un client injecté (doublure de test, client alternatif) n'est pas
    # tenu d'exposer la santé du token — l'absence de mesure n'est pas un échec.
    token_health = getattr(client, "token_health", None)
    write_json_atomic(
        settings.noise_health_path,
        {
            "last_run_unix": collected_at.timestamp(),
            "last_run_iso": collected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "token": token_health() if callable(token_health) else {"ok": None},
            "stations": report["stations"],
            "station_days_ok": report["station_days_ok"],
            "station_days": report["station_days"],
            "events_written": report["written_total"],
            "api_calls": report["api_calls"],
            "window_resplits": report["window_resplits"],
            "rate_limited": report["rate_limited"],
        },
    )


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
    parser.add_argument(
        "--every",
        type=float,
        default=None,
        help="Boucler indéfiniment en attendant N secondes entre deux collectes "
        "(service de collecte continue ; sans cette option, un seul passage).",
    )
    args = parser.parse_args(argv)

    end_day = date.fromisoformat(args.end_day) if args.end_day else None
    settings = get_settings()

    # --- Contrôles de démarrage : échec bruyant plutôt que collecte muette ---
    try:
        stations = settings.stations
    except StationConfigError as exc:
        logger.error("Configuration de stations invalide — collecte refusée.\n%s", exc)
        return 2
    try:
        # Le token est scrapé dans le HTML de la SPA : si le motif ne correspond
        # plus (front redéployé), on s'arrête ici, fort et tout de suite, au lieu
        # de collecter zéro événement pendant six jours.
        probe_client = BruitparifClient(settings)
        try:
            probe_client.token()
        finally:
            health = probe_client.token_health()
            write_json_atomic(
                settings.noise_health_path,
                {"last_run_unix": time.time(), "token": health, "startup_check": True},
            )
            probe_client.close()
    except Exception as exc:  # noqa: BLE001 — démarrage impossible
        logger.error(
            "Token Bruitparif indisponible — collecte de bruit refusée au démarrage.\n%s", exc
        )
        return 3

    logger.info(
        "Collecte bruit : %d station(s), %d jour(s), pause %.1fs entre appels.",
        len(stations), args.days, settings.polite_pause_s,
    )
    while True:
        try:
            report = collect(settings=settings, days_back=args.days, end_day=end_day)
            print("Collecte bruit terminée :")
            for k, v in report.items():
                if k not in ("days", "stations"):
                    print(f"  {k}: {v}")
            if not args.no_duckdb:
                n = load_to_duckdb(settings)
                print(f"  noise_events (DuckDB): {n} lignes")
        except Exception:  # noqa: BLE001 — un passage raté ne doit pas tuer le service
            logger.exception("Passage de collecte bruit en échec.")
            if args.every is None:
                return 1
        if args.every is None:
            return 0
        logger.info("Prochaine collecte bruit dans %.0f s.", args.every)
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
