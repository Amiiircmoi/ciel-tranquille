"""Poller micro-batch — cœur de l'ingestion.

Boucle quasi temps réel : à chaque tick (cadence `CT_POLL_INTERVAL_S`, 30 s), on
récupère un snapshot d'états avions (live OpenSky *ou* replay déterministe), on
le normalise, et on l'écrit en **landing zone Parquet partitionnée par date**
(`$CIEL_DATA_DIR/landing/date=YYYY-MM-DD/`). Chaque batch est **idempotent**
(nom de fichier = horodatage du snapshot) et journalise ses métriques.

Pourquoi Parquet + partition par date : format colonnaire compressé, lisible
nativement par DuckDB, partitionnement temporel = élagage de partitions à la
lecture → **scalabilité horizontale** du stockage (on ajoute des fichiers, on
ne réécrit rien). Les dizaines de milliers de petits fichiers produits sur une
collecte longue sont regroupés a posteriori par `ciel_tranquille.compact`.

Deux garde-fous pour tourner six jours sans supervision :
- **heartbeat** réécrit à chaque snapshot réussi (`status/heartbeat.json`) ;
- **budget crédits** : la cadence nominale est tenue tant que le budget quotidien
  couvre la fin de journée ; sinon elle est *étirée* (jamais raccourcie) pour ne
  pas épuiser l'allocation avant minuit UTC, et le poller s'arrête sous le
  plancher de sécurité.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ciel_tranquille.config import Settings, StationConfigError, get_settings
from ciel_tranquille.ingest.opensky_client import OpenSkyClient, OpenSkyError, states_to_records
from ciel_tranquille.ingest.replay import replay_snapshots
from ciel_tranquille.monitoring.heartbeat import Heartbeat, write_heartbeat
from ciel_tranquille.monitoring.logging_setup import configure_logging
from ciel_tranquille.monitoring.metrics import BatchMetric, MetricsLogger

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400

_PARQUET_SCHEMA = pa.schema(
    [
        ("icao24", pa.string()),
        ("callsign", pa.string()),
        ("origin_country", pa.string()),
        ("time_position_unix", pa.int64()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("baro_altitude_m", pa.float64()),
        ("velocity_m_s", pa.float64()),
        ("heading_deg", pa.float64()),
        ("squawk", pa.string()),
        ("last_contact_unix", pa.int64()),
        ("on_ground", pa.bool_()),
        ("snapshot_ts", pa.int64()),
    ]
)


def _partition_dir(partition_root: Path, snapshot_ts: int) -> Path:
    """`<racine>/date=YYYY-MM-DD/` à partir de l'epoch (UTC) du snapshot."""
    day = time.strftime("%Y-%m-%d", time.gmtime(snapshot_ts))
    return partition_root / f"date={day}"


def write_batch(records: list[dict], partition_root: Path) -> tuple[Path, int]:
    """Écrit un micro-batch en Parquet. Retourne (chemin, octets écrits).

    `partition_root` contient directement les partitions `date=…` (landing zone
    du poller, ou `raw/states` pour le monde synthétique). **Idempotent** : la clé
    du fichier est le `snapshot_ts`, donc rejouer un batch réécrit le même fichier
    au lieu d'en créer un second. L'écriture passe par un fichier temporaire suivi
    d'un `os.replace` : un arrêt brutal ne laisse jamais un Parquet tronqué dans
    la landing (ce qui casserait toute lecture DuckDB ultérieure).
    """
    if not records:
        raise ValueError("Batch vide : rien à écrire.")
    snapshot_ts = int(records[0]["snapshot_ts"])
    part_dir = _partition_dir(partition_root, snapshot_ts)
    part_dir.mkdir(parents=True, exist_ok=True)
    out_path = part_dir / f"states_{snapshot_ts}.parquet"
    tmp_path = part_dir / f".states_{snapshot_ts}.{os.getpid()}.tmp"

    table = pa.Table.from_pylist(records, schema=_PARQUET_SCHEMA)
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path, out_path.stat().st_size


def seconds_until_utc_midnight(now: float | None = None) -> float:
    """Secondes restantes avant la réinitialisation quotidienne du budget OpenSky."""
    now = time.time() if now is None else now
    return SECONDS_PER_DAY - (now % SECONDS_PER_DAY)


def credit_guarded_interval(
    nominal_s: float,
    credits_remaining: int | None,
    credit_floor: int,
    max_interval_s: float,
    now: float | None = None,
) -> float:
    """Cadence effective d'un tick, sous garde-fou de budget quotidien.

    Les crédits OpenSky sont une allocation **quotidienne** (réinitialisée à
    minuit UTC). Tant que le solde utilisable couvre la fin de journée à la
    cadence nominale, on garde **exactement** la cadence nominale (30 s) : le
    garde-fou n'accélère jamais et ne modifie rien en fonctionnement normal.

    S'il ne la couvre plus (journée entamée avec un solde amputé, reprise après
    incident, quota partagé), on **étire** l'intervalle pour répartir le solde
    restant jusqu'à minuit, borné par `max_interval_s` : mieux vaut collecter
    moins dense jusqu'au bout que s'arrêter à 18 h.
    """
    if credits_remaining is None:
        return nominal_s
    usable = credits_remaining - credit_floor
    if usable <= 0:
        return max_interval_s
    seconds_left = seconds_until_utc_midnight(now)
    calls_at_nominal = seconds_left / nominal_s
    if usable >= calls_at_nominal:
        return nominal_s
    return min(max(seconds_left / usable, nominal_s), max_interval_s)


def _fetch_live_batch(client: OpenSkyClient) -> list[dict]:
    payload = client.fetch_states()
    return states_to_records(payload)


def _record_batch(
    records: list[dict],
    settings: Settings,
    mode: str,
    batch_id: str,
    credits_remaining: int | None,
    metrics_logger: MetricsLogger,
) -> BatchMetric:
    """Écrit un micro-batch + journalise sa métrique. Un snapshot vide (aucun
    aéronef dans la bbox) est valide : rows=0, ok=True, aucun fichier écrit."""
    t0 = time.perf_counter()
    error: str | None = None
    rows = 0
    size = 0
    snapshot_ts = int(records[0]["snapshot_ts"]) if records else 0
    ok = True
    try:
        if records:
            _, size = write_batch(records, settings.landing_dir)
            rows = len(records)
    except Exception as exc:  # noqa: BLE001 — on journalise et on continue
        ok = False
        error = str(exc)
        logger.exception("Batch %s en échec", batch_id)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    metric = BatchMetric(
        batch_id=batch_id,
        mode=mode,
        snapshot_ts=snapshot_ts,
        rows=rows,
        duration_ms=round(duration_ms, 2),
        bytes_written=size,
        ok=ok,
        error=error,
        credits_remaining=credits_remaining,
    )
    metrics_logger.log(metric)
    logger.info(
        "batch %s ok=%s rows=%d %.0f ms %.1f rows/s crédits=%s",
        batch_id,
        ok,
        rows,
        duration_ms,
        metric.throughput_rows_per_s,
        credits_remaining,
    )
    return metric


def _beat(
    settings: Settings,
    metric: BatchMetric,
    mode: str,
    batches_total: int,
    interval_s: float,
) -> None:
    """Preuve de vie après un snapshot réussi (jamais bloquante)."""
    if not metric.ok:
        return
    write_heartbeat(
        settings.heartbeat_path,
        Heartbeat(
            updated_at_unix=time.time(),
            snapshot_ts=metric.snapshot_ts,
            rows=metric.rows,
            batches_total=batches_total,
            credits_remaining=metric.credits_remaining,
            poll_interval_s=round(interval_s, 1),
            mode=mode,
            pid=os.getpid(),
        ),
    )


def run(
    n_batches: int,
    settings: Settings | None = None,
    sleep_between: bool = True,
) -> list[BatchMetric]:
    """Exécute `n_batches` micro-batches selon le mode configuré.

    `sleep_between=False` accélère les tests (pas d'attente réelle). En mode
    live, journalise le solde de crédits et s'arrête si le plancher est atteint.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    metrics_logger = MetricsLogger(settings.curated_dir / "pipeline_metrics.jsonl")
    collected: list[BatchMetric] = []

    mode = settings.ingest_mode.lower()
    logger.info("Poller démarré : mode=%s, n_batches=%d", mode, n_batches)

    client = OpenSkyClient(settings) if mode == "live" else None
    replay_gen = None if mode == "live" else replay_snapshots(n_batches, settings.poll_interval_s)
    try:
        for i in range(n_batches):
            if client is not None:
                records = _fetch_live_batch(client)
                credits = client.last_rate_limit_remaining
            else:
                records = next(replay_gen, [])
                credits = None
            metric = _record_batch(
                records, settings, mode, f"{mode}-{i:04d}", credits, metrics_logger
            )
            collected.append(metric)
            _beat(settings, metric, mode, len(collected), settings.poll_interval_s)
            if credits is not None and credits <= settings.credit_floor:
                logger.warning(
                    "Plancher crédits atteint (%s <= %s) — arrêt du poller.",
                    credits,
                    settings.credit_floor,
                )
                break
            if sleep_between and i < n_batches - 1:
                time.sleep(settings.poll_interval_s)
    finally:
        if client is not None:
            client.close()

    return collected


def preflight_stations(settings: Settings) -> None:
    """Contrôle de démarrage des stations actives — échec = refus de démarrer.

    Le poller collecte des positions d'aéronefs, mais ces positions n'ont de
    valeur que rapportées à une station de mesure. Deux défauts silencieux à
    écarter avant six jours de collecte :

    - une station **hors bbox** : on ne verra jamais l'aéronef qui la survole ;
    - une station **muette** : elle ne produira jamais d'événement à apparier.

    Dans les deux cas la collecte « fonctionnerait » sans rien produire d'utile.
    On préfère un démarrage qui échoue avec un message explicite : un silence de
    collecte ne doit jamais être une panne muette.
    """
    from ciel_tranquille.config import StationConfigError
    from ciel_tranquille.ingest.stations import (
        StationValidationError,
        validate_active_stations,
    )

    try:
        report = validate_active_stations(settings)
    except (StationConfigError, StationValidationError) as exc:
        logger.error("Contrôle des stations en échec — démarrage du poller refusé.\n%s", exc)
        raise
    logger.info(
        "Stations actives validées : %s",
        ", ".join(f"{r['measurement_id']} ({r['usable_events']} évts)" for r in report),
    )


def run_forward(
    settings: Settings | None = None,
    duration_s: float | None = None,
    max_batches: int | None = None,
    sleep=time.sleep,
    wait_for_reset: bool = True,
    validate_stations: bool = True,
) -> list[BatchMetric]:
    """Collecte *forward* continue (live) — cœur de la captation co-localisée.

    Boucle jusqu'à l'une des conditions d'arrêt : durée écoulée (`duration_s`),
    nombre de batches (`max_batches`) ou signal (SIGINT/SIGTERM) → arrêt propre.
    Chaque tick journalise le solde de crédits (`x-rate-limit-remaining`), écrit
    un heartbeat et réévalue la cadence sous garde-fou de budget.

    Au **plancher de crédits**, `wait_for_reset=True` (défaut) met la collecte en
    pause jusqu'à la réinitialisation quotidienne UTC au lieu de rendre la main :
    un process qui sort serait relancé aussitôt par le superviseur de conteneurs
    et brûlerait un crédit à chaque redémarrage. `wait_for_reset=False` conserve
    l'ancien comportement (sortie), utile en exécution ponctuelle.

    `validate_stations=True` (défaut) refuse de démarrer si une station active
    est hors bbox ou muette (cf. `preflight_stations`).

    Les erreurs transitoires d'un tick sont journalisées sans interrompre la
    collecte ; une erreur d'auth l'arrête.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    if settings.ingest_mode.lower() != "live":
        raise RuntimeError("run_forward exige CT_INGEST_MODE=live (collecte réelle).")
    metrics_logger = MetricsLogger(settings.curated_dir / "pipeline_metrics.jsonl")
    collected: list[BatchMetric] = []

    stop = {"flag": False}

    def _handle(signum, _frame):
        logger.info("Signal %s reçu — arrêt propre du poller forward.", signum)
        stop["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # hors thread principal : on ignore
            pass

    if validate_stations:
        preflight_stations(settings)
    client = OpenSkyClient(settings)
    start = time.monotonic()
    if not settings.has_contact:
        logger.warning(
            "Aucun contact configuré (CIEL_CONTACT / CIEL_USER_AGENT) : le trafic sortant "
            "n'est pas identifiable. Renseignez-le avant toute collecte longue."
        )
    logger.info(
        "Poller forward démarré : bbox=%s (%.2f deg²), cadence=%ds, plancher_crédits=%d, "
        "durée=%s, landing=%s",
        settings.bbox.as_params(),
        settings.bbox.area_deg2,
        settings.poll_interval_s,
        settings.credit_floor,
        f"{duration_s}s" if duration_s else "illimitée",
        settings.landing_dir,
    )
    i = 0
    interval = float(settings.poll_interval_s)
    try:
        while not stop["flag"]:
            if max_batches is not None and i >= max_batches:
                break
            if duration_s is not None and (time.monotonic() - start) >= duration_s:
                break
            try:
                records = _fetch_live_batch(client)
                credits = client.last_rate_limit_remaining
            except OpenSkyError:
                logger.exception("Erreur OpenSky non récupérable — arrêt du poller.")
                break
            except Exception:  # noqa: BLE001 — tick en échec : on journalise et on continue
                logger.exception("Tick %d en échec (transitoire) — on poursuit.", i)
                _interruptible_sleep(interval, stop, sleep)
                i += 1
                continue
            metric = _record_batch(
                records, settings, "live", f"forward-{i:06d}", credits, metrics_logger
            )
            collected.append(metric)
            # Garde-fou budget : la cadence nominale est tenue tant que le solde
            # quotidien couvre la fin de journée, puis étirée plutôt qu'épuisée.
            previous = interval
            interval = credit_guarded_interval(
                float(settings.poll_interval_s),
                credits,
                settings.credit_floor,
                float(settings.max_poll_interval_s),
            )
            if abs(interval - previous) > 0.5:
                logger.warning(
                    "Cadence ajustée %.0fs -> %.0fs (crédits restants=%s, plancher=%d).",
                    previous,
                    interval,
                    credits,
                    settings.credit_floor,
                )
            _beat(settings, metric, "live", len(collected), interval)
            if credits is not None and credits <= settings.credit_floor:
                if not wait_for_reset:
                    logger.warning(
                        "Plancher crédits atteint (%s <= %s) — arrêt du poller forward.",
                        credits,
                        settings.credit_floor,
                    )
                    break
                # Deux raisons de ne pas sortir du process : le superviseur de
                # conteneurs le relancerait aussitôt, brûlant un crédit à chaque
                # redémarrage pour reconstater le plancher ; et le solde OpenSky
                # se **réapprovisionne** en cours de journée (observé en direct :
                # remontée de +24 après 22 appels). Sortir, ou dormir jusqu'à
                # minuit, ferait perdre des heures de collecte alors que le quota
                # est déjà revenu. On patiente donc par paliers, et le prochain
                # appel relit le solde réel.
                wait_s = min(
                    float(settings.credit_recheck_s), seconds_until_utc_midnight() + 60.0
                )
                logger.warning(
                    "Plancher crédits atteint (%s <= %s) — pause de %.0f min avant "
                    "nouvelle lecture du solde.",
                    credits,
                    settings.credit_floor,
                    wait_s / 60.0,
                )
                _interruptible_sleep(wait_s, stop, sleep)
                interval = float(settings.poll_interval_s)
                i += 1
                continue
            i += 1
            _interruptible_sleep(interval, stop, sleep)
    finally:
        client.close()
        logger.info("Poller forward terminé : %d batches, dernier solde crédits=%s",
                    len(collected),
                    collected[-1].credits_remaining if collected else None)
    return collected


def _interruptible_sleep(seconds: float, stop: dict, sleep=time.sleep) -> None:
    """Dort `seconds` en tranches de 1 s, réactif au drapeau d'arrêt."""
    waited = 0.0
    while waited < seconds and not stop["flag"]:
        step = min(1.0, seconds - waited)
        sleep(step)
        waited += step


def main(argv: list[str] | None = None) -> int:
    from ciel_tranquille.ingest.stations import StationValidationError

    configure_logging("poller")
    parser = argparse.ArgumentParser(description="Poller micro-batch Ciel Tranquille.")
    parser.add_argument("--batches", type=int, default=5, help="Nombre de micro-batches.")
    parser.add_argument("--no-sleep", action="store_true", help="Pas d'attente entre batches.")
    parser.add_argument(
        "--mode",
        choices=["live", "replay"],
        default=None,
        help="Force le mode (sinon CT_INGEST_MODE).",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Collecte forward continue (live) jusqu'à --duration / plancher crédits / signal.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Durée max (s) du mode --forward (défaut : illimité).",
    )
    parser.add_argument(
        "--stop-on-floor",
        action="store_true",
        help="Sortir au plancher de crédits au lieu d'attendre la réinitialisation quotidienne.",
    )
    parser.add_argument(
        "--skip-station-check",
        action="store_true",
        help="Ne pas valider les stations actives au démarrage (diagnostic hors ligne).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.mode:
        settings.ingest_mode = args.mode

    if args.forward:
        try:
            metrics = run_forward(
                settings=settings,
                duration_s=args.duration,
                wait_for_reset=not args.stop_on_floor,
                validate_stations=not args.skip_station_check,
            )
        except (StationConfigError, StationValidationError):
            # Message déjà journalisé par `preflight_stations` : on sort avec un
            # code distinct pour que le superviseur ne confonde pas configuration
            # invalide et panne transitoire de collecte.
            return 2
    else:
        metrics = run(args.batches, settings=settings, sleep_between=not args.no_sleep)
    ok = sum(1 for m in metrics if m.ok)
    total_rows = sum(m.rows for m in metrics)
    last_credits = metrics[-1].credits_remaining if metrics else None
    print(
        f"Terminé : {ok}/{len(metrics)} batches OK, {total_rows} lignes ingérées, "
        f"crédits restants={last_credits}."
    )
    return 0 if ok == len(metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
