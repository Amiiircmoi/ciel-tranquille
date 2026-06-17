"""Poller micro-batch — cœur de l'ingestion.

Boucle quasi temps réel : à chaque tick (≈12 s), on récupère un snapshot d'états
avions (live OpenSky *ou* replay déterministe), on le normalise, et on l'écrit
en **landing zone Parquet partitionnée par date** (architecture *medallion* :
`raw/`). Chaque batch est **idempotent** (nom de fichier = horodatage du
snapshot) et journalise ses métriques.

Pourquoi Parquet + partition par date : format colonnaire compressé, lisible
nativement par DuckDB, partitionnement temporel = élagage de partitions à la
lecture → **scalabilité horizontale** du stockage (on ajoute des fichiers, on
ne réécrit rien).
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.ingest.opensky_client import OpenSkyClient, OpenSkyError, states_to_records
from ciel_tranquille.ingest.replay import replay_snapshots
from ciel_tranquille.monitoring.metrics import BatchMetric, MetricsLogger

logger = logging.getLogger(__name__)

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


def _partition_dir(raw_dir: Path, snapshot_ts: int) -> Path:
    """`raw/states/date=YYYY-MM-DD/` à partir de l'epoch du snapshot."""
    day = time.strftime("%Y-%m-%d", time.gmtime(snapshot_ts))
    return raw_dir / "states" / f"date={day}"


def write_batch(records: list[dict], raw_dir: Path) -> tuple[Path, int]:
    """Écrit un micro-batch en Parquet. Retourne (chemin, octets écrits).

    Idempotent : si le fichier (clé = snapshot_ts) existe déjà, on le réécrit à
    l'identique plutôt que de dupliquer.
    """
    if not records:
        raise ValueError("Batch vide : rien à écrire.")
    snapshot_ts = int(records[0]["snapshot_ts"])
    part_dir = _partition_dir(raw_dir, snapshot_ts)
    part_dir.mkdir(parents=True, exist_ok=True)
    out_path = part_dir / f"states_{snapshot_ts}.parquet"

    table = pa.Table.from_pylist(records, schema=_PARQUET_SCHEMA)
    pq.write_table(table, out_path, compression="snappy")
    return out_path, out_path.stat().st_size


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
            _, size = write_batch(records, settings.raw_dir)
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


def run_forward(
    settings: Settings | None = None,
    duration_s: float | None = None,
    max_batches: int | None = None,
    sleep=time.sleep,
) -> list[BatchMetric]:
    """Collecte *forward* continue (live) — cœur de la captation co-localisée.

    Boucle jusqu'à l'une des conditions d'arrêt : durée écoulée (`duration_s`),
    nombre de batches (`max_batches`), **plancher de crédits** atteint, ou
    signal (SIGINT/SIGTERM) → arrêt propre. Chaque tick journalise le solde de
    crédits (`x-rate-limit-remaining`). Les erreurs transitoires d'un tick sont
    journalisées sans interrompre la collecte ; une erreur d'auth l'arrête.
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

    client = OpenSkyClient(settings)
    start = time.monotonic()
    logger.info(
        "Poller forward démarré : bbox=%s, cadence=%ds, plancher_crédits=%d, durée=%s",
        settings.bbox.as_params(),
        settings.poll_interval_s,
        settings.credit_floor,
        f"{duration_s}s" if duration_s else "illimitée",
    )
    i = 0
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
                _interruptible_sleep(settings.poll_interval_s, stop, sleep)
                i += 1
                continue
            metric = _record_batch(
                records, settings, "live", f"forward-{i:06d}", credits, metrics_logger
            )
            collected.append(metric)
            if credits is not None and credits <= settings.credit_floor:
                logger.warning(
                    "Plancher crédits atteint (%s <= %s) — arrêt du poller forward.",
                    credits,
                    settings.credit_floor,
                )
                break
            i += 1
            _interruptible_sleep(settings.poll_interval_s, stop, sleep)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.mode:
        settings.ingest_mode = args.mode

    if args.forward:
        metrics = run_forward(settings=settings, duration_s=args.duration)
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
