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
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.ingest.opensky_client import OpenSkyClient, states_to_records
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


def run(
    n_batches: int,
    settings: Settings | None = None,
    sleep_between: bool = True,
) -> list[BatchMetric]:
    """Exécute `n_batches` micro-batches selon le mode configuré.

    `sleep_between=False` accélère les tests (pas d'attente réelle).
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    metrics_logger = MetricsLogger(settings.curated_dir / "pipeline_metrics.jsonl")
    collected: list[BatchMetric] = []

    mode = settings.ingest_mode.lower()
    logger.info("Poller démarré : mode=%s, n_batches=%d", mode, n_batches)

    if mode == "live":
        client = OpenSkyClient(settings)
        batches = (_fetch_live_batch(client) for _ in range(n_batches))
    else:
        client = None
        batches = replay_snapshots(n_batches, settings.poll_interval_s)

    try:
        for i, records in enumerate(batches):
            t0 = time.perf_counter()
            error: str | None = None
            rows = 0
            size = 0
            snapshot_ts = 0
            try:
                snapshot_ts = int(records[0]["snapshot_ts"]) if records else 0
                _, size = write_batch(records, settings.raw_dir)
                rows = len(records)
                ok = True
            except Exception as exc:  # noqa: BLE001 — on journalise et on continue
                ok = False
                error = str(exc)
                logger.exception("Batch %d en échec", i)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            metric = BatchMetric(
                batch_id=f"{mode}-{i:04d}",
                mode=mode,
                snapshot_ts=snapshot_ts,
                rows=rows,
                duration_ms=round(duration_ms, 2),
                bytes_written=size,
                ok=ok,
                error=error,
            )
            metrics_logger.log(metric)
            collected.append(metric)
            logger.info(
                "batch %d ok=%s rows=%d %.0f ms %.1f rows/s",
                i,
                ok,
                rows,
                duration_ms,
                metric.throughput_rows_per_s,
            )
            if sleep_between and i < n_batches - 1:
                time.sleep(settings.poll_interval_s)
    finally:
        if client is not None:
            client.close()

    return collected


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
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.mode:
        settings.ingest_mode = args.mode

    metrics = run(args.batches, settings=settings, sleep_between=not args.no_sleep)
    ok = sum(1 for m in metrics if m.ok)
    total_rows = sum(m.rows for m in metrics)
    print(f"Terminé : {ok}/{len(metrics)} batches OK, {total_rows} lignes ingérées.")
    return 0 if ok == len(metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
