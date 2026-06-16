"""Orchestration d'un *cycle* de pipeline micro-batch.

Un cycle = N micro-batches d'ingestion -> reconstruction de la couche curated.
C'est l'unité ordonnançable : on la déclenche périodiquement via un
**ordonnanceur léger adapté à un déploiement solo/VPS** (cron ou systemd timer ;
cf. `deploy/`). Airflow/Prefect seraient surdimensionnés ici — ils sont cités
comme axe de montée en charge.

Le monitoring (débit, latence, volumes) est journalisé par batch
(`monitoring.metrics`) et résumé ici pour le dashboard.
"""

from __future__ import annotations

import argparse
import logging
import time

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.ingest import poller
from ciel_tranquille.monitoring.metrics import MetricsLogger
from ciel_tranquille.storage import build_curated

logger = logging.getLogger(__name__)


def run_cycle(n_batches: int, settings: Settings | None = None, sleep_between: bool = True) -> dict:
    """Exécute un cycle complet : ingestion -> curated. Retourne un résumé."""
    settings = settings or get_settings()
    t0 = time.perf_counter()
    metrics = poller.run(n_batches, settings=settings, sleep_between=sleep_between)
    report = build_curated.build(settings=settings)
    elapsed = time.perf_counter() - t0
    summary = {
        "batches": len(metrics),
        "batches_ok": sum(1 for m in metrics if m.ok),
        "rows_ingested": sum(m.rows for m in metrics),
        "cycle_seconds": round(elapsed, 2),
        "tables": report["tables"],
    }
    logger.info("Cycle terminé en %.1fs : %s", elapsed, summary)
    return summary


def monitoring_summary(settings: Settings | None = None) -> dict:
    """Indicateurs agrégés du pipeline pour le dashboard."""
    settings = settings or get_settings()
    rows = MetricsLogger(settings.curated_dir / "pipeline_metrics.jsonl").read_all()
    if not rows:
        return {"batches": 0}
    durations = [r["duration_ms"] for r in rows]
    throughputs = [r["throughput_rows_per_s"] for r in rows]
    return {
        "batches": len(rows),
        "batches_ok": sum(1 for r in rows if r["ok"]),
        "rows_total": sum(r["rows"] for r in rows),
        "latency_ms_p50": round(sorted(durations)[len(durations) // 2], 1),
        "latency_ms_max": round(max(durations), 1),
        "throughput_rows_per_s_avg": round(sum(throughputs) / len(throughputs), 1),
        "bytes_total": sum(r["bytes_written"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Exécute un cycle de pipeline micro-batch.")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args(argv)
    summary = run_cycle(args.batches, sleep_between=not args.no_sleep)
    print("Résumé du cycle :")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
