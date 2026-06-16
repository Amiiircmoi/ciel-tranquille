"""Métriques de performance du pipeline.

Chaque micro-batch journalise débit (lignes/s), latence (ms) et volume dans un
fichier JSONL append-only. Le dashboard lit ce journal pour afficher la santé
du pipeline (monitoring).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BatchMetric:
    """Indicateurs d'un micro-batch d'ingestion."""

    batch_id: str
    mode: str  # "live" | "replay"
    snapshot_ts: int
    rows: int
    duration_ms: float
    bytes_written: int
    ok: bool
    error: str | None = None

    @property
    def throughput_rows_per_s(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return self.rows / (self.duration_ms / 1000.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["throughput_rows_per_s"] = round(self.throughput_rows_per_s, 1)
        return d


class MetricsLogger:
    """Écriture append-only des métriques de batch (JSON Lines)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metric: BatchMetric) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metric.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
