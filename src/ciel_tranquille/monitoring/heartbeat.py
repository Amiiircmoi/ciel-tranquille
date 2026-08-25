"""Heartbeat de la collecte — preuve de vie écrite à chaque snapshot réussi.

Une collecte de six jours tourne **sans supervision humaine** : il faut pouvoir
répondre à « est-ce que ça collecte encore ? » sans lire des logs ni ouvrir un
Parquet. Le heartbeat est un petit JSON réécrit à chaque snapshot réussi, dans
`$CIEL_DATA_DIR/status/heartbeat.json`.

Deux propriétés tiennent l'usage :

- **Écriture atomique** (fichier temporaire + `os.replace`) : le fichier n'est
  jamais lu à moitié écrit, même si le conteneur est tué pendant l'écriture.
- **Aucun secret, aucune donnée brute** : uniquement des compteurs et des
  horodatages. Ni jeton, ni identifiant d'aéronef, ni position.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Heartbeat:
    """État de la dernière itération réussie du poller."""

    updated_at_unix: float
    snapshot_ts: int
    rows: int
    batches_total: int
    credits_remaining: int | None
    poll_interval_s: float
    mode: str
    pid: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["updated_at_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.updated_at_unix)
        )
        return d


def write_json_atomic(path: Path, payload: dict) -> None:
    """Écrit `payload` en JSON de façon atomique (tmp + `os.replace`).

    `os.replace` est atomique sur le même système de fichiers : un lecteur voit
    soit l'ancien fichier, soit le nouveau, jamais un fichier tronqué.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_heartbeat(path: Path, beat: Heartbeat) -> None:
    """Persiste le heartbeat ; un échec d'écriture ne doit jamais tuer la collecte."""
    try:
        write_json_atomic(path, beat.to_dict())
    except OSError:
        logger.exception("Écriture du heartbeat impossible (%s) — collecte poursuivie.", path)


def read_heartbeat(path: Path) -> dict | None:
    """Relit le heartbeat (None s'il est absent ou illisible)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Heartbeat illisible : %s", path)
        return None
