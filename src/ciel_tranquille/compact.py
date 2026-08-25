"""Compaction horaire de la landing zone — `ciel_tranquille.compact`.

**Le problème.** Un snapshot toutes les 30 s produit 2 880 fichiers Parquet par
jour, soit **plus de 17 000 fichiers** sur six jours de collecte. Chacun pèse
quelques dizaines de kilo-octets : c'est le pire cas pour un moteur analytique
(un `read_parquet` doit ouvrir, lire les métadonnées et refermer chaque fichier)
comme pour le système de fichiers (un inode par snapshot, `ls` inutilisable).

**La solution.** Regrouper les snapshots d'une même **heure UTC** en un seul
Parquet, écrit dans `curated/states_hourly/date=YYYY-MM-DD/`. Le partitionnement
par date est conservé, donc l'élagage de partitions continue de fonctionner ;
seule la granularité des fichiers change.

Trois propriétés rendent l'opération sûre sur une collecte non supervisée :

1. **On ne touche jamais à l'heure en cours** (`CIEL_COMPACT_LAG_H`, 2 h par
   défaut) : le poller y écrit encore.
2. **Écriture puis vérification puis suppression** : les fichiers sources ne sont
   effacés qu'après relecture du fichier compacté et contrôle du nombre de
   lignes. En cas de doute, on garde les sources.
3. **Idempotence** : relancer la compaction sur une heure déjà compactée fusionne
   les deux jeux et déduplique sur `(icao24, snapshot_ts)`. Un passage
   interrompu peut donc être rejoué sans créer de doublon ni perdre de ligne.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.monitoring.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# `states_<epoch>.parquet` — l'horodatage du snapshot est la clé d'idempotence.
SNAPSHOT_RE = re.compile(r"^states_(\d+)\.parquet$")
SECONDS_PER_HOUR = 3600


def hour_key(snapshot_ts: int) -> tuple[str, str]:
    """(date, heure) UTC d'un snapshot, tel qu'utilisé dans les chemins."""
    tm = time.gmtime(snapshot_ts)
    return time.strftime("%Y-%m-%d", tm), time.strftime("%H", tm)


def scan_landing(landing_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Regroupe les snapshots de la landing par (date, heure UTC).

    L'heure est déduite du **nom de fichier** (epoch du snapshot), pas de la date
    de modification : rejouer ou recopier un fichier ne change donc pas sa place.
    """
    groups: dict[tuple[str, str], list[Path]] = {}
    if not landing_dir.exists():
        return groups
    for path in landing_dir.rglob("states_*.parquet"):
        match = SNAPSHOT_RE.match(path.name)
        if not match:
            continue
        groups.setdefault(hour_key(int(match.group(1))), []).append(path)
    for files in groups.values():
        files.sort()
    return groups


def is_hour_closed(date_str: str, hour_str: str, lag_h: int, now: float | None = None) -> bool:
    """Vrai si l'heure est terminée depuis plus de `lag_h` heures.

    On compare la **fin** de l'heure à `now`. Avec lag_h=2 à 14 h 05, l'heure 11
    (finie à 12 h 00) est éligible, l'heure 12 (finie à 13 h) ne l'est pas encore.
    """
    now = time.time() if now is None else now
    # `calendar.timegm` interprète le tuple en UTC (contrairement à `time.mktime`,
    # qui appliquerait le fuseau — et le passage à l'heure d'été — de la machine).
    start = calendar.timegm(time.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H"))
    return (now - (start + SECONDS_PER_HOUR)) >= lag_h * SECONDS_PER_HOUR


def _read_tables(paths: list[Path]) -> list[pa.Table]:
    tables: list[pa.Table] = []
    for path in paths:
        try:
            tables.append(pq.read_table(path))
        except (OSError, pa.ArrowInvalid):
            # Fichier tronqué (arrêt brutal d'une ancienne version) : on le
            # signale et on continue, plutôt que d'échouer sur toute l'heure.
            logger.error("Parquet illisible, ignoré : %s", path)
    return tables


def _dedup(table: pa.Table) -> pa.Table:
    """Déduplique sur (icao24, snapshot_ts) — clé naturelle d'un état d'aéronef."""
    if table.num_rows == 0 or "snapshot_ts" not in table.column_names:
        return table
    df = table.to_pandas()
    keys = [c for c in ("icao24", "snapshot_ts") if c in df.columns]
    df = df.drop_duplicates(subset=keys).sort_values("snapshot_ts", kind="stable")
    return pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)


@dataclass
class CompactionReport:
    """Résultat d'un passage de compaction (journalisable, sans donnée brute)."""

    hours_compacted: int = 0
    hours_skipped_open: int = 0
    files_merged: int = 0
    files_removed: int = 0
    rows: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        ratio = (self.bytes_before / self.bytes_after) if self.bytes_after else 0.0
        return {
            "hours_compacted": self.hours_compacted,
            "hours_skipped_open": self.hours_skipped_open,
            "files_merged": self.files_merged,
            "files_removed": self.files_removed,
            "rows": self.rows,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "compaction_ratio": round(ratio, 2),
            "errors": self.errors,
        }


def compact_hour(
    date_str: str,
    hour_str: str,
    sources: list[Path],
    settings: Settings,
    delete_sources: bool = True,
) -> tuple[int, int, int]:
    """Compacte une heure. Retourne (lignes, octets avant, octets après).

    Fusionne avec un éventuel fichier déjà compacté pour la même heure, écrit via
    un temporaire + `os.replace` (atomique), relit le résultat pour vérifier le
    compte de lignes, et ne supprime les sources qu'ensuite.
    """
    out_dir = settings.compacted_states_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"states_{date_str.replace('-', '')}T{hour_str}.parquet"

    bytes_before = sum(p.stat().st_size for p in sources if p.exists())
    tables = _read_tables(sources)
    if out_path.exists():
        # Reprise : l'heure a déjà été (partiellement) compactée.
        tables.extend(_read_tables([out_path]))
        bytes_before += out_path.stat().st_size
    if not tables:
        return 0, bytes_before, 0

    merged = _dedup(pa.concat_tables(tables, promote_options="default"))
    tmp_path = out_dir / f".{out_path.name}.{os.getpid()}.tmp"
    try:
        pq.write_table(merged, tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Vérification avant toute suppression : on relit ce qui vient d'être écrit.
    written = pq.read_metadata(out_path).num_rows
    if written != merged.num_rows:
        raise RuntimeError(
            f"Compaction {date_str} {hour_str}h : {written} lignes relues pour "
            f"{merged.num_rows} attendues — sources conservées."
        )
    if delete_sources:
        for path in sources:
            path.unlink(missing_ok=True)
        _prune_empty_dirs(sources)
    return merged.num_rows, bytes_before, out_path.stat().st_size


def _prune_empty_dirs(sources: list[Path]) -> None:
    """Retire les partitions `date=…` devenues vides après compaction."""
    for parent in {p.parent for p in sources}:
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            logger.debug("Partition non supprimée (non vide ou verrouillée) : %s", parent)


def compact(
    settings: Settings | None = None,
    lag_h: int | None = None,
    delete_sources: bool = True,
    now: float | None = None,
    include_open_hours: bool = False,
) -> CompactionReport:
    """Compacte les heures closes de la landing zone.

    `include_open_hours=True` compacte **aussi** l'heure en cours. À n'utiliser
    que le poller arrêté (fin de collecte) : tant qu'il tourne, il écrit encore
    dans l'heure courante, et compacter sous ses pieds ferait réapparaître ses
    snapshots suivants comme fichiers unitaires — sans perte, mais sans intérêt.
    """
    settings = settings or get_settings()
    lag_h = settings.compact_lag_h if lag_h is None else lag_h
    report = CompactionReport()

    groups = scan_landing(settings.landing_dir)
    for (date_str, hour_str), sources in sorted(groups.items()):
        if not include_open_hours and not is_hour_closed(date_str, hour_str, lag_h, now):
            report.hours_skipped_open += 1
            continue
        try:
            rows, before, after = compact_hour(
                date_str, hour_str, sources, settings, delete_sources=delete_sources
            )
        except Exception as exc:  # noqa: BLE001 — une heure en échec n'arrête pas les autres
            logger.exception("Compaction %s %sh en échec", date_str, hour_str)
            report.errors.append(f"{date_str} {hour_str}h: {exc}")
            continue
        if rows == 0:
            continue
        report.hours_compacted += 1
        report.files_merged += len(sources)
        report.files_removed += len(sources) if delete_sources else 0
        report.rows += rows
        report.bytes_before += before
        report.bytes_after += after
        logger.info(
            "compacté %s %sh : %d fichiers -> 1 (%d lignes, %.0f Ko -> %.0f Ko)",
            date_str, hour_str, len(sources), rows, before / 1024, after / 1024,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    configure_logging("compact")
    parser = argparse.ArgumentParser(description="Compaction horaire des snapshots Parquet.")
    parser.add_argument(
        "--lag-hours",
        type=int,
        default=None,
        help="Ne compacter que les heures closes depuis N heures (défaut CIEL_COMPACT_LAG_H).",
    )
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="Conserver les snapshots unitaires après compaction (vérification).",
    )
    parser.add_argument(
        "--include-open-hours",
        action="store_true",
        help="Compacter aussi l'heure en cours — poller ARRÊTÉ uniquement (fin de collecte).",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=None,
        help="Boucler en attendant N secondes entre deux passages (ordonnanceur "
        "conteneurisé ; sans cette option, un seul passage).",
    )
    args = parser.parse_args(argv)

    while True:
        try:
            report = compact(
                lag_h=args.lag_hours,
                delete_sources=not args.keep_sources,
                include_open_hours=args.include_open_hours,
            )
            logger.info("compaction : %s", report.to_dict())
            if args.every is None:
                print("Compaction terminée :")
                for key, value in report.to_dict().items():
                    print(f"  {key}: {value}")
                return 0 if not report.errors else 1
        except Exception:  # noqa: BLE001 — un passage raté ne tue pas l'ordonnanceur
            logger.exception("Passage de compaction en échec.")
            if args.every is None:
                return 1
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
