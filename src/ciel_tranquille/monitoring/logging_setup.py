"""Journalisation vers le volume de données.

Les logs du moteur de conteneurs (`docker logs`) vivent hors du volume : ils
disparaissent avec le conteneur et ne sont pas lisibles depuis les données. Pour
une collecte de six jours sans supervision, la trace doit être **à côté des
données**, dans `$CIEL_DATA_DIR/logs/`, consultable avec un simple `tail`.

Rotation obligatoire : six jours de logs à la seconde saturent un disque partagé
avec une production tierce. On borne à quelques mégaoctets par fichier.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from ciel_tranquille.config import Settings, get_settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
MAX_BYTES = 5_000_000
BACKUP_COUNT = 3


def configure_logging(
    name: str,
    settings: Settings | None = None,
    level: int = logging.INFO,
) -> None:
    """Console + fichier tournant `$CIEL_DATA_DIR/logs/<name>.log`.

    Un échec d'ouverture du fichier ne doit jamais empêcher la collecte : on se
    rabat sur la console seule. Perdre la trace est ennuyeux ; perdre la collecte
    parce qu'on n'a pas pu écrire la trace serait absurde.
    """
    settings = settings or get_settings()
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.logs_dir / f"{name}.log",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("Journalisation fichier indisponible (%s) — console seule.", exc)

    # httpx journalise chaque requête en INFO : à 30 s de cadence sur six jours,
    # cela noierait les événements qui comptent.
    logging.getLogger("httpx").setLevel(logging.WARNING)
