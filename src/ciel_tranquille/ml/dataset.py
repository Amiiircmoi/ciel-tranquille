"""Gel d'un instantané **immuable** du jeu d'entraînement.

Pourquoi geler plutôt que lire la collecte en direct ? Parce qu'un chiffre de
performance n'a de sens que s'il est reproductible. Tant que le modèle lit le
répertoire de collecte, deux exécutions à dix minutes d'écart ne portent pas sur
les mêmes données : le MAE bouge sans qu'on sache si c'est le modèle ou le jeu
qui a changé, et aucune comparaison entre modèles n'est défendable. Pire, une
métrique publiée devient invérifiable dès le lendemain.

L'instantané est donc écrit **hors du répertoire de collecte**, accompagné d'un
manifeste qui porte la fenêtre couverte, le nombre de paires, la liste des
stations, le taux d'association et la **somme de contrôle SHA-256** du fichier.
Ces cinq éléments suffisent à rejouer une évaluation et à prouver qu'elle porte
bien sur les mêmes lignes.

Règle de gel : seules les heures **closes** entrent dans l'instantané. Une heure
est close quand le bruit de cette heure a eu le temps d'être publié par
Bruitparif *puis* collecté (`CIEL_PAIRS_LAG_H`, 6 h par défaut). Prendre l'heure
en cours ferait entrer des événements dont les aéronefs ne sont pas encore
associés, et sous-estimerait le taux d'association.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

from ciel_tranquille.compact import SECONDS_PER_HOUR
from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.status import _closed_hours, _load_states_hour, _noise_events_frame
from ciel_tranquille.transform.colocate import build_pairs

logger = logging.getLogger(__name__)

# Champs de l'événement de bruit conservés à côté de la géométrie de vol.
EVENT_COLS = ("event_id", "station", "airport", "category", "max_laeq", "laeq",
              "sel", "duration_s", "start_ts_unix", "end_ts_unix", "max_ts_unix")

CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """Somme de contrôle du fichier, lue par blocs (l'instantané peut grossir)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs_for_hour(settings: Settings, date_str: str, hour_str: str) -> pd.DataFrame:
    """Paires propres d'une heure close, enrichies du contexte de l'événement.

    Même règle d'association que le comptage de supervision
    (`status.count_pairs_for_hour`) : l'aéronef le plus proche en distance
    oblique à `max_ts`, retenu si <= `CIEL_PAIRS_MAX_SLANT_KM`. Le dénominateur
    ne retient que les stations actives — cf. la note de `count_pairs_for_hour`.
    """
    start_ts = calendar.timegm(time.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H"))
    end_ts = start_ts + SECONDS_PER_HOUR

    events = _noise_events_frame(settings, date_str)
    if events.empty or "max_ts_unix" not in events.columns:
        return pd.DataFrame()
    events = events[events["max_ts_unix"].between(start_ts, end_ts)]
    stations = {st.measurement_id: st for st in settings.stations}
    events = events[events["station"].isin(stations)]
    if events.empty:
        return pd.DataFrame()

    states = _load_states_hour(settings, start_ts, end_ts)
    if states.empty:
        return pd.DataFrame()

    pairs = build_pairs(events, states, stations)
    pairs = pairs[pairs["slant_km"].notna() & (pairs["slant_km"] <= settings.pairs_max_slant_km)]
    if pairs.empty:
        return pd.DataFrame()

    # On rapatrie le contexte acoustique de l'événement : `build_pairs` ne
    # renvoie que la cible et la géométrie, or SEL et durée servent au contrôle
    # de cohérence (et jamais de feature — cf. `features_real.LEAKAGE_COLS`).
    contexte = events[[c for c in EVENT_COLS if c in events.columns]]
    merged = pairs.merge(contexte, on=["event_id", "station"], how="left", suffixes=("", "_evt"))
    merged["station_lat"] = merged["station"].map(lambda s: stations[s].latitude)
    merged["station_lon"] = merged["station"].map(lambda s: stations[s].longitude)
    merged["hour_utc"] = f"{date_str}T{hour_str}"
    return merged


def build_dataset(settings: Settings | None = None, lag_h: int | None = None) -> pd.DataFrame:
    """Concatène les paires propres de toutes les heures closes disponibles."""
    settings = settings or get_settings()
    lag = settings.pairs_lag_h if lag_h is None else lag_h
    frames = []
    for date_str, hour_str in _closed_hours(settings, time.time(), lag):
        try:
            frame = pairs_for_hour(settings, date_str, hour_str)
        except Exception:  # noqa: BLE001 — une heure en échec ne perd pas les autres
            logger.exception("Heure %s %sh en échec — ignorée.", date_str, hour_str)
            continue
        if not frame.empty:
            frames.append(frame)
        logger.info("%s %sh : %d paires", date_str, hour_str, len(frame))
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    return dataset.sort_values("max_ts_unix").reset_index(drop=True)


def _association_rate(settings: Settings, n_pairs: int) -> tuple[float | None, int]:
    """Taux d'association relu depuis le suivi de supervision, pas recalculé.

    Recalculer le dénominateur ici donnerait un second chiffre, potentiellement
    différent de celui affiché par `status.json`, sans qu'on sache lequel croire.
    """
    path = settings.pairs_progress_path
    if not path.exists():
        return None, 0
    try:
        totals = json.loads(path.read_text(encoding="utf-8")).get("totals", {})
    except (OSError, json.JSONDecodeError):
        return None, 0
    events = int(totals.get("events", 0))
    return (round(n_pairs / events, 4) if events else None), events


def export_snapshot(
    settings: Settings | None = None,
    out_dir: Path | str | None = None,
    label: str | None = None,
    lag_h: int | None = None,
) -> dict:
    """Écrit l'instantané figé + son manifeste. Retourne le manifeste."""
    settings = settings or get_settings()
    dataset = build_dataset(settings, lag_h=lag_h)
    if dataset.empty:
        raise RuntimeError("Aucune paire propre sur les heures closes — rien à geler.")

    out = Path(out_dir) if out_dir else Path.cwd() / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    tag = label or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    data_path = out / f"pairs_{tag}.parquet"
    manifest_path = out / f"pairs_{tag}.manifest.json"

    dataset.to_parquet(data_path, index=False)
    taux, events = _association_rate(settings, len(dataset))
    stations = sorted(dataset["station"].unique())

    manifeste = {
        "exported_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fichier": data_path.name,
        "sha256": sha256_file(data_path),
        "octets": data_path.stat().st_size,
        "n_paires": int(len(dataset)),
        "n_evenements_denominateur": events,
        "taux_association": taux,
        "fenetre": {
            "premier_evenement_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(dataset["max_ts_unix"].min()))
            ),
            "dernier_evenement_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(dataset["max_ts_unix"].max()))
            ),
            "heures_closes": int(dataset["hour_utc"].nunique()),
        },
        "stations": stations,
        "paires_par_station": {k: int(v) for k, v in dataset["station"].value_counts().items()},
        "regle_association": (
            "aéronef le plus proche en distance oblique à max_ts, retenu si <= "
            f"{settings.pairs_max_slant_km} km"
        ),
        "pairs_lag_h": settings.pairs_lag_h if lag_h is None else lag_h,
        "colonnes": list(dataset.columns),
    }
    manifest_path.write_text(
        json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Instantané figé : %s (%d paires)", data_path, len(dataset))
    return manifeste


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Gèle un instantané du jeu d'entraînement.")
    parser.add_argument("--out", default=None, help="répertoire de sortie (hors collecte vive)")
    parser.add_argument("--label", default=None, help="étiquette du gel (défaut : horodatage UTC)")
    parser.add_argument("--lag-hours", type=int, default=None, help="heures de recul avant gel")
    args = parser.parse_args(argv)

    manifeste = export_snapshot(out_dir=args.out, label=args.label, lag_h=args.lag_hours)
    print(json.dumps(manifeste, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
