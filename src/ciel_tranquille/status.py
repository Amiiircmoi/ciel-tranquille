"""Supervision de la collecte — `ciel_tranquille.status`.

Écrit `$CIEL_DATA_DIR/status/status.json` : **un seul fichier** à regarder pour
savoir si six jours de collecte non supervisée se passent bien.

    {
      "verdict": "OK",
      "checks": [...],
      "collecte": {"last_snapshot_age_s": 34, "snapshots_last_hour": 118, ...},
      "credits": {"remaining": 3120, ...},
      "bruit": {"events_today": 214, "par_station": {...}},
      "paires": {"clean_pairs_total": 512, ...}
    }

**Aucun secret, aucune donnée brute.** Le fichier ne contient que des compteurs,
des âges et des identifiants publics de station : ni jeton, ni identifiant
d'aéronef, ni position, ni mesure individuelle. Il peut donc être lu, copié ou
collé dans un rapport sans précaution particulière.

Le **cumul de paires propres** est calculé de façon **incrémentale** : chaque
passage ne traite que les heures nouvellement figées (bruit publié *et* collecté,
cf. `CIEL_PAIRS_LAG_H`) et mémorise le résultat dans `status/pairs_progress.json`.
Un passage horaire coûte donc une heure de données, pas six jours — et le
compteur ne dépend pas de la survie du process.

La règle d'appariement réutilise `transform.colocate` **sans la modifier** :
aéronef le plus proche en distance oblique à l'instant du pic, retenu si cette
distance est inférieure à `CIEL_PAIRS_MAX_SLANT_KM`. C'est le critère du
cross-check ; la validation stricte « unique dominant » reste celle de
`scripts/validate_join.py`, qui fait autorité pour le rapport.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import time
from pathlib import Path

import pandas as pd

from ciel_tranquille.compact import SECONDS_PER_HOUR, SNAPSHOT_RE, hour_key
from ciel_tranquille.config import Settings, StationConfigError, get_settings
from ciel_tranquille.monitoring.heartbeat import read_heartbeat, write_json_atomic
from ciel_tranquille.monitoring.logging_setup import configure_logging
from ciel_tranquille.monitoring.notify import notify_from_status
from ciel_tranquille.storage.duck import states_globs, states_roots
from ciel_tranquille.transform.colocate import build_pairs

logger = logging.getLogger(__name__)

VERDICT_OK = "OK"
VERDICT_KO = "KO"


# --------------------------------------------------------------- observations
def _iter_snapshot_ts(settings: Settings, landing_only: bool = False) -> list[int]:
    """Horodatages des snapshots unitaires, d'après le nom de fichier.

    `landing_only=True` restreint à la landing courante (débit de la dernière
    heure : la compaction ne touche jamais aux heures récentes). Par défaut on
    balaie **toutes** les racines d'états — y compris `raw/states/`, où dorment
    les collectes antérieures au découpage `landing/`. Ne regarder que la landing
    ferait silencieusement compter zéro paire sur un historique pourtant présent.
    """
    roots = [settings.landing_dir] if landing_only else states_roots(settings)
    out: list[int] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("states_*.parquet"):
            match = SNAPSHOT_RE.match(path.name)
            if match:
                out.append(int(match.group(1)))
    return out


def _compacted_hours(settings: Settings) -> set[tuple[str, str]]:
    """Heures présentes sous forme compactée (`states_YYYYMMDDTHH.parquet`)."""
    hours: set[tuple[str, str]] = set()
    for path in settings.compacted_states_dir.rglob("states_*T*.parquet"):
        raw = path.stem.split("_", 1)[-1]
        if "T" in raw and len(raw) >= 11:
            day, hour = raw.split("T", 1)
            hours.add((f"{day[:4]}-{day[4:6]}-{day[6:8]}", hour[:2]))
    return hours


def snapshot_hours(settings: Settings) -> set[tuple[str, str]]:
    """Toutes les heures UTC pour lesquelles des états existent, où qu'ils soient."""
    hours = {hour_key(ts) for ts in _iter_snapshot_ts(settings)}
    hours |= _compacted_hours(settings)
    return hours


def collection_state(settings: Settings, now: float) -> dict:
    """Âge du dernier snapshot et débit de la dernière heure.

    Deux sources, dans cet ordre : le **heartbeat** du poller (fiable même quand
    la compaction a déjà emporté les fichiers) puis les noms de fichiers de la
    landing. La compaction ne touchant pas aux heures récentes, la dernière heure
    est toujours comptable depuis la landing.
    """
    beat = read_heartbeat(settings.heartbeat_path) or {}
    landing = _iter_snapshot_ts(settings, landing_only=True)
    snapshots = landing or _iter_snapshot_ts(settings)
    last_ts = max(snapshots, default=None)
    beat_ts = beat.get("snapshot_ts")
    if beat_ts and (last_ts is None or beat_ts > last_ts):
        last_ts = int(beat_ts)

    beat_age = None
    if beat.get("updated_at_unix"):
        beat_age = max(0.0, now - float(beat["updated_at_unix"]))

    return {
        "last_snapshot_ts": last_ts,
        "last_snapshot_age_s": None if last_ts is None else round(max(0.0, now - last_ts), 1),
        "heartbeat_age_s": None if beat_age is None else round(beat_age, 1),
        "snapshots_last_hour": sum(1 for ts in landing if now - ts <= SECONDS_PER_HOUR),
        "snapshots_in_landing": len(landing),
        "poll_interval_s": beat.get("poll_interval_s"),
        "mode": beat.get("mode"),
    }


def _noise_events_frame(settings: Settings, day: str) -> pd.DataFrame:
    """Événements de survol d'un jour (Parquet partitionné station/date)."""
    frames = []
    for part in settings.noise_events_dir.glob(f"station=*/date={day}"):
        for path in part.glob("*.parquet"):
            try:
                frames.append(pd.read_parquet(path))
            except (OSError, ValueError):
                logger.warning("Parquet d'événements illisible : %s", path)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _events_last_24h(settings: Settings, now: float) -> int:
    """Nombre d'événements de bruit sur les 24 dernières heures **glissantes**.

    Compter « aujourd'hui » en jour calendaire crée une falaise à minuit UTC : à
    00:06 le compteur du jour vaut zéro et la collecte paraît morte alors qu'elle
    se porte bien. Observé en production — une alerte urgente à 00:06, rétablie
    à 02:51 quand le collecteur a écrit ses premiers événements du jour. Une
    fenêtre glissante n'a pas de falaise.
    """
    total = 0
    for offset in (0, 1):
        jour = time.strftime("%Y-%m-%d", time.gmtime(now - offset * 86_400))
        frame = _noise_events_frame(settings, jour)
        if not frame.empty and "max_ts_unix" in frame.columns:
            total += int((frame["max_ts_unix"] >= now - 86_400).sum())
    return total


def _last_event_ts(settings: Settings, now: float) -> float | None:
    """Horodatage du dernier événement de bruit collecté (aujourd'hui ou hier)."""
    latest: float | None = None
    for offset in (0, 1):
        day = time.strftime("%Y-%m-%d", time.gmtime(now - offset * 86_400))
        frame = _noise_events_frame(settings, day)
        if not frame.empty and "max_ts_unix" in frame.columns:
            value = float(frame["max_ts_unix"].max())
            latest = value if latest is None else max(latest, value)
    return latest


def is_daytime(settings: Settings, now: float) -> bool:
    """Vrai pendant la plage où du trafic est attendu (heures UTC configurables).

    La nuit, l'absence d'événement de survol est **normale** (couvre-feu Orly,
    trafic CDG résiduel) : la déclarer en panne produirait une alerte fausse
    chaque nuit, et une alerte qu'on apprend à ignorer ne sert plus à rien.
    """
    hour = time.gmtime(now).tm_hour
    return settings.noise_day_start_utc <= hour < settings.noise_day_end_utc


def noise_state(settings: Settings, now: float) -> dict:
    """Santé de la source bruit : volume du jour, fraîcheur, état du token.

    Contrôle **distinct** de celui du poller avion. Les deux sources sont
    indépendantes : le poller peut tourner parfaitement pendant que la collecte
    de bruit est morte (motif d'extraction du token cassé par un redéploiement
    de la SPA Bruitparif). Sans suivi séparé, la panne ne se verrait qu'à
    l'analyse finale — trop tard.
    """
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    events = _noise_events_frame(settings, day)
    per_station: dict[str, int] = {}
    if not events.empty and "station" in events.columns:
        per_station = {str(k): int(v) for k, v in events["station"].value_counts().items()}

    try:
        stations_configured = len(settings.stations)
        config_error = None
    except StationConfigError as exc:
        stations_configured = 0
        config_error = str(exc)

    last_ts = _last_event_ts(settings, now)
    health = read_heartbeat(settings.noise_health_path) or {}
    token = health.get("token") or {}
    last_run = health.get("last_run_unix")
    return {
        "day_utc": day,
        "events_today": int(len(events)),
        "events_last_24h": _events_last_24h(settings, now),
        "stations_configured": stations_configured,
        "stations_reporting_today": len(per_station),
        "par_station": per_station,
        "station_config_error": config_error,
        "last_event_ts": last_ts,
        "last_event_age_s": None if last_ts is None else round(max(0.0, now - last_ts), 1),
        "daytime": is_daytime(settings, now),
        "last_collect_run_iso": health.get("last_run_iso"),
        "last_collect_run_age_s": (
            None if not last_run else round(max(0.0, now - float(last_run)), 1)
        ),
        "silence_budget_s": settings.noise_silence_budget_s,
        "window_resplits_last_run": health.get("window_resplits"),
        "rate_limited_last_run": health.get("rate_limited"),
        "token": {
            "ok": token.get("ok"),
            "checked_at_iso": token.get("checked_at_iso"),
            "error": token.get("error"),
        },
    }


# ------------------------------------------------------------------- paires
def _load_states_hour(settings: Settings, start_ts: int, end_ts: int) -> pd.DataFrame:
    """États d'aéronefs d'une heure, avec la marge d'appariement (±45 s)."""
    frames = []
    margin = 60
    for glob in states_globs(settings):
        for path in Path(glob.split("**")[0]).rglob("*.parquet"):
            match = SNAPSHOT_RE.match(path.name)
            if match:
                ts = int(match.group(1))
                if not (start_ts - margin <= ts <= end_ts + margin):
                    continue
            try:
                frame = pd.read_parquet(path)
            except (OSError, ValueError):
                logger.warning("Parquet d'états illisible : %s", path)
                continue
            if "snapshot_ts" in frame.columns:
                frame = frame[
                    frame["snapshot_ts"].between(start_ts - margin, end_ts + margin)
                ]
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def count_pairs_for_hour(settings: Settings, date_str: str, hour_str: str) -> dict:
    """Compte les paires propres d'une heure figée (événements ↔ aéronefs).

    Le dénominateur ne retient que les événements des stations **actives**. Les
    partitions d'une station retirée de la liste (station en panne, campagne
    terminée) restent sur disque : les compter alors qu'aucun appariement ne sera
    jamais tenté pour elles écraserait le taux d'association. Mesuré sur la
    collecte de juin, l'écart n'est pas anecdotique : 49 % avec les événements
    d'une station retirée au dénominateur, 97 % sans.
    """
    start_ts = calendar.timegm(time.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H"))
    end_ts = start_ts + SECONDS_PER_HOUR
    vide = {"events": 0, "pairs": 0, "events_stations_inactives": 0}

    events = _noise_events_frame(settings, date_str)
    if events.empty or "max_ts_unix" not in events.columns:
        return vide
    events = events[events["max_ts_unix"].between(start_ts, end_ts)]
    if events.empty:
        return vide

    stations = {st.measurement_id: st for st in settings.stations}
    known = events[events["station"].isin(stations)]
    hors_perimetre = int(len(events) - len(known))
    if known.empty:
        return {"events": 0, "pairs": 0, "events_stations_inactives": hors_perimetre}

    states = _load_states_hour(settings, start_ts, end_ts)
    if states.empty:
        return {
            "events": int(len(known)),
            "pairs": 0,
            "events_stations_inactives": hors_perimetre,
        }

    pairs = build_pairs(known, states, stations)
    clean = pairs[
        pairs["slant_km"].notna() & (pairs["slant_km"] <= settings.pairs_max_slant_km)
    ]
    return {
        "events": int(len(known)),
        "pairs": int(len(clean)),
        "events_stations_inactives": hors_perimetre,
    }


def _load_progress(settings: Settings) -> dict:
    if not settings.pairs_progress_path.exists():
        return {"hours": {}}
    try:
        payload = json.loads(settings.pairs_progress_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) and "hours" in payload else {"hours": {}}
    except (OSError, json.JSONDecodeError):
        logger.warning("Avancement des paires illisible — recomptage depuis zéro.")
        return {"hours": {}}


def _closed_hours(settings: Settings, now: float, lag_h: int) -> list[tuple[str, str]]:
    """Heures dont les données bruit et trafic sont considérées complètes."""
    hours = snapshot_hours(settings)
    cutoff = now - lag_h * SECONDS_PER_HOUR
    closed = []
    for date_str, hour_str in hours:
        start = calendar.timegm(time.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H"))
        if start + SECONDS_PER_HOUR <= cutoff:
            closed.append((date_str, hour_str))
    return sorted(closed)


def update_pairs(
    settings: Settings,
    now: float,
    max_hours: int = 12,
    recount: bool = False,
) -> dict:
    """Met à jour le cumul de paires propres (incrémental et idempotent).

    `max_hours` borne le travail d'un passage : au démarrage, un rattrapage sur
    plusieurs jours se fait en quelques passages horaires au lieu d'un seul
    calcul très long qui monopoliserait le CPU limité du conteneur.
    """
    progress = {"hours": {}} if recount else _load_progress(settings)
    pending = [
        h for h in _closed_hours(settings, now, settings.pairs_lag_h)
        if f"{h[0]}T{h[1]}" not in progress["hours"]
    ]
    for date_str, hour_str in pending[:max_hours]:
        try:
            counts = count_pairs_for_hour(settings, date_str, hour_str)
        except Exception as exc:  # noqa: BLE001 — une heure en échec n'arrête pas le reste
            logger.exception("Comptage des paires %s %sh en échec", date_str, hour_str)
            counts = {"events": 0, "pairs": 0, "error": str(exc)}
        progress["hours"][f"{date_str}T{hour_str}"] = counts
        logger.info(
            "paires %s %sh : %d / %d événements",
            date_str, hour_str, counts.get("pairs", 0), counts.get("events", 0),
        )

    hours = progress["hours"]
    total_events = sum(h.get("events", 0) for h in hours.values())
    total_pairs = sum(h.get("pairs", 0) for h in hours.values())
    progress["totals"] = {
        "hours_counted": len(hours),
        "events": total_events,
        "events_stations_inactives": sum(
            h.get("events_stations_inactives", 0) for h in hours.values()
        ),
        "clean_pairs": total_pairs,
        "association_rate": round(total_pairs / total_events, 3) if total_events else None,
        "updated_at_unix": now,
    }
    write_json_atomic(settings.pairs_progress_path, progress)
    return {
        "clean_pairs_total": total_pairs,
        "events_counted": total_events,
        "events_stations_inactives": progress["totals"]["events_stations_inactives"],
        "association_rate": progress["totals"]["association_rate"],
        "hours_counted": len(hours),
        "hours_pending": max(0, len(pending) - max_hours),
        "rule": (
            f"aéronef le plus proche en distance oblique à max_ts, retenu si "
            f"<= {settings.pairs_max_slant_km} km"
        ),
    }


# ------------------------------------------------------------------- verdict
def _checks(settings: Settings, collecte: dict, credits: dict, bruit: dict) -> list[dict]:
    age = collecte["last_snapshot_age_s"]
    remaining = credits.get("remaining")
    return [
        {
            "name": "snapshot_recent",
            "ok": age is not None and age <= settings.status_max_snapshot_age_s,
            "detail": f"dernier snapshot il y a {age} s (seuil {settings.status_max_snapshot_age_s} s)",
        },
        {
            "name": "debit_horaire",
            "ok": collecte["snapshots_last_hour"] > 0,
            "detail": f"{collecte['snapshots_last_hour']} snapshots sur la dernière heure",
        },
        {
            "name": "budget_credits",
            "ok": remaining is None or remaining > settings.credit_floor,
            "detail": f"crédits restants={remaining} (plancher {settings.credit_floor})",
        },
        # Fenêtre GLISSANTE, pas jour calendaire : « aujourd'hui » vaut zéro à
        # 00:06 UTC et ferait passer une collecte saine pour morte chaque nuit.
        {
            "name": "collecte_bruit",
            "ok": bruit["events_last_24h"] > 0,
            "detail": (
                f"{bruit['events_last_24h']} événements de survol sur 24 h glissantes "
                f"({bruit['events_today']} depuis 00:00 UTC)"
            ),
        },
        # Contrôle DISTINCT du poller avion : le token Bruitparif est scrapé dans
        # le HTML de la SPA. Un redéploiement du front casse le motif et arrête la
        # collecte de bruit alors que le poller continue de tourner.
        {
            "name": "token_bruitparif",
            "ok": bruit["token"]["ok"] is not False,
            "detail": (
                f"dernière récupération de token : {bruit['token']['ok']} "
                f"({bruit['token']['checked_at_iso']})"
                + (f" — {bruit['token']['error']}" if bruit["token"]["error"] else "")
            ),
        },
        {
            "name": "bruit_recent",
            "ok": _noise_is_fresh(settings, bruit),
            "detail": (
                f"dernier événement de bruit il y a {bruit['last_event_age_s']} s "
                f"(budget {settings.noise_silence_budget_s} s = cadence du collecteur "
                f"+ latence de publication ; journée={bruit['daytime']})"
            ),
        },
        # Signal DIRECT : le collecteur tourne-t-il encore ? Indépendant de la
        # latence de publication, donc bien plus rapide à lever le doute.
        {
            "name": "collecteur_bruit_vivant",
            "ok": (
                bruit["last_collect_run_age_s"] is not None
                and bruit["last_collect_run_age_s"] <= settings.noise_run_max_age_s
            ),
            "detail": (
                f"dernier passage du collecteur il y a {bruit['last_collect_run_age_s']} s "
                f"(seuil {settings.noise_run_max_age_s} s)"
            ),
        },
        {
            "name": "config_stations",
            "ok": bruit["station_config_error"] is None,
            "detail": bruit["station_config_error"] or f"{bruit['stations_configured']} station(s) active(s)",
        },
    ]


def _noise_is_fresh(settings: Settings, bruit: dict) -> bool:
    """La source bruit est-elle vivante ?

    De nuit, on ne juge pas : aucun survol n'est attendu. En journée, un silence
    de plus de `CIEL_NOISE_MAX_SILENCE_S` (2 h par défaut) fait passer le verdict
    global à KO **même si le poller avion tourne normalement** — c'est tout
    l'intérêt d'un contrôle séparé.
    """
    if not bruit["daytime"]:
        return True
    age = bruit["last_event_age_s"]
    if age is None:
        return False
    return age <= settings.noise_silence_budget_s


def build_status(
    settings: Settings | None = None,
    now: float | None = None,
    with_pairs: bool = True,
    recount_pairs: bool = False,
) -> dict:
    """Assemble le document de supervision (sans l'écrire)."""
    settings = settings or get_settings()
    now = time.time() if now is None else now

    beat = read_heartbeat(settings.heartbeat_path) or {}
    collecte = collection_state(settings, now)
    bruit = noise_state(settings, now)
    credits = {
        "remaining": beat.get("credits_remaining"),
        "daily_budget": settings.daily_credit_budget,
        "floor": settings.credit_floor,
        "as_of_age_s": collecte["heartbeat_age_s"],
    }
    paires = (
        update_pairs(settings, now, recount=recount_pairs)
        if with_pairs
        else {"clean_pairs_total": None}
    )

    checks = _checks(settings, collecte, credits, bruit)
    failed = [c["name"] for c in checks if not c["ok"]]
    return {
        "generated_at_unix": round(now, 1),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "verdict": VERDICT_KO if failed else VERDICT_OK,
        "failed_checks": failed,
        "checks": checks,
        "collecte": collecte,
        "credits": credits,
        "bruit": bruit,
        "paires": paires,
        "bbox_deg2": round(settings.bbox.area_deg2, 3),
    }


def write_status(
    settings: Settings | None = None,
    now: float | None = None,
    with_pairs: bool = True,
    recount_pairs: bool = False,
    notify: bool = True,
) -> dict:
    """Construit puis écrit `status/status.json` (atomique). Retourne le document.

    Publie ensuite une notification **si et seulement si** l'état a changé
    (cf. `monitoring.notify`). L'écriture du rapport prime : un échec de
    notification ne doit jamais empêcher le fichier d'être à jour.
    """
    settings = settings or get_settings()
    payload = build_status(settings, now, with_pairs=with_pairs, recount_pairs=recount_pairs)
    write_json_atomic(settings.status_path, payload)
    if notify:
        try:
            notify_from_status(settings, payload, now)
        except Exception:  # noqa: BLE001 — la supervision passe avant l'alerte
            logger.exception("Notification impossible — rapport tout de même écrit.")
    return payload


def main(argv: list[str] | None = None) -> int:
    configure_logging("status")
    parser = argparse.ArgumentParser(description="Écrit status/status.json (supervision).")
    parser.add_argument("--no-pairs", action="store_true", help="Ne pas compter les paires.")
    parser.add_argument("--recount", action="store_true", help="Recompter toutes les heures.")
    parser.add_argument("--print", action="store_true", help="Afficher le JSON produit.")
    parser.add_argument(
        "--no-notify", action="store_true", help="Ne pas publier de notification ntfy."
    )
    parser.add_argument(
        "--every",
        type=float,
        default=None,
        help="Boucler en attendant N secondes entre deux rapports (ordonnanceur "
        "conteneurisé ; sans cette option, un seul passage).",
    )
    args = parser.parse_args(argv)

    while True:
        try:
            payload = write_status(
                with_pairs=not args.no_pairs,
                recount_pairs=args.recount,
                notify=not args.no_notify,
            )
            resume = (
                f"{payload['verdict']} — dernier snapshot "
                f"{payload['collecte']['last_snapshot_age_s']} s, "
                f"{payload['collecte']['snapshots_last_hour']} snapshots/h, "
                f"crédits={payload['credits']['remaining']}, "
                f"bruit du jour={payload['bruit']['events_today']}, "
                f"paires={payload['paires'].get('clean_pairs_total')}"
            )
            logger.info("%s | controles en echec : %s", resume, payload["failed_checks"])
            if args.every is None:
                print(json.dumps(payload, ensure_ascii=False, indent=2) if args.print else resume)
                return 0 if payload["verdict"] == VERDICT_OK else 1
        except Exception:  # noqa: BLE001 — un passage raté ne tue pas l'ordonnanceur
            logger.exception("Passage de supervision en échec.")
            if args.every is None:
                return 1
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
