"""Stations Bruitparif — **sondage** de rendement, pas catalogue de confiance.

Le réseau Survol compte quelques dizaines de stations permanentes et **évolue** :
des points ouvrent, ferment, tombent en panne. Deux conséquences de conception :

1. **Rien n'est écrit en dur.** Aucun identifiant, nom ou coordonnée n'est inventé
   ni recopié de mémoire. Les candidats viennent de `GET /sites`, seul endroit où
   l'API expose les identifiants **et leurs coordonnées** — coordonnées sans
   lesquelles ni la distance oblique de la jointure ni la validation de bbox ne
   sont possibles (`/events` ne renvoie aucune position, vérifié sur les payloads).
2. **C'est le sondage qui tranche, pas le catalogue.** Une station peut être
   listée « active » et ne rien produire. On interroge donc l'endpoint événements
   sur une **fenêtre diurne chargée** (6 h par défaut) et on ne retient que celles
   qui renvoient des événements exploitables, classées par rendement observé.

Politesse réseau, non négociable — l'IP sortante est partagée avec une production
tierce : **≥ 1 s entre deux appels** (plancher dur), back-off exponentiel sur
erreur, User-Agent identifiable avec contact, et **arrêt propre immédiat sur 429**
(on conserve ce qui a déjà été sondé, on n'insiste jamais).

    python -m ciel_tranquille.ingest.stations probe --write
    python -m ciel_tranquille.ingest.stations list
    python -m ciel_tranquille.ingest.stations validate
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ciel_tranquille.config import (
    MAX_BBOX_AREA_DEG2,
    STATIONS,
    BoundingBox,
    Settings,
    Station,
    StationConfigError,
    get_settings,
)
from ciel_tranquille.ingest.bruitparif_client import (
    BruitparifClient,
    BruitparifRateLimited,
)
from ciel_tranquille.monitoring.heartbeat import write_json_atomic

logger = logging.getLogger(__name__)

# Plateformes de référence (coordonnées publiques) : sert à étiqueter chaque
# station par le couloir dont elle est la plus proche.
AIRPORTS: dict[str, tuple[float, float]] = {
    "CDG": (49.0097, 2.5479),   # Paris-Charles-de-Gaulle
    "ORY": (48.7233, 2.3794),   # Paris-Orly
    "LBG": (48.9694, 2.4414),   # Paris-Le Bourget
}

# Couloirs figés des stations socle : leur étiquette vient de l'analyse déjà
# publiée, pas de la géométrie (Gonesse est géométriquement plus près du Bourget
# mais s'analyse sous CDG). La changer casserait la comparabilité des agrégats.
_SOCLE_AIRPORTS: dict[str, str] = {s.measurement_id: s.airport for s in STATIONS}

# Une station trop éloignée de toute plateforme n'est pas sous un couloir exploitable.
MAX_AIRPORT_DISTANCE_KM = 30.0
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_airport(lat: float, lon: float) -> tuple[str, float]:
    """Plateforme la plus proche et sa distance (km)."""
    return min(
        ((code, _haversine_km(lat, lon, alat, alon)) for code, (alat, alon) in AIRPORTS.items()),
        key=lambda kv: kv[1],
    )


# ------------------------------------------------------------------ candidats
# Schéma réel de `/sites`, relevé sur la réponse de production :
#   categories    : 'air' | 'route' | 'fer'   (PLURIEL — pas `category`)
#   status        : 1 = « Mesure en cours », 0 = « Pas de mesure » (entier)
#   permanent     : station permanente vs campagne temporaire
#   last_available_data : horodatage ISO de la dernière mesure publiée
_ACTIVE_STATUS_LABELS = ("mesure en cours",)
# Au-delà, la station est déclarée active mais ne publie plus : inutile de la
# sonder, l'appel serait gaspillé sur une IP partagée avec une production tierce.
STALE_DATA_MAX_AGE_H = 48.0


@dataclass(frozen=True)
class SiteCandidate:
    """Station candidate issue de `/sites`, enrichie de son couloir."""

    measurement_id: str
    latitude: float
    longitude: float
    label: str
    airport: str
    distance_km: float
    listed_active: bool
    last_data_iso: str | None = None
    last_data_age_h: float | None = None
    permanent: bool = True

    def to_station(self) -> Station:
        return Station(
            measurement_id=self.measurement_id,
            latitude=self.latitude,
            longitude=self.longitude,
            airport=self.airport,
            label=self.label,
        )


def _first(payload: dict, *keys, default=None):
    for key in keys:
        if payload.get(key) not in (None, ""):
            return payload[key]
    return default


def _last_data_age_h(site: dict, now: datetime) -> tuple[str | None, float | None]:
    """Fraîcheur déclarée de la station (`last_available_data`)."""
    raw = _first(site, "last_available_data", "lastAvailableData")
    if not raw:
        return None, None
    text = str(raw).strip().replace("Z", "+00:00")
    if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return str(raw), None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return str(raw), max(0.0, (now - parsed).total_seconds() / 3600.0)


def parse_sites(
    sites: list[dict], bbox: BoundingBox, now: datetime | None = None
) -> list[SiteCandidate]:
    """Filtre `/sites` : catégorie **air**, dans la bbox, sous un couloir.

    Le schéma de `/sites` n'est pas contractuel côté Bruitparif : on lit les clés
    par tolérance (`categories`/`category`, `measurement_id`/`id`, `latitude`/`lat`…)
    et on ignore les entrées inexploitables plutôt que d'échouer sur un champ
    renommé. La catégorie est en revanche **obligatoire** : le réseau mélange
    observatoires aérien, ferroviaire et routier sur les mêmes communes, et sonder
    une station ferroviaire pour des survols ne peut que gaspiller un appel.
    """
    now = now or datetime.now(timezone.utc)
    out: list[SiteCandidate] = []
    for site in sites:
        if not isinstance(site, dict):
            continue
        category = str(_first(site, "categories", "category", "type", default="")).lower()
        if category != "air":
            continue
        mid = _first(site, "measurement_id", "measurementId", "site", "id")
        lat = _first(site, "latitude", "lat")
        lon = _first(site, "longitude", "lon", "lng")
        if mid is None or lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if not bbox.contains(lat, lon):
            continue
        airport, dist = nearest_airport(lat, lon)
        if dist > MAX_AIRPORT_DISTANCE_KM:
            continue
        # `status` est un ENTIER (1 = mesure en cours) : le comparer à des
        # chaînes comme « active » ne matcherait jamais, et une station éteinte
        # depuis 2018 passerait pour disponible.
        status = _first(site, "status", "state", default=None)
        label = str(_first(site, "status_label", default="")).strip().lower()
        declared = (status in (1, "1", True)) or label in _ACTIVE_STATUS_LABELS
        last_iso, age_h = _last_data_age_h(site, now)
        fresh = age_h is not None and age_h <= STALE_DATA_MAX_AGE_H
        listed_active = bool(declared and fresh)
        # Le couloir est déduit de la plateforme la plus proche — sauf pour les
        # stations socle, dont l'étiquette est figée par l'historique déjà
        # collecté : la rebaptiser ici rendrait les agrégats incomparables.
        out.append(
            SiteCandidate(
                measurement_id=str(mid),
                latitude=lat,
                longitude=lon,
                label=str(_first(site, "name", "label", "commune", default=str(mid))),
                airport=_SOCLE_AIRPORTS.get(str(mid), airport),
                distance_km=round(dist, 2),
                listed_active=listed_active,
                last_data_iso=last_iso,
                last_data_age_h=None if age_h is None else round(age_h, 1),
                permanent=bool(_first(site, "permanent", default=True)),
            )
        )
    return out


def rank_candidates(
    candidates: list[SiteCandidate],
    limit: int,
    socle: tuple[Station, ...] = STATIONS,
) -> list[SiteCandidate]:
    """Borne la liste à sonder, en couvrant les trois couloirs à tour de rôle.

    On sert CDG, Orly puis Le Bourget alternativement, en prenant à chaque tour la
    station la plus proche de sa plateforme. Un couloir dense n'écrase donc pas
    les autres : la diversité géométrique est précisément ce qui donne au modèle
    de quoi apprendre autre chose que « proche = fort ».

    Les stations **socle** passent devant, hors quota : sans cela, un bornage
    serré pourrait les écarter du sondage et donc de la liste active — ce qui
    romprait la continuité avec l'historique déjà collecté.
    """
    socle_ids = {s.measurement_id for s in socle}
    forced = [c for c in candidates if c.measurement_id in socle_ids]
    candidates = [c for c in candidates if c.measurement_id not in socle_ids]
    pool = [c for c in candidates if c.listed_active] or list(candidates)
    by_airport: dict[str, list[SiteCandidate]] = {}
    for cand in sorted(pool, key=lambda c: c.distance_km):
        by_airport.setdefault(cand.airport, []).append(cand)

    ordered: list[SiteCandidate] = list(forced)
    while len(ordered) < max(limit, len(forced)):
        progressed = False
        for code in AIRPORTS:
            queue = by_airport.get(code) or []
            if not queue or len(ordered) >= limit:
                continue
            ordered.append(queue.pop(0))
            progressed = True
        if not progressed:
            break
    return ordered


# -------------------------------------------------------------------- sondage
@dataclass
class ProbeResult:
    """Rendement observé d'une station sur la fenêtre de sondage."""

    measurement_id: str
    events: int
    usable: int
    latitude: float | None
    longitude: float | None
    airport: str
    label: str
    error: str | None = None
    saturated: bool = False
    distance_km: float | None = None

    @property
    def responds(self) -> bool:
        return self.error is None

    def to_dict(self, window_hours: int) -> dict:
        return {
            "measurement_id": self.measurement_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "airport": self.airport,
            "label": self.label,
            "distance_km": self.distance_km,
            f"events_{window_hours}h": self.events,
            f"usable_events_{window_hours}h": self.usable,
            # Au plafond de l'API, le compte est un PLANCHER : la station produit
            # « au moins » ce nombre. Deux stations saturées ne sont pas départageables
            # par cette mesure — et elles sont toutes deux bonnes, ce qui suffit ici.
            "saturated": self.saturated,
            "error": self.error,
        }


def probe_window(settings: Settings, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Fenêtre de sondage : `CIEL_PROBE_WINDOW_H` heures diurnes chargées.

    On sonde le **dernier jour complet** plutôt que les heures qui viennent de
    s'écouler : les événements Bruitparif sont publiés avec de la latence, et une
    fenêtre trop fraîche ferait passer une bonne station pour muette. Six heures
    de bank matinale suffisent à classer, et coûtent quatre fois moins d'appels
    qu'une fenêtre de 24 h.
    """
    now = now or datetime.now(timezone.utc)
    day = (now - timedelta(days=1)).date()
    start = datetime(
        day.year, day.month, day.day, settings.probe_window_start_utc, tzinfo=timezone.utc
    )
    return start, start + timedelta(hours=settings.probe_window_hours)


def _is_usable(event: dict) -> bool:
    """Un événement est exploitable s'il porte la cible ET la clé de jointure."""
    return (
        event.get("category") == "air"
        and event.get("max_ts") is not None
        and event.get("max_laeq") is not None
    )


def probe_stations(
    candidates: list[SiteCandidate],
    settings: Settings | None = None,
    client: BruitparifClient | None = None,
    now: datetime | None = None,
    sleep=time.sleep,
) -> list[ProbeResult]:
    """Sonde chaque candidate sur la fenêtre diurne. Un appel par station.

    Interrompt **proprement** la campagne sur 429 : les stations déjà sondées sont
    conservées et retournées, les suivantes ne sont pas tentées.
    """
    settings = settings or get_settings()
    own = client is None
    client = client or BruitparifClient(settings)
    start, end = probe_window(settings, now)
    pause = settings.polite_pause_s
    results: list[ProbeResult] = []

    logger.info(
        "Sondage de %d station(s) sur %s -> %s (%d h), pause %.1fs entre appels.",
        len(candidates), start.isoformat(), end.isoformat(), settings.probe_window_hours, pause,
    )
    try:
        for index, cand in enumerate(candidates):
            if index:
                sleep(pause)
            try:
                events = _fetch_with_backoff(client, cand.measurement_id, start, end, settings, sleep)
            except BruitparifRateLimited as exc:
                logger.error(
                    "429 pendant le sondage après %d station(s) — arrêt propre. %s",
                    len(results), exc,
                )
                break
            except Exception as exc:  # noqa: BLE001 — station muette : on la note et on continue
                logger.warning("Station %s injoignable : %s", cand.measurement_id, exc)
                results.append(
                    ProbeResult(
                        cand.measurement_id, 0, 0, cand.latitude, cand.longitude,
                        cand.airport, cand.label, error=str(exc),
                        distance_km=cand.distance_km,
                    )
                )
                continue
            usable = [e for e in events if _is_usable(e)]
            saturated = len(events) >= settings.events_saturation_threshold
            results.append(
                ProbeResult(
                    cand.measurement_id, len(events), len(usable),
                    cand.latitude, cand.longitude, cand.airport, cand.label,
                    saturated=saturated, distance_km=cand.distance_km,
                )
            )
            logger.info(
                "sondage %-44s %3d événements dont %3d exploitables%s",
                cand.measurement_id, len(events), len(usable),
                " (au plafond de l'API)" if saturated else "",
            )
    finally:
        if own:
            client.close()
    return results


def _fetch_with_backoff(
    client: BruitparifClient,
    station_id: str,
    start: datetime,
    end: datetime,
    settings: Settings,
    sleep,
) -> list[dict]:
    """Un appel de sondage, avec back-off exponentiel (429 exclu : arrêt net)."""
    last: Exception | None = None
    for attempt in range(max(1, settings.bruitparif_max_retries)):
        try:
            return client.fetch_events(station_id, start, end)
        except BruitparifRateLimited:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 >= max(1, settings.bruitparif_max_retries):
                break
            wait_s = settings.bruitparif_backoff_base_s * (2**attempt)
            logger.warning(
                "Sondage %s : tentative %d en échec, back-off %.0fs.",
                station_id, attempt + 1, wait_s,
            )
            sleep(wait_s)
    raise last if last else RuntimeError("Sondage en échec sans exception.")


def _selection_key(result: ProbeResult) -> tuple:
    """Ordre de mérite : rendement d'abord, puis PROXIMITÉ à la plateforme.

    Le plafond de ~50 réponses par appel sature une bonne moitié des stations :
    elles arrivent toutes ex æquo et le rendement ne les départage plus. Le
    second critère est alors la distance à la plateforme, et ce n'est pas un
    pis-aller. Une station lointaine qui entend 50 survols entend des avions
    haut et loin : distance oblique élevée, plusieurs appareils audibles à la
    fois, appariement ambigu. Une station sous l'axe entend le survol qui la
    concerne. À rendement égal, la proximité est la meilleure paire.
    """
    return (-result.usable, result.distance_km if result.distance_km is not None else 1e9,
            result.measurement_id)


def select_active(
    results: list[ProbeResult],
    socle: tuple[Station, ...] = STATIONS,
    max_active: int = 10,
) -> list[str]:
    """Liste active : socle d'abord, puis les meilleures, réparties sur les couloirs.

    Le socle (stations déjà validées par la collecte réelle) est conservé même si
    la fenêtre de sondage le place derrière : changer de stations socle en cours
    de route casserait la comparabilité avec l'historique déjà collecté — mais
    seulement s'il produit encore, une station muette n'étant retenue par rien.

    Les places restantes sont servies **par couloir, à tour de rôle**. Concentrer
    les dix stations sur CDG donnerait dix fois la même géométrie d'approche ;
    l'intérêt d'élargir est de varier distances et azimuts, c'est-à-dire de donner
    au modèle autre chose à apprendre que « proche = fort ».
    """
    responding = {r.measurement_id: r for r in results if r.responds and r.usable > 0}
    active: list[str] = [s.measurement_id for s in socle if s.measurement_id in responding]

    par_couloir: dict[str, list[ProbeResult]] = {}
    for result in sorted(responding.values(), key=_selection_key):
        if result.measurement_id in active:
            continue
        par_couloir.setdefault(result.airport, []).append(result)

    while len(active) < max_active:
        progressed = False
        for code in AIRPORTS:
            queue = par_couloir.get(code) or []
            if not queue or len(active) >= max_active:
                continue
            active.append(queue.pop(0).measurement_id)
            progressed = True
        if not progressed:
            break
    return active[:max_active]


def build_config(
    results: list[ProbeResult],
    active: list[str],
    bbox: BoundingBox,
    window_hours: int,
    now: float | None = None,
) -> dict:
    """Document `config/stations.json` : catalogue sondé + liste active explicite."""
    now = time.time() if now is None else now
    return {
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "source": f"sondage /events sur {window_hours} h (candidats issus de /sites)",
        "probe_window_hours": window_hours,
        "bbox": {
            "lat_min": bbox.lat_min,
            "lon_min": bbox.lon_min,
            "lat_max": bbox.lat_max,
            "lon_max": bbox.lon_max,
            "area_deg2": round(bbox.area_deg2, 3),
        },
        "active": active,
        "stations": [r.to_dict(window_hours) for r in results],
    }


def run_probe(
    settings: Settings | None = None,
    client: BruitparifClient | None = None,
    max_candidates: int | None = None,
    now: datetime | None = None,
    sleep=time.sleep,
) -> tuple[list[ProbeResult], list[str]]:
    """Campagne complète : `/sites` → bornage → sondage → classement."""
    settings = settings or get_settings()
    own = client is None
    client = client or BruitparifClient(settings)
    try:
        sites = client.fetch_sites()
        candidates = rank_candidates(
            parse_sites(sites, settings.bbox),
            limit=max_candidates or settings.probe_max_candidates,
        )
        logger.info(
            "/sites : %d entrées -> %d candidates dans la bbox (%.2f deg²).",
            len(sites), len(candidates), settings.bbox.area_deg2,
        )
        results = probe_stations(candidates, settings, client=client, now=now, sleep=sleep)
    finally:
        if own:
            client.close()
    active = select_active(results, max_active=settings.max_active_stations)
    return results, active


# ----------------------------------------------------------------- validation
class StationValidationError(RuntimeError):
    """Une station active ne répond pas ou sort de la bbox : démarrage refusé."""


# Écart au-delà duquel les coordonnées configurées et celles de `/sites` sont
# jugées incompatibles. 50 m : bien au-delà de l'arrondi décimal, bien en deçà
# d'un déplacement de station réel.
COORD_DIVERGENCE_MAX_M = 50.0


def check_coordinates(
    stations: tuple[Station, ...],
    sites: list[dict],
    bbox: BoundingBox,
    now: datetime | None = None,
) -> list[str]:
    """Compare les coordonnées configurées à celles publiées par `/sites`.

    Retourne la liste des divergences (vide si tout concorde). On **ne corrige
    jamais** en silence : les coordonnées de station sont l'origine du calcul de
    distance oblique, donc de la cible du modèle. Les remplacer à la volée
    changerait rétroactivement le sens des paires déjà collectées, sans que rien
    ne l'indique. Une divergence est un fait à trancher par un humain.
    """
    published = {}
    for cand in parse_sites(sites, bbox, now):
        published[cand.measurement_id] = (cand.latitude, cand.longitude)
    # `parse_sites` filtre sur la bbox et la catégorie : on complète avec le brut
    # pour ne pas rater une station dont les coordonnées publiées sortent de la bbox.
    for site in sites:
        if not isinstance(site, dict):
            continue
        mid = _first(site, "measurement_id", "measurementId", "site", "id")
        lat = _first(site, "latitude", "lat")
        lon = _first(site, "longitude", "lon", "lng")
        if mid is None or lat is None or lon is None:
            continue
        try:
            published.setdefault(str(mid), (float(lat), float(lon)))
        except (TypeError, ValueError):
            continue

    problems: list[str] = []
    for station in stations:
        coords = published.get(station.measurement_id)
        if coords is None:
            continue  # absence traitée ailleurs (la station ne répondra pas)
        delta_m = _haversine_km(station.latitude, station.longitude, *coords) * 1000.0
        if delta_m > COORD_DIVERGENCE_MAX_M:
            problems.append(
                f"{station.measurement_id} : coordonnées configurées "
                f"({station.latitude}, {station.longitude}) vs /sites "
                f"({coords[0]}, {coords[1]}) — écart {delta_m:.0f} m. "
                "Tranchez explicitement : la distance oblique, donc la cible du "
                "modèle, dépend de cette position. Aucune correction automatique."
            )
    return problems


def validate_active_stations(
    settings: Settings | None = None,
    client: BruitparifClient | None = None,
    check_network: bool = True,
    now: datetime | None = None,
    sleep=time.sleep,
) -> list[dict]:
    """Contrôle de démarrage : chaque station active répond et est dans la bbox.

    Échec = exception, pas avertissement. Une station hors bbox ne verra jamais
    d'aéronef ; une station muette ne produira jamais d'événement. Dans les deux
    cas la collecte tournerait six jours pour rien, et **un silence de collecte ne
    doit jamais être une panne muette**.
    """
    settings = settings or get_settings()
    stations = settings.stations  # lève StationConfigError si la config manque
    bbox = settings.bbox
    problems: list[str] = []
    report: list[dict] = []

    outside = [s for s in stations if not bbox.contains(s.latitude, s.longitude)]
    for station in outside:
        problems.append(
            f"{station.measurement_id} est hors bbox "
            f"({station.latitude}, {station.longitude}) — élargissez CT_BBOX_* "
            f"ou retirez-la de 'active'."
        )

    if check_network:
        own = client is None
        client = client or BruitparifClient(settings)
        start, end = probe_window(settings, now)
        try:
            try:
                problems.extend(check_coordinates(stations, client.fetch_sites(), bbox, now))
            except BruitparifRateLimited as exc:
                problems.append(f"429 pendant la validation ({exc}) — réessayer plus tard.")
            except Exception as exc:  # noqa: BLE001 — /sites indisponible : on le signale
                logger.warning("Contrôle des coordonnées impossible (/sites) : %s", exc)
            for index, station in enumerate(stations):
                if index:
                    sleep(settings.polite_pause_s)
                try:
                    events = client.fetch_events(station.measurement_id, start, end)
                except BruitparifRateLimited as exc:
                    problems.append(f"429 pendant la validation ({exc}) — réessayer plus tard.")
                    break
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{station.measurement_id} ne répond pas : {exc}")
                    continue
                usable = sum(1 for e in events if _is_usable(e))
                report.append(
                    {
                        "measurement_id": station.measurement_id,
                        "in_bbox": bbox.contains(station.latitude, station.longitude),
                        "responds": True,
                        "usable_events": usable,
                    }
                )
                if usable == 0:
                    logger.warning(
                        "%s répond mais n'a produit aucun événement exploitable sur la "
                        "fenêtre de contrôle — station peut-être en panne.",
                        station.measurement_id,
                    )
        finally:
            if own:
                client.close()
    else:
        report = [
            {
                "measurement_id": s.measurement_id,
                "in_bbox": bbox.contains(s.latitude, s.longitude),
                "responds": None,
                "usable_events": None,
            }
            for s in stations
        ]

    if problems:
        raise StationValidationError(
            "Validation des stations actives en échec :\n  - " + "\n  - ".join(problems)
        )
    logger.info("Validation OK : %d station(s) active(s) dans la bbox.", len(stations))
    return report


# ------------------------------------------------------------------------ CLI
def _cmd_list(settings: Settings) -> int:
    try:
        stations = settings.stations
    except StationConfigError as exc:
        print(f"Configuration de stations invalide : {exc}")
        return 1
    print(f"{len(stations)} station(s) active(s) — source : {settings.stations_path}")
    for station in stations:
        print(f"  {station.measurement_id:44s} {station.airport:3s} {station.label}")
    print(f"bbox {settings.bbox.as_params()} — {settings.bbox.area_deg2:.2f} deg²")
    return 0


def _cmd_probe(settings: Settings, args) -> int:
    if settings.bbox.area_deg2 > MAX_BBOX_AREA_DEG2:
        logger.warning(
            "bbox de %.2f deg² : au-delà de %.0f deg², /states/all coûte plus d'un crédit.",
            settings.bbox.area_deg2, MAX_BBOX_AREA_DEG2,
        )
    results, active = run_probe(settings, max_candidates=args.limit)
    window_h = settings.probe_window_hours

    print(f"\nSondage sur {window_h} h — {len(results)} station(s) interrogée(s) :")
    for result in sorted(results, key=lambda r: -r.usable):
        flag = "actif " if result.measurement_id in active else "      "
        print(
            f"  {flag}{result.measurement_id:44s} {result.airport:3s} "
            f"{result.usable:3d} exploitables / {result.events:3d}"
            + (f"  [{result.error}]" if result.error else "")
        )
    print(f"\nListe active retenue ({len(active)}/{settings.max_active_stations}) :")
    for measurement_id in active:
        print(f"  {measurement_id}")

    payload = build_config(results, active, settings.bbox, window_h)
    out = Path(args.out) if args.out else (settings.data_path / "stations.json")
    if args.write or args.out:
        write_json_atomic(out, payload)
        print(f"\nÉcrit : {out}")
        print("Relisez le champ 'active' avant de lancer la collecte.")
    else:
        print("\n(aucune écriture — ajoutez --write pour enregistrer la configuration)")
    return 0 if active else 1


def _cmd_validate(settings: Settings, args) -> int:
    try:
        report = validate_active_stations(settings, check_network=not args.offline)
    except (StationConfigError, StationValidationError) as exc:
        print(f"ÉCHEC : {exc}")
        return 1
    for row in report:
        print(
            f"  {row['measurement_id']:44s} bbox={row['in_bbox']} "
            f"réponse={row['responds']} exploitables={row['usable_events']}"
        )
    print("Validation OK.")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Stations Bruitparif : sondage, liste, validation.")
    sub = parser.add_subparsers(dest="command")

    probe = sub.add_parser("probe", help="Sonder les stations et classer par rendement.")
    probe.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max de candidates sondées (défaut CIEL_PROBE_MAX_CANDIDATES).",
    )
    probe.add_argument("--write", action="store_true", help="Écrire $CIEL_DATA_DIR/stations.json.")
    probe.add_argument("--out", type=str, default=None, help="Chemin de sortie explicite.")

    sub.add_parser("list", help="Afficher la liste active courante.")

    validate = sub.add_parser("validate", help="Valider les stations actives (bbox + réponse).")
    validate.add_argument("--offline", action="store_true", help="Ne vérifier que la bbox.")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "probe":
        return _cmd_probe(settings, args)
    if args.command == "validate":
        return _cmd_validate(settings, args)
    return _cmd_list(settings)


if __name__ == "__main__":
    raise SystemExit(main())
