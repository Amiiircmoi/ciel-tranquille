"""Tests du sondage des stations, de la config externalisée et de la validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ciel_tranquille.config import (
    MAX_BBOX_AREA_DEG2,
    STATIONS,
    BoundingBox,
    Settings,
    StationConfigError,
    load_stations_file,
)
from ciel_tranquille.ingest import stations as st_mod
from ciel_tranquille.ingest.bruitparif_client import BruitparifRateLimited

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
FRESH_DATA_ISO = "2026-08-25T11:00:00.000000+0000"   # 1 h avant NOW
STALE_DATA_ISO = "2018-01-29T16:45:00.000000+0000"   # station éteinte de longue date


def default_bbox() -> BoundingBox:
    """Bbox telle qu'elle est LIVRÉE, indépendamment du `.env` de la machine.

    Les tests doivent qualifier la configuration par défaut du dépôt, pas la
    configuration locale du poste : un `.env` de développement plus étroit ferait
    passer ou échouer ces tests pour de mauvaises raisons.
    """
    fields = Settings.model_fields
    return BoundingBox(
        lat_min=fields["bbox_lat_min"].default,
        lon_min=fields["bbox_lon_min"].default,
        lat_max=fields["bbox_lat_max"].default,
        lon_max=fields["bbox_lon_max"].default,
    )


def _wide(settings: Settings) -> Settings:
    """Force la bbox livrée sur un Settings (neutralise le `.env` local)."""
    bbox = default_bbox()
    settings.bbox_lat_min, settings.bbox_lat_max = bbox.lat_min, bbox.lat_max
    settings.bbox_lon_min, settings.bbox_lon_max = bbox.lon_min, bbox.lon_max
    return settings


def _site(mid: str, lat: float, lon: float, **extra) -> dict:
    """Entrée `/sites` au schéma RÉEL relevé en production.

    `categories` au pluriel, `status` entier (1 = mesure en cours), et une
    fraîcheur `last_available_data` — les trois pièges du vrai schéma.
    """
    payload = {
        "measurement_id": mid,
        "site_name": mid,
        "latitude": lat,
        "longitude": lon,
        "categories": "air",
        "status": 1,
        "status_label": "Mesure en cours",
        "permanent": True,
        "last_available_data": FRESH_DATA_ISO,
        "description": mid,
    }
    payload.update(extra)
    return payload


def _event(event_id: int, usable: bool = True) -> dict:
    return {
        "category": "air",
        "id": event_id,
        "start": "2026-08-24T06:00:00Z",
        "end": "2026-08-24T06:00:40Z",
        "max_ts": "2026-08-24T06:00:20Z" if usable else None,
        "max_laeq": 78.0 if usable else None,
    }


class _StubClient:
    """Client Bruitparif factice : /sites fixe, /events paramétrable par station."""

    def __init__(self, sites: list[dict], events: dict[str, list[dict]] | None = None):
        self._sites = sites
        self._events = events or {}
        self.event_calls: list[tuple[str, datetime, datetime]] = []
        self.closed = False

    def fetch_sites(self):
        return self._sites

    def fetch_events(self, station, frm, to):
        self.event_calls.append((station, frm, to))
        result = self._events.get(station, [])
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


# ------------------------------------------------------------------------ bbox
def test_default_bbox_stays_in_the_single_credit_tier():
    """Une bbox <= 25 deg² coûte 1 crédit : c'est l'invariant du budget quotidien."""
    bbox = default_bbox()
    assert bbox.area_deg2 <= MAX_BBOX_AREA_DEG2
    # ...et elle est dimensionnée large : elle englobe les trois plateformes.
    for lat, lon in st_mod.AIRPORTS.values():
        assert bbox.contains(lat, lon)
    for station in STATIONS:
        assert bbox.contains(station.latitude, station.longitude)


# -------------------------------------------------------- configuration active
def test_repo_station_file_declares_an_explicit_active_list():
    from ciel_tranquille.config import REPO_STATIONS_FILE

    assert REPO_STATIONS_FILE.exists(), "config/stations.json doit être livré avec le dépôt"
    payload = json.loads(REPO_STATIONS_FILE.read_text(encoding="utf-8"))
    assert payload["active"], "le champ 'active' doit être explicite"
    stations = load_stations_file(REPO_STATIONS_FILE)
    assert len(stations) == len(payload["active"])
    bbox = default_bbox()
    for station in stations:
        assert bbox.contains(station.latitude, station.longitude), station.measurement_id


def test_active_list_drives_what_is_collected(tmp_path, monkeypatch):
    """Une station décrite mais absente de 'active' n'est pas collectée."""
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "active": ["TEST-B"],
                "stations": [
                    {"measurement_id": "TEST-A", "latitude": 48.9, "longitude": 2.4, "airport": "CDG"},
                    {"measurement_id": "TEST-B", "latitude": 48.7, "longitude": 2.4, "airport": "ORY"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CIEL_STATIONS_FILE", str(path))
    assert [s.measurement_id for s in Settings().stations] == ["TEST-B"]


def test_missing_active_field_fails_loudly(tmp_path, monkeypatch):
    """Sans liste active explicite, on refuse de deviner — pas de repli silencieux."""
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps({"stations": [{"measurement_id": "A", "latitude": 48.9, "longitude": 2.4}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CIEL_STATIONS_FILE", str(path))
    with pytest.raises(StationConfigError, match="active"):
        _ = Settings().stations


def test_unreadable_station_file_fails_loudly(tmp_path, monkeypatch):
    path = tmp_path / "stations.json"
    path.write_text("{ pas du json", encoding="utf-8")
    monkeypatch.setenv("CIEL_STATIONS_FILE", str(path))
    with pytest.raises(StationConfigError):
        _ = Settings().stations


def test_active_station_absent_from_catalogue_is_rejected(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "active": ["FANTOME"],
                "stations": [{"measurement_id": "A", "latitude": 48.9, "longitude": 2.4}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StationConfigError, match="FANTOME"):
        load_stations_file(path)


# --------------------------------------------------------------------- /sites
def test_parse_sites_filters_category_and_bbox():
    bbox = default_bbox()
    sites = [
        _site("DANS-BBOX", 48.99, 2.45),
        _site("HORS-BBOX", 43.30, 5.40),                     # Marseille
        _site("FERROVIAIRE", 48.99, 2.45, categories="fer"),  # observatoire SNCF
        _site("ROUTIER", 48.99, 2.45, categories="route"),
        _site("SANS-CATEGORIE", 48.99, 2.45, categories=None),
        _site("TROP-LOIN", 49.45, 3.95),                     # dans la bbox, loin des pistes
        {"measurement_id": "INCOMPLET"},                     # sans coordonnées
    ]
    kept = {c.measurement_id for c in st_mod.parse_sites(sites, bbox, NOW)}
    assert kept == {"DANS-BBOX"}


def test_parse_sites_reads_the_real_schema_field_names():
    """`categories` (pluriel) et `status` ENTIER : les deux pièges du vrai schéma.

    Une station ferroviaire lue via `category` (singulier, absent de la réponse)
    passerait le filtre et serait sondée pour des survols — un appel gaspillé sur
    une IP partagée avec une production tierce.
    """
    bbox = default_bbox()
    rail = {
        "measurement_id": "FER", "latitude": 48.99, "longitude": 2.45,
        "categories": "fer", "status": 0, "status_label": "Pas de mesure",
    }
    assert st_mod.parse_sites([rail], bbox, NOW) == []


def test_stale_or_stopped_stations_are_not_listed_active():
    """Déclarée « Mesure en cours » mais muette depuis 10 jours : pas une candidate."""
    bbox = default_bbox()
    sites = [
        _site("VIVANTE", 48.99, 2.45),
        _site("ETEINTE", 48.991, 2.451, status=0, status_label="Pas de mesure"),
        _site("PERIMEE", 48.992, 2.452, last_available_data=STALE_DATA_ISO),
        _site("SANS-DATE", 48.993, 2.453, last_available_data=None),
    ]
    got = {c.measurement_id: c.listed_active for c in st_mod.parse_sites(sites, bbox, NOW)}
    assert got == {"VIVANTE": True, "ETEINTE": False, "PERIMEE": False, "SANS-DATE": False}


def test_last_data_age_is_reported():
    bbox = default_bbox()
    cand = st_mod.parse_sites([_site("A", 48.99, 2.45)], bbox, NOW)[0]
    assert cand.last_data_age_h == 1.0
    assert cand.last_data_iso == FRESH_DATA_ISO


def test_parse_sites_labels_the_nearest_corridor():
    bbox = default_bbox()
    sites = [
        _site("PRES-CDG", 49.00, 2.55),
        _site("PRES-ORY", 48.72, 2.38),
        _site("PRES-LBG", 48.965, 2.44),
    ]
    got = {c.measurement_id: c.airport for c in st_mod.parse_sites(sites, bbox, NOW)}
    assert got == {"PRES-CDG": "CDG", "PRES-ORY": "ORY", "PRES-LBG": "LBG"}


def test_rank_candidates_spreads_across_corridors():
    """Le bornage sert les trois couloirs à tour de rôle, CDG ne prend pas tout."""
    bbox = default_bbox()
    sites = [_site(f"CDG-{i}", 49.00 + i * 0.001, 2.55) for i in range(8)]
    sites += [_site(f"ORY-{i}", 48.72 + i * 0.001, 2.38) for i in range(4)]
    sites += [_site(f"LBG-{i}", 48.965 + i * 0.001, 2.44) for i in range(2)]

    ranked = st_mod.rank_candidates(st_mod.parse_sites(sites, bbox, NOW), limit=9)
    by_corridor: dict[str, int] = {}
    for cand in ranked:
        by_corridor[cand.airport] = by_corridor.get(cand.airport, 0) + 1

    assert len(ranked) == 9
    assert set(by_corridor) == {"CDG", "ORY", "LBG"}
    assert by_corridor["ORY"] >= 3 and by_corridor["LBG"] >= 2


# -------------------------------------------------------------------- sondage
def test_probe_window_targets_a_busy_daytime_slot(settings):
    settings.probe_window_hours = 6
    settings.probe_window_start_utc = 6
    start, end = st_mod.probe_window(settings, NOW)
    # Dernier jour COMPLET : les événements du jour même ne sont pas encore publiés.
    assert start.isoformat() == "2026-08-24T06:00:00+00:00"
    assert (end - start).total_seconds() == 6 * 3600


def test_probe_makes_one_call_per_station_and_ranks_by_yield(settings):
    _wide(settings)
    sites = [_site("A-CDG", 49.00, 2.55), _site("B-ORY", 48.72, 2.38)]
    stub = _StubClient(
        sites,
        events={
            "A-CDG": [_event(i) for i in range(12)],
            "B-ORY": [_event(i) for i in range(30)],
        },
    )
    results, active = st_mod.run_probe(settings, client=stub, now=NOW, sleep=lambda _s: None)

    assert len(stub.event_calls) == 2  # un seul appel par station, pas 24
    by_id = {r.measurement_id: r for r in results}
    assert by_id["B-ORY"].usable == 30
    assert by_id["A-CDG"].usable == 12
    # Les deux sont retenues, une par couloir (service à tour de rôle).
    assert set(active) == {"B-ORY", "A-CDG"}


def test_probe_ignores_unusable_events(settings):
    _wide(settings)
    stub = _StubClient(
        [_site("A-CDG", 49.00, 2.55)],
        events={"A-CDG": [_event(1), _event(2, usable=False), {"category": "road", "id": 3}]},
    )
    results, active = st_mod.run_probe(settings, client=stub, now=NOW, sleep=lambda _s: None)
    assert results[0].events == 3
    assert results[0].usable == 1
    assert active == ["A-CDG"]


def test_probe_stops_cleanly_on_429(settings):
    """429 = arrêt immédiat : on garde le sondé, on ne tente pas les suivantes."""
    _wide(settings)
    sites = [_site("A-CDG", 49.00, 2.55), _site("B-ORY", 48.72, 2.38), _site("C-LBG", 48.965, 2.44)]
    stub = _StubClient(
        sites,
        events={
            "A-CDG": [_event(1)],
            "B-ORY": BruitparifRateLimited(60.0),
            "C-LBG": [_event(2)],
        },
    )
    candidates = st_mod.rank_candidates(st_mod.parse_sites(sites, settings.bbox, NOW), limit=10)
    results = st_mod.probe_stations(candidates, settings, client=stub, now=NOW, sleep=lambda _s: None)

    assert [r.measurement_id for r in results] == ["A-CDG"]
    assert len(stub.event_calls) == 2  # la 3e n'est jamais tentée


def test_probe_respects_the_one_second_floor(settings):
    """Le plancher d'1 s entre appels ne peut pas être contourné par la config."""
    _wide(settings)
    settings.bruitparif_pause_s = 0.0  # tentative de configuration impolie
    waits: list[float] = []
    sites = [_site("A-CDG", 49.00, 2.55), _site("B-ORY", 48.72, 2.38)]
    stub = _StubClient(sites, events={"A-CDG": [_event(1)], "B-ORY": [_event(2)]})
    candidates = st_mod.rank_candidates(st_mod.parse_sites(sites, settings.bbox, NOW), limit=10)
    st_mod.probe_stations(candidates, settings, client=stub, now=NOW, sleep=waits.append)

    assert waits and all(w >= 1.0 for w in waits)


def test_select_active_keeps_the_validated_socle_first():
    """Les stations socle restent actives : changer de socle casserait l'historique."""
    results = [
        st_mod.ProbeResult("NOUVELLE", 40, 40, 48.9, 2.4, "CDG", "n"),
        st_mod.ProbeResult(STATIONS[0].measurement_id, 5, 5, 48.98, 2.44, "CDG", "s"),
    ]
    active = st_mod.select_active(results, max_active=10)
    assert active[0] == STATIONS[0].measurement_id
    assert "NOUVELLE" in active


def test_select_active_caps_at_ten_stations():
    results = [
        st_mod.ProbeResult(f"S{i:02d}", 50 - i, 50 - i, 48.9, 2.4, "CDG", "x") for i in range(20)
    ]
    active = st_mod.select_active(results, socle=(), max_active=10)
    assert len(active) == 10
    assert active[0] == "S00"  # meilleur rendement en tête


def test_select_active_excludes_mute_and_failing_stations():
    results = [
        st_mod.ProbeResult("MUETTE", 0, 0, 48.9, 2.4, "CDG", "x"),
        st_mod.ProbeResult("KO", 0, 0, 48.9, 2.4, "CDG", "x", error="timeout"),
        st_mod.ProbeResult("BONNE", 20, 20, 48.9, 2.4, "CDG", "x"),
    ]
    assert st_mod.select_active(results, socle=(), max_active=10) == ["BONNE"]


def test_build_config_writes_an_explicit_active_field():
    results = [st_mod.ProbeResult("A", 10, 9, 48.9, 2.4, "CDG", "lbl")]
    payload = st_mod.build_config(results, ["A"], default_bbox(), 6, now=0.0)
    assert payload["active"] == ["A"]
    assert payload["stations"][0]["usable_events_6h"] == 9
    assert payload["stations"][0]["latitude"] == 48.9
    assert payload["bbox"]["area_deg2"] <= MAX_BBOX_AREA_DEG2


# ----------------------------------------------------------------- validation
def test_validation_rejects_a_station_outside_the_bbox(settings, tmp_path, monkeypatch):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "active": ["LOIN"],
                "stations": [
                    {"measurement_id": "LOIN", "latitude": 43.30, "longitude": 5.40, "airport": "CDG"}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings.stations_file = str(path)
    with pytest.raises(st_mod.StationValidationError, match="hors bbox"):
        st_mod.validate_active_stations(settings, check_network=False)


def test_validation_rejects_a_station_that_does_not_answer(settings, tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "active": ["MUETTE"],
                "stations": [
                    {"measurement_id": "MUETTE", "latitude": 48.985, "longitude": 2.447, "airport": "CDG"}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings.stations_file = str(path)
    stub = _StubClient([], events={"MUETTE": RuntimeError("503")})
    with pytest.raises(st_mod.StationValidationError, match="ne répond pas"):
        st_mod.validate_active_stations(settings, client=stub, now=NOW, sleep=lambda _s: None)


def test_validation_passes_on_a_healthy_station(settings, tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "active": ["OK-STATION"],
                "stations": [
                    {"measurement_id": "OK-STATION", "latitude": 48.985, "longitude": 2.447, "airport": "CDG"}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings.stations_file = str(path)
    stub = _StubClient([], events={"OK-STATION": [_event(1), _event(2)]})
    report = st_mod.validate_active_stations(settings, client=stub, now=NOW, sleep=lambda _s: None)
    assert report == [
        {"measurement_id": "OK-STATION", "in_bbox": True, "responds": True, "usable_events": 2}
    ]


# ------------------------------------------- divergence de coordonnées
def test_coordinate_divergence_is_reported_not_overwritten():
    """Une position qui a bougé de plus de 50 m est un fait à trancher, pas à corriger.

    Les coordonnées de station sont l'origine de la distance oblique, donc de la
    cible du modèle. Les remplacer en silence changerait rétroactivement le sens
    des paires déjà collectées.
    """
    from ciel_tranquille.config import Station

    bbox = default_bbox()
    configuree = Station("DEPLACEE", 48.98500, 2.44712, "CDG")
    # ~340 m plus au nord
    sites = [_site("DEPLACEE", 48.98805, 2.44712)]
    problems = st_mod.check_coordinates((configuree,), sites, bbox, NOW)

    assert len(problems) == 1
    assert "DEPLACEE" in problems[0] and "écart" in problems[0]
    # La station configurée est INCHANGÉE : aucune écriture silencieuse.
    assert configuree.latitude == 48.98500


def test_coordinate_rounding_is_not_a_divergence():
    """L'arrondi décimal entre config et /sites ne doit pas déclencher d'alerte."""
    from ciel_tranquille.config import Station

    configuree = Station("ARRONDIE", 48.985000, 2.427967, "ORY")
    sites = [_site("ARRONDIE", 48.985000, 2.4279673)]
    assert st_mod.check_coordinates((configuree,), sites, default_bbox(), NOW) == []


def test_validation_fails_loudly_on_diverging_coordinates(settings, tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(
        json.dumps(
            {
                "active": ["DEPLACEE"],
                "stations": [
                    {"measurement_id": "DEPLACEE", "latitude": 48.985, "longitude": 2.447, "airport": "CDG"}
                ],
            }
        ),
        encoding="utf-8",
    )
    settings.stations_file = str(path)
    stub = _StubClient([_site("DEPLACEE", 48.99500, 2.447)], events={"DEPLACEE": [_event(1)]})
    with pytest.raises(st_mod.StationValidationError, match="écart"):
        st_mod.validate_active_stations(settings, client=stub, now=NOW, sleep=lambda _s: None)


# ------------------------------------ départage des stations saturées
def _probe(mid, usable, airport, distance_km):
    return st_mod.ProbeResult(
        mid, usable, usable, 48.9, 2.4, airport, mid,
        saturated=usable >= 45, distance_km=distance_km,
    )


def test_saturated_stations_are_separated_by_proximity():
    """À rendement plafonné, c'est la proximité qui départage, pas l'ordre alphabétique.

    Le plafond de ~50 réponses met la moitié des stations ex æquo. Départager par
    identifiant retiendrait des stations à 20 km, qui n'entendent que des survols
    hauts et lointains — donc des appariements ambigus.
    """
    results = [
        _probe("ZZ-PROCHE", 50, "CDG", 1.2),
        _probe("AA-LOINTAINE", 50, "CDG", 22.5),
    ]
    assert st_mod.select_active(results, socle=(), max_active=1) == ["ZZ-PROCHE"]


def test_yield_still_beats_proximity():
    """La proximité n'est qu'un départage : un vrai écart de rendement prime."""
    results = [
        _probe("PROCHE-MAIS-CALME", 3, "CDG", 0.5),
        _probe("LOIN-MAIS-ACTIVE", 50, "CDG", 18.0),
    ]
    assert st_mod.select_active(results, socle=(), max_active=1) == ["LOIN-MAIS-ACTIVE"]


def test_selection_spreads_across_corridors_not_only_the_densest():
    """Dix stations sur CDG donneraient dix fois la même géométrie d'approche."""
    results = [_probe(f"CDG-{i}", 50, "CDG", 1.0 + i) for i in range(10)]
    results += [_probe(f"ORY-{i}", 50, "ORY", 1.0 + i) for i in range(5)]
    results += [_probe(f"LBG-{i}", 50, "LBG", 1.0 + i) for i in range(5)]

    active = st_mod.select_active(results, socle=(), max_active=9)
    couloirs = {}
    for mid in active:
        couloirs[mid.split("-")[0]] = couloirs.get(mid.split("-")[0], 0) + 1
    assert len(active) == 9
    assert couloirs == {"CDG": 3, "ORY": 3, "LBG": 3}


def test_mute_socle_station_is_not_kept_active():
    """Une station socle qui ne produit plus n'est retenue par rien : c'est le but."""
    from ciel_tranquille.config import STATIONS as SOCLE

    results = [
        st_mod.ProbeResult(SOCLE[0].measurement_id, 0, 0, 48.9, 2.4, "CDG", "muette",
                           distance_km=1.0),
        _probe("VIVANTE", 40, "CDG", 5.0),
    ]
    active = st_mod.select_active(results, max_active=5)
    assert SOCLE[0].measurement_id not in active
    assert active == ["VIVANTE"]


def test_station_file_is_found_from_the_working_directory(tmp_path, monkeypatch):
    """Paquet installé : `REPO_ROOT` pointe dans site-packages, pas sur /app.

    Régression du smoke test conteneurisé : sans repli sur le répertoire de
    travail, l'image ne trouvait aucune station et le poller refusait de démarrer.
    """
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "stations.json").write_text(
        json.dumps(
            {
                "active": ["DEPUIS-CWD"],
                "stations": [
                    {"measurement_id": "DEPUIS-CWD", "latitude": 48.9, "longitude": 2.4, "airport": "CDG"}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CIEL_STATIONS_FILE", "")
    monkeypatch.setenv("CIEL_DATA_DIR", str(tmp_path / "vide"))
    monkeypatch.setattr("ciel_tranquille.config.REPO_STATIONS_FILE", tmp_path / "absent.json")
    monkeypatch.chdir(tmp_path)
    assert [s.measurement_id for s in Settings().stations] == ["DEPUIS-CWD"]
