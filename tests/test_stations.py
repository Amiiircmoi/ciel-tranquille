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
    payload = {
        "measurement_id": mid,
        "latitude": lat,
        "longitude": lon,
        "category": "air",
        "status": "active",
        "name": mid,
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
        _site("HORS-BBOX", 43.30, 5.40),                 # Marseille
        _site("PAS-AIR", 48.99, 2.45, category="road"),
        _site("TROP-LOIN", 49.45, 3.95),                 # dans la bbox, loin des pistes
        {"measurement_id": "INCOMPLET"},                 # sans coordonnées
    ]
    kept = {c.measurement_id for c in st_mod.parse_sites(sites, bbox)}
    assert kept == {"DANS-BBOX"}


def test_parse_sites_labels_the_nearest_corridor():
    bbox = default_bbox()
    sites = [
        _site("PRES-CDG", 49.00, 2.55),
        _site("PRES-ORY", 48.72, 2.38),
        _site("PRES-LBG", 48.965, 2.44),
    ]
    got = {c.measurement_id: c.airport for c in st_mod.parse_sites(sites, bbox)}
    assert got == {"PRES-CDG": "CDG", "PRES-ORY": "ORY", "PRES-LBG": "LBG"}


def test_rank_candidates_spreads_across_corridors():
    """Le bornage sert les trois couloirs à tour de rôle, CDG ne prend pas tout."""
    bbox = default_bbox()
    sites = [_site(f"CDG-{i}", 49.00 + i * 0.001, 2.55) for i in range(8)]
    sites += [_site(f"ORY-{i}", 48.72 + i * 0.001, 2.38) for i in range(4)]
    sites += [_site(f"LBG-{i}", 48.965 + i * 0.001, 2.44) for i in range(2)]

    ranked = st_mod.rank_candidates(st_mod.parse_sites(sites, bbox), limit=9)
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
    # Aucune station socle ne répond ici : classement pur rendement.
    assert active == ["B-ORY", "A-CDG"]


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
    candidates = st_mod.rank_candidates(st_mod.parse_sites(sites, settings.bbox), limit=10)
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
    candidates = st_mod.rank_candidates(st_mod.parse_sites(sites, settings.bbox), limit=10)
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
