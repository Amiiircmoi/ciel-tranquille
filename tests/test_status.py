"""Tests du rapport de supervision `status/status.json`."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from ciel_tranquille import status
from ciel_tranquille.config import Station
from ciel_tranquille.ingest.poller import write_batch
from ciel_tranquille.monitoring.heartbeat import Heartbeat, write_heartbeat

# 2026-08-25T12:00:00Z
NOON_UTC = 1_787_659_200
STATION = Station("95500-GONESSE-MEDIATHEQUE-M", 48.985000, 2.447118, "CDG")


def _state_record(ts: int, icao: str, lat: float, lon: float, alt: float = 800.0) -> dict:
    return {
        "icao24": icao,
        "callsign": "TEST123",
        "origin_country": "France",
        "time_position_unix": ts,
        "longitude": lon,
        "latitude": lat,
        "baro_altitude_m": alt,
        "velocity_m_s": 90.0,
        "heading_deg": 90.0,
        "squawk": None,
        "last_contact_unix": ts,
        "on_ground": False,
        "snapshot_ts": ts,
    }


def _seed_snapshots(settings, stamps, lat=48.985, lon=2.447):
    for ts in stamps:
        write_batch([_state_record(ts, "aa0001", lat, lon)], settings.landing_dir)


def _seed_noise_events(settings, day: str, events: list[dict]) -> None:
    part = settings.noise_events_dir / f"station={STATION.measurement_id}" / f"date={day}"
    part.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(events)
    pq.write_table(table, part / f"events_{day}.parquet")


def _noise_event(event_id: int, max_ts: float) -> dict:
    return {
        "station": STATION.measurement_id,
        "airport": "CDG",
        "latitude": STATION.latitude,
        "longitude": STATION.longitude,
        "event_id": event_id,
        "category": "air",
        "start_ts_unix": max_ts - 20,
        "end_ts_unix": max_ts + 20,
        "max_ts_unix": max_ts,
        "max_ts_iso": "",
        "duration_s": 40.0,
        "laeq": 68.0,
        "max_laeq": 78.0,
        "sel": 84.0,
        "nrj_laeq": 1.0,
        "valid": True,
        "collect_day": "2026-08-25",
    }


def test_collection_state_reports_age_and_hourly_rate(settings):
    now = NOON_UTC + 3600
    # 3 snapshots dans la dernière heure, 1 bien plus ancien.
    _seed_snapshots(settings, [now - 30, now - 60, now - 90, now - 7200])
    state = status.collection_state(settings, now)

    assert state["last_snapshot_age_s"] == 30
    assert state["snapshots_last_hour"] == 3
    assert state["snapshots_in_landing"] == 4


def test_collection_state_falls_back_to_heartbeat(settings):
    """Sans fichier en landing (compaction passée), le heartbeat fait foi."""
    now = NOON_UTC + 100
    write_heartbeat(
        settings.heartbeat_path,
        Heartbeat(
            updated_at_unix=now - 40,
            snapshot_ts=NOON_UTC + 60,
            rows=12,
            batches_total=5,
            credits_remaining=3500,
            poll_interval_s=30,
            mode="live",
            pid=1,
        ),
    )
    state = status.collection_state(settings, now)
    assert state["last_snapshot_ts"] == NOON_UTC + 60
    assert state["heartbeat_age_s"] == 40
    assert state["mode"] == "live"


def test_noise_state_counts_today_only(settings):
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, NOON_UTC), _noise_event(2, NOON_UTC + 60)])
    _seed_noise_events(settings, "2026-08-24", [_noise_event(3, NOON_UTC - 86_400)])
    state = status.noise_state(settings, NOON_UTC)

    assert state["day_utc"] == "2026-08-25"
    assert state["events_today"] == 2
    assert state["stations_reporting_today"] == 1


def test_verdict_ok_when_collection_is_healthy(settings, monkeypatch):
    monkeypatch.setenv("CIEL_STATIONS_FILE", "")
    now = NOON_UTC + 3600
    _seed_snapshots(settings, [now - i * 30 for i in range(1, 20)])
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 300)])
    write_heartbeat(
        settings.heartbeat_path,
        Heartbeat(now - 20, now - 30, 40, 100, 3200, 30, "live", 1),
    )
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "OK"
    assert payload["failed_checks"] == []
    assert payload["credits"]["remaining"] == 3200


def test_verdict_ko_when_snapshot_is_stale(settings):
    now = NOON_UTC + 7200
    _seed_snapshots(settings, [NOON_UTC])  # vieux de 2 h
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "KO"
    assert "snapshot_recent" in payload["failed_checks"]
    assert "debit_horaire" in payload["failed_checks"]


def test_verdict_ko_when_credits_below_floor(settings):
    now = NOON_UTC + 60
    _seed_snapshots(settings, [now - 30])
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 300)])
    write_heartbeat(
        settings.heartbeat_path,
        Heartbeat(now - 10, now - 30, 40, 100, 50, 30, "live", 1),
    )
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "KO"
    assert "budget_credits" in payload["failed_checks"]


def test_status_json_is_written_atomically_and_carries_no_secret(settings):
    now = NOON_UTC + 3600
    _seed_snapshots(settings, [now - 30])
    payload = status.write_status(settings, now=now, with_pairs=False)

    assert settings.status_path.exists()
    written = json.loads(settings.status_path.read_text(encoding="utf-8"))
    assert written["verdict"] == payload["verdict"]

    raw = settings.status_path.read_text(encoding="utf-8").lower()
    # Le mot « token » figure légitimement comme NOM de contrôle de supervision ;
    # ce qui ne doit jamais apparaître, c'est une valeur de secret.
    for forbidden in ("secret", "client_id", "bearer", "password", "access_token"):
        assert forbidden not in raw
    assert "value" not in json.dumps(written["bruit"]["token"])
    # Ni identifiant d'aéronef ni position : uniquement des agrégats.
    assert "icao" not in raw and "callsign" not in raw
    assert not list(settings.status_dir.glob(".*tmp"))


def test_pairs_are_counted_incrementally(settings):
    """Une heure figée est comptée une fois, puis mémorisée (cumul stable)."""
    settings.stations_file = ""
    stamps = [NOON_UTC + i * 30 for i in range(0, 120)]
    _seed_snapshots(settings, stamps, lat=STATION.latitude + 0.002, lon=STATION.longitude)
    _seed_noise_events(
        settings,
        "2026-08-25",
        [_noise_event(1, NOON_UTC + 300), _noise_event(2, NOON_UTC + 900)],
    )

    now = NOON_UTC + 12 * 3600  # bien au-delà de CIEL_PAIRS_LAG_H
    first = status.update_pairs(settings, now)
    assert first["events_counted"] == 2
    assert first["clean_pairs_total"] == 2  # avion à ~200 m : sous le seuil oblique
    assert first["hours_counted"] == 1

    # Deuxième passage : rien de neuf à compter, le cumul ne bouge pas.
    second = status.update_pairs(settings, now)
    assert second["clean_pairs_total"] == 2
    assert second["hours_counted"] == 1

    progress = json.loads(settings.pairs_progress_path.read_text(encoding="utf-8"))
    assert progress["totals"]["clean_pairs"] == 2


def test_pairs_ignore_hours_that_are_not_settled_yet(settings):
    """Une heure trop récente n'est pas comptée : le bruit n'est pas encore publié."""
    _seed_snapshots(settings, [NOON_UTC + i * 30 for i in range(4)])
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, NOON_UTC + 60)])

    result = status.update_pairs(settings, NOON_UTC + 3600)  # 1 h plus tard seulement
    assert result["hours_counted"] == 0
    assert result["clean_pairs_total"] == 0


# ----------------------------- santé de la source bruit (contrôle DISTINCT)
def _write_noise_health(settings, payload):
    from ciel_tranquille.monitoring.heartbeat import write_json_atomic

    write_json_atomic(settings.noise_health_path, payload)


def _healthy_poller(settings, now):
    """Poller avion parfaitement sain : isole l'effet des contrôles bruit."""
    _seed_snapshots(settings, [now - i * 30 for i in range(1, 20)])
    write_heartbeat(
        settings.heartbeat_path,
        Heartbeat(now - 20, now - 30, 40, 100, 3200, 30, "live", 1),
    )


def test_broken_token_is_its_own_failed_check(settings):
    """Motif de scraping cassé : le contrôle token tombe, pas seulement le volume."""
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 300)])
    _write_noise_health(
        settings,
        {"token": {"ok": False, "checked_at_iso": "2026-08-25T13:00:00Z",
                   "error": "Token public introuvable dans la page Bruitparif"}},
    )
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "KO"
    assert "token_bruitparif" in payload["failed_checks"]
    # Le poller avion, lui, reste sain : la panne est bien attribuée.
    assert "snapshot_recent" not in payload["failed_checks"]
    assert payload["bruit"]["token"]["ok"] is False


def test_daytime_noise_silence_turns_the_verdict_ko(settings):
    """Bruit muet depuis plus de 2 h en journée -> KO, même poller avion nominal."""
    now = NOON_UTC + 3600  # 13 h UTC : journée
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 3 * 3600)])
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "KO"
    assert "bruit_recent" in payload["failed_checks"]
    assert payload["bruit"]["last_event_age_s"] == 3 * 3600
    assert payload["collecte"]["snapshots_last_hour"] > 0  # le poller tourne


def test_recent_noise_keeps_the_verdict_ok(settings):
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 1800)])
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "OK"
    assert payload["bruit"]["daytime"] is True


def test_night_silence_is_not_an_alert(settings):
    """La nuit, l'absence de survol est normale : pas d'alerte à ignorer chaque nuit."""
    night = NOON_UTC + 11 * 3600 + 1800  # 23 h 30 UTC
    _healthy_poller(settings, night)
    payload = status.build_status(settings, now=night, with_pairs=False)

    assert payload["bruit"]["daytime"] is False
    assert "bruit_recent" not in payload["failed_checks"]


def test_missing_station_config_is_a_failed_check_not_a_crash(settings, tmp_path):
    """Une config de stations absente doit se voir dans status.json, pas planter."""
    settings.stations_file = str(tmp_path / "inexistant.json")
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "KO"
    assert "config_stations" in payload["failed_checks"]
    assert payload["bruit"]["station_config_error"]


# ------------------- régression : le comptage de paires ne doit pas court-circuiter
def _seed_legacy_snapshots(settings, stamps, lat, lon):
    """Écrit des snapshots dans la racine HISTORIQUE `raw/states/`."""
    from ciel_tranquille.ingest.poller import write_batch

    for ts in stamps:
        write_batch([_state_record(ts, "bb0001", lat, lon)], settings.raw_dir / "states")


def test_pairs_see_snapshots_left_in_the_legacy_root(settings):
    """Des états présents hors `landing/` ne doivent pas donner zéro paire.

    Régression : le comptage ne balayait que `landing/`. Un historique rangé sous
    `raw/states/` — ce que produisent les collectes antérieures — rendait le
    cumul de paires nul alors que les lecteurs, eux, voyaient bien les données.
    Zéro par court-circuit est le pire résultat possible : indiscernable d'une
    collecte qui n'apparie rien.
    """
    stamps = [NOON_UTC + i * 30 for i in range(120)]
    _seed_legacy_snapshots(settings, stamps, STATION.latitude + 0.002, STATION.longitude)
    _seed_noise_events(
        settings,
        "2026-08-25",
        [_noise_event(1, NOON_UTC + 300), _noise_event(2, NOON_UTC + 900)],
    )
    assert status._iter_snapshot_ts(settings, landing_only=True) == []  # rien en landing

    result = status.update_pairs(settings, NOON_UTC + 12 * 3600)
    assert result["hours_counted"] == 1
    assert result["clean_pairs_total"] == 2
    assert result["events_counted"] == 2


def test_pairs_see_compacted_hours(settings):
    """Après compaction, les heures restent comptables (fichiers horaires)."""
    from ciel_tranquille import compact

    stamps = [NOON_UTC + i * 30 for i in range(120)]
    _seed_snapshots(settings, stamps, lat=STATION.latitude + 0.002, lon=STATION.longitude)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, NOON_UTC + 300)])
    compact.compact(settings=settings, lag_h=2, now=NOON_UTC + 6 * 3600)
    assert list(settings.landing_dir.rglob("states_*.parquet")) == []  # tout compacté

    result = status.update_pairs(settings, NOON_UTC + 12 * 3600)
    assert result["hours_counted"] >= 1
    assert result["clean_pairs_total"] == 1


def test_hourly_rate_counts_landing_only(settings):
    """Le débit horaire se lit sur la landing : la compaction n'y touche jamais."""
    now = NOON_UTC + 3600
    _seed_snapshots(settings, [now - 30, now - 60])
    _seed_legacy_snapshots(settings, [now - 90], STATION.latitude, STATION.longitude)
    state = status.collection_state(settings, now)
    assert state["snapshots_last_hour"] == 2
    assert state["snapshots_in_landing"] == 2


def test_association_rate_ignores_events_of_inactive_stations(settings, tmp_path):
    """Une station retirée de la liste ne doit pas écraser le taux d'association.

    Ses partitions restent sur disque, mais aucun appariement ne sera tenté pour
    elle. La compter au dénominateur ferait passer un taux réel de 97 % pour 49 %.
    """
    import json as _json

    autre = "94290-STATION-RETIREE"
    path = tmp_path / "stations.json"
    path.write_text(
        _json.dumps(
            {
                "active": [STATION.measurement_id],
                "stations": [
                    {
                        "measurement_id": STATION.measurement_id,
                        "latitude": STATION.latitude,
                        "longitude": STATION.longitude,
                        "airport": "CDG",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings.stations_file = str(path)

    _seed_snapshots(
        settings,
        [NOON_UTC + i * 30 for i in range(120)],
        lat=STATION.latitude + 0.002,
        lon=STATION.longitude,
    )
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, NOON_UTC + 300)])
    # Partition d'une station qui n'est plus collectée.
    part = settings.noise_events_dir / f"station={autre}" / "date=2026-08-25"
    part.mkdir(parents=True, exist_ok=True)
    rows = [dict(_noise_event(900 + k, NOON_UTC + 400 + k), station=autre) for k in range(9)]
    pq.write_table(pa.Table.from_pylist(rows), part / "events.parquet")

    result = status.update_pairs(settings, NOON_UTC + 12 * 3600)
    assert result["events_counted"] == 1          # seule la station active compte
    assert result["events_stations_inactives"] == 9
    assert result["clean_pairs_total"] == 1
    assert result["association_rate"] == 1.0      # et non 0.1
