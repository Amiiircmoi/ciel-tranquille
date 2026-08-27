"""Tests du rapport de supervision `status/status.json`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
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


def _write_noise_health(settings, payload):
    from ciel_tranquille.monitoring.heartbeat import write_json_atomic

    write_json_atomic(settings.noise_health_path, payload)


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
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 300)])
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
def _healthy_poller(settings, now, collecteur_vivant=True):
    """Poller avion sain, et collecteur de bruit vivant par défaut.

    Les deux sources ont des contrôles distincts : pour isoler l'effet de l'une,
    l'autre doit être saine, sinon le verdict échoue pour une raison étrangère
    à ce que le test veut prouver.
    """
    _seed_snapshots(settings, [now - i * 30 for i in range(1, 20)])
    write_heartbeat(
        settings.heartbeat_path,
        Heartbeat(now - 20, now - 30, 40, 100, 3200, 30, "live", 1),
    )
    if collecteur_vivant:
        _write_noise_health(
            settings,
            {
                "last_run_unix": now - 600,
                "token": {"ok": True, "checked_at_iso": "2026-08-25T13:00:00Z", "error": None},
            },
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
    """Bruit muet au-delà du budget en journée -> KO, même poller avion nominal.

    Le budget vaut cadence du collecteur + latence de publication (5 h par
    défaut) : en deçà, le silence est normal et alerter serait un faux positif.
    """
    now = NOON_UTC + 3600  # 13 h UTC : journée
    _healthy_poller(settings, now)
    silence = settings.noise_silence_budget_s + 3600
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - silence)])
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert payload["verdict"] == "KO"
    assert "bruit_recent" in payload["failed_checks"]
    assert payload["bruit"]["last_event_age_s"] == silence
    assert payload["collecte"]["snapshots_last_hour"] > 0  # le poller tourne


def test_silence_within_the_budget_is_not_an_alert(settings):
    """3 h de silence avec un collecteur à 3 h : normal, pas une panne."""
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 3 * 3600)])
    payload = status.build_status(settings, now=now, with_pairs=False)
    assert "bruit_recent" not in payload["failed_checks"]


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


# ------------------ budget de silence coherent avec la cadence du collecteur
def test_noise_silence_budget_accounts_for_collector_cadence(settings):
    """Un seuil plus serré que la cadence du collecteur alerterait à chaque cycle.

    Régression observée en production : collecteur toutes les 3 h + ~1 h de
    latence de publication = âge du dernier événement montant à 4 h. Le seuil
    de 2 h basculait KO avant chaque passage — une fausse alerte toutes les 3 h,
    donc un canal de notification qu'on apprend à ignorer.
    """
    settings.noise_interval_s = 10800      # 3 h
    settings.noise_publication_lag_s = 3600  # 1 h
    settings.noise_max_silence_s = 7200    # 2 h, trop serré
    assert settings.noise_silence_budget_s >= 10800 + 3600
    assert settings.noise_silence_budget_s == 18000


def test_stricter_silence_threshold_is_honoured_when_cadence_allows(settings):
    """Collecteur horaire : un seuil serré redevient légitime."""
    settings.noise_interval_s = 3600
    settings.noise_publication_lag_s = 3600
    settings.noise_max_silence_s = 14400
    assert settings.noise_silence_budget_s == 14400


def test_dead_noise_collector_is_its_own_check(settings):
    """Signal direct : le collecteur ne tourne plus, indépendamment des données."""
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 600)])
    _write_noise_health(
        settings,
        {"last_run_unix": now - 5 * 3600, "token": {"ok": True, "checked_at_iso": "", "error": None}},
    )
    payload = status.build_status(settings, now=now, with_pairs=False)

    assert "collecteur_bruit_vivant" in payload["failed_checks"]
    # Les données, elles, sont fraîches : les deux signaux sont bien distincts.
    assert "bruit_recent" not in payload["failed_checks"]


def test_live_noise_collector_passes(settings):
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, now - 600)])
    _write_noise_health(
        settings,
        {"last_run_unix": now - 600, "token": {"ok": True, "checked_at_iso": "", "error": None}},
    )
    payload = status.build_status(settings, now=now, with_pairs=False)
    assert payload["verdict"] == "OK"


# ------------------------------- pas de falaise a minuit UTC
def test_noise_volume_check_uses_a_rolling_window(settings):
    """À 00:06 UTC, le compteur du JOUR vaut zéro alors que tout va bien.

    Régression observée en production : alerte urgente à 00:06, rétablie à 02:51
    quand le collecteur a écrit ses premiers événements du nouveau jour. Une
    fausse alerte quotidienne à minuit vide le canal de notification de son sens.
    """
    minuit_passe = NOON_UTC + 12 * 3600 + 360  # 2026-08-26 00:06 UTC
    _healthy_poller(settings, minuit_passe)
    # Événements de la veille au soir : rien encore pour le nouveau jour.
    _seed_noise_events(settings, "2026-08-25", [_noise_event(1, minuit_passe - 2 * 3600)])

    payload = status.build_status(settings, now=minuit_passe, with_pairs=False)

    assert payload["bruit"]["events_today"] == 0       # jour calendaire : vide
    assert payload["bruit"]["events_last_24h"] == 1    # fenêtre glissante : vivant
    assert "collecte_bruit" not in payload["failed_checks"]


def test_noise_volume_check_still_catches_a_real_stop(settings):
    """La fenêtre glissante ne doit pas rendre le contrôle complaisant."""
    now = NOON_UTC + 3600
    _healthy_poller(settings, now)
    # Dernier événement il y a plus de 24 h : la source est réellement morte.
    _seed_noise_events(settings, "2026-08-23", [_noise_event(1, now - 30 * 3600)])

    payload = status.build_status(settings, now=now, with_pairs=False)
    assert payload["bruit"]["events_last_24h"] == 0
    assert "collecte_bruit" in payload["failed_checks"]


def test_compacted_files_outside_the_hour_are_skipped(settings, monkeypatch):
    """Le nom du fichier suffit à écarter une heure compactée hors sujet.

    Sans ce tri, chaque heure calculée relisait tous les fichiers compactés :
    un coût qui croît avec la durée de collecte alors que le travail utile est
    constant. Le test compte les lectures, pas la durée — une mesure de temps
    serait instable en intégration continue.
    """
    from ciel_tranquille import compact

    stamps = [NOON_UTC + i * 30 for i in range(120)]
    _seed_snapshots(settings, stamps, lat=STATION.latitude + 0.002, lon=STATION.longitude)
    # Une seconde heure, très éloignée : elle n'a rien à voir avec la fenêtre visée.
    lointain = NOON_UTC + 10 * 3600
    _seed_snapshots(
        settings, [lointain + i * 30 for i in range(120)],
        lat=STATION.latitude + 0.002, lon=STATION.longitude,
    )
    compact.compact(settings=settings, lag_h=2, now=lointain + 6 * 3600)

    lus = []
    vrai = pd.read_parquet
    monkeypatch.setattr(
        status.pd, "read_parquet",
        lambda path, *a, **k: (lus.append(Path(path).name), vrai(path, *a, **k))[1],
    )
    frame = status._load_states_hour(settings, NOON_UTC, NOON_UTC + 3600)

    assert not frame.empty
    horaires = [n for n in lus if "T" in n]
    assert len(horaires) == 1, f"heures compactées relues à tort : {horaires}"
    assert horaires[0].startswith("states_")


def test_compacted_file_span_is_read_from_its_name():
    from ciel_tranquille.compact import hour_file_span

    debut, fin = hour_file_span("states_20260825T12.parquet")
    assert time.strftime("%Y-%m-%dT%HZ", time.gmtime(debut)) == "2026-08-25T12Z"
    assert fin - debut == 3600
    # Un snapshot brut n'est pas un fichier horaire : on ne doit pas le confondre.
    assert hour_file_span("states_1787659200.parquet") is None
