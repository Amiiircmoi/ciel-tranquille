"""Tests du garde-fou de budget quotidien et du heartbeat du poller."""

from __future__ import annotations

import pytest

from ciel_tranquille.ingest import poller
from ciel_tranquille.monitoring.heartbeat import Heartbeat, read_heartbeat, write_heartbeat

# Repère temporel : 2026-08-25T12:00:00Z — il reste exactement 12 h avant la
# réinitialisation quotidienne UTC du budget OpenSky.
NOON_UTC = 1_787_659_200.0
HALF_DAY_S = 43_200.0


def test_seconds_until_utc_midnight():
    assert poller.seconds_until_utc_midnight(NOON_UTC) == HALF_DAY_S
    assert poller.seconds_until_utc_midnight(NOON_UTC - HALF_DAY_S) == 86_400.0


def test_nominal_cadence_is_untouched_with_healthy_budget():
    """Budget confortable -> exactement 30 s. Le garde-fou n'accélère jamais."""
    interval = poller.credit_guarded_interval(
        nominal_s=30, credits_remaining=3500, credit_floor=200, max_interval_s=300, now=NOON_UTC
    )
    assert interval == 30


def test_cadence_untouched_when_budget_exactly_covers_day():
    # 12 h restantes à 30 s = 1440 appels ; 1640 - plancher 200 = 1440 utilisables.
    interval = poller.credit_guarded_interval(
        nominal_s=30, credits_remaining=1640, credit_floor=200, max_interval_s=300, now=NOON_UTC
    )
    assert interval == 30


def test_cadence_stretches_when_budget_runs_short():
    """Solde insuffisant -> on étire pour tenir jusqu'à minuit, sans dépasser le plafond."""
    # 12 h restantes, 400 appels utilisables -> 43200/400 = 108 s.
    interval = poller.credit_guarded_interval(
        nominal_s=30, credits_remaining=600, credit_floor=200, max_interval_s=300, now=NOON_UTC
    )
    assert interval == 108.0
    assert interval > 30  # ralenti, jamais accéléré


def test_cadence_capped_by_max_interval():
    interval = poller.credit_guarded_interval(
        nominal_s=30, credits_remaining=210, credit_floor=200, max_interval_s=300, now=NOON_UTC
    )
    assert interval == 300


def test_cadence_at_or_below_floor_uses_max_interval():
    assert (
        poller.credit_guarded_interval(
            nominal_s=30, credits_remaining=200, credit_floor=200, max_interval_s=300, now=NOON_UTC
        )
        == 300
    )


def test_unknown_credits_keep_nominal_cadence():
    """En replay/offline, aucun solde n'est connu : pas de ralentissement arbitraire."""
    assert (
        poller.credit_guarded_interval(
            nominal_s=30, credits_remaining=None, credit_floor=200, max_interval_s=300, now=NOON_UTC
        )
        == 30
    )


def test_heartbeat_roundtrip_is_atomic_and_secret_free(settings):
    beat = Heartbeat(
        updated_at_unix=NOON_UTC,
        snapshot_ts=int(NOON_UTC),
        rows=421,
        batches_total=17,
        credits_remaining=3120,
        poll_interval_s=30,
        mode="live",
        pid=4242,
    )
    write_heartbeat(settings.heartbeat_path, beat)
    loaded = read_heartbeat(settings.heartbeat_path)

    assert loaded["rows"] == 421
    assert loaded["credits_remaining"] == 3120
    assert loaded["updated_at_iso"].startswith("2026-08-25T12:00")
    # Aucun temporaire résiduel : l'écriture passe par tmp + os.replace.
    assert not list(settings.status_dir.glob(".*tmp"))
    # Aucun secret ne transite par le heartbeat.
    raw = settings.heartbeat_path.read_text(encoding="utf-8").lower()
    assert "token" not in raw and "secret" not in raw


def test_read_heartbeat_missing_returns_none(settings):
    assert read_heartbeat(settings.status_dir / "absent.json") is None


def test_poller_writes_heartbeat_on_each_batch(settings):
    poller.run(2, settings=settings, sleep_between=False)
    beat = read_heartbeat(settings.heartbeat_path)
    assert beat is not None
    assert beat["batches_total"] == 2
    assert beat["mode"] == "replay"
    assert beat["rows"] > 0


# ----------------------------------- contrôle de démarrage des stations actives
def test_poller_refuses_to_start_on_invalid_station_config(settings, tmp_path):
    """Config de stations absente -> refus de démarrer, pas six jours à vide."""
    from ciel_tranquille.config import StationConfigError

    settings.stations_file = str(tmp_path / "inexistant.json")
    settings.ingest_mode = "live"
    with pytest.raises(StationConfigError):
        poller.preflight_stations(settings)


def test_poller_refuses_to_start_on_station_outside_bbox(settings, tmp_path):
    """Une station hors bbox ne verra jamais l'aéronef qui la survole."""
    import json

    from ciel_tranquille.ingest.stations import StationValidationError

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
    with pytest.raises(StationValidationError, match="hors bbox"):
        poller.preflight_stations(settings)
