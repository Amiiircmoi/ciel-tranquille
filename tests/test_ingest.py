"""Tests de l'ingestion : replay déterministe, poller, parsing OpenSky."""

from __future__ import annotations

from ciel_tranquille.ingest import poller, replay
from ciel_tranquille.ingest.opensky_client import states_to_records


def test_replay_is_deterministic(settings):
    base = replay.load_base_snapshot(settings.samples_dir / "opensky_snapshot.csv")
    run1 = list(replay.replay_snapshots(3, interval_s=12, base=base))
    run2 = list(replay.replay_snapshots(3, interval_s=12, base=base))
    assert [r[0]["latitude"] for r in run1] == [r[0]["latitude"] for r in run2]


def test_replay_advances_positions(settings):
    base = replay.load_base_snapshot(settings.samples_dir / "opensky_snapshot.csv")
    snaps = list(replay.replay_snapshots(2, interval_s=60, base=base))
    # batch 0 = positions de base ; batch 1 = positions avancées -> diffèrent
    moved = [
        a["latitude"] != b["latitude"] or a["longitude"] != b["longitude"]
        for a, b in zip(snaps[0], snaps[1], strict=True)
        if a["heading_deg"] is not None and (a["velocity_m_s"] or 0) > 0
    ]
    assert any(moved)


def test_poller_replay_writes_parquet_and_metrics(settings):
    # La landing zone du poller est `$CIEL_DATA_DIR/landing/date=…` (contrat
    # d'hébergement) ; `raw/states/` reste lu pour l'historique déjà collecté.
    metrics = poller.run(3, settings=settings, sleep_between=False)
    assert len(metrics) == 3
    assert all(m.ok for m in metrics)
    files = list(settings.landing_dir.rglob("*.parquet"))
    assert len(files) == 3
    # idempotence : relancer ne crée pas de doublons de fichiers
    poller.run(3, settings=settings, sleep_between=False)
    assert len(list(settings.landing_dir.rglob("*.parquet"))) == 3


def test_states_to_records_parses_opensky_format():
    payload = {
        "time": 1_700_000_000,
        "states": [
            ["abc123", "AFR1  ", "France", 1_700_000_000, 1_700_000_000,
             2.5, 48.9, 1000.0, False, 120.0, 90.0, 0.0, None, 1100.0, "1000", False, 0],
        ],
    }
    recs = states_to_records(payload)
    assert len(recs) == 1
    r = recs[0]
    assert r["icao24"] == "abc123"
    assert r["callsign"] == "AFR1"  # trim
    assert r["latitude"] == 48.9 and r["baro_altitude_m"] == 1000.0
