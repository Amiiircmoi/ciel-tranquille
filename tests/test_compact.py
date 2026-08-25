"""Tests de la compaction horaire : regroupement, sûreté, idempotence."""

from __future__ import annotations

import pyarrow.parquet as pq

from ciel_tranquille import compact
from ciel_tranquille.ingest.poller import write_batch

# 2026-08-25T12:00:00Z
NOON_UTC = 1_787_659_200


def _record(ts: int, icao: str = "abc123") -> dict:
    return {
        "icao24": icao,
        "callsign": "TEST123",
        "origin_country": "France",
        "time_position_unix": ts,
        "longitude": 2.4,
        "latitude": 48.9,
        "baro_altitude_m": 900.0,
        "velocity_m_s": 100.0,
        "heading_deg": 90.0,
        "squawk": None,
        "last_contact_unix": ts,
        "on_ground": False,
        "snapshot_ts": ts,
    }


def _seed_hour(settings, start_ts: int, n: int = 6, per_snapshot: int = 3) -> list[int]:
    """Écrit `n` snapshots espacés de 30 s dans la landing zone."""
    stamps = [start_ts + i * 30 for i in range(n)]
    for ts in stamps:
        write_batch([_record(ts, f"ac{j:04x}") for j in range(per_snapshot)], settings.landing_dir)
    return stamps


def test_scan_landing_groups_by_utc_hour(settings):
    _seed_hour(settings, NOON_UTC, n=4)          # heure 12
    _seed_hour(settings, NOON_UTC + 3600, n=2)   # heure 13
    groups = compact.scan_landing(settings.landing_dir)
    assert set(groups) == {("2026-08-25", "12"), ("2026-08-25", "13")}
    assert len(groups[("2026-08-25", "12")]) == 4


def test_is_hour_closed_respects_lag():
    # Heure 12 (finie à 13 h). À 15 h, elle est close depuis 2 h.
    assert compact.is_hour_closed("2026-08-25", "12", lag_h=2, now=NOON_UTC + 3 * 3600)
    assert not compact.is_hour_closed("2026-08-25", "12", lag_h=2, now=NOON_UTC + 2 * 3600 - 1)
    # L'heure en cours n'est jamais close.
    assert not compact.is_hour_closed("2026-08-25", "12", lag_h=2, now=NOON_UTC + 600)


def test_compact_merges_hour_and_removes_sources(settings):
    _seed_hour(settings, NOON_UTC, n=6, per_snapshot=3)
    report = compact.compact(settings=settings, lag_h=2, now=NOON_UTC + 4 * 3600)

    assert report.hours_compacted == 1
    assert report.files_merged == 6
    assert report.rows == 18
    assert report.errors == []
    # Un seul fichier en sortie, plus aucun snapshot unitaire en landing.
    out = list(settings.compacted_states_dir.rglob("*.parquet"))
    assert len(out) == 1
    assert out[0].name == "states_20260825T12.parquet"
    assert list(settings.landing_dir.rglob("states_*.parquet")) == []
    assert pq.read_metadata(out[0]).num_rows == 18


def test_compact_leaves_open_hours_alone(settings):
    _seed_hour(settings, NOON_UTC, n=3)                 # close
    _seed_hour(settings, NOON_UTC + 3 * 3600, n=3)      # en cours
    report = compact.compact(settings=settings, lag_h=2, now=NOON_UTC + 3 * 3600 + 600)

    assert report.hours_compacted == 1
    assert report.hours_skipped_open == 1
    # Les snapshots de l'heure ouverte restent intacts pour le poller.
    remaining = list(settings.landing_dir.rglob("states_*.parquet"))
    assert len(remaining) == 3


def test_compact_is_idempotent_and_dedups(settings):
    _seed_hour(settings, NOON_UTC, n=4, per_snapshot=2)
    first = compact.compact(settings=settings, lag_h=2, now=NOON_UTC + 4 * 3600)
    assert first.rows == 8

    # Rejeu : les mêmes snapshots réapparaissent (reprise après incident).
    _seed_hour(settings, NOON_UTC, n=4, per_snapshot=2)
    second = compact.compact(settings=settings, lag_h=2, now=NOON_UTC + 4 * 3600)

    out = list(settings.compacted_states_dir.rglob("*.parquet"))
    assert len(out) == 1
    # Déduplication sur (icao24, snapshot_ts) : toujours 8 lignes, pas 16.
    assert pq.read_metadata(out[0]).num_rows == 8
    assert second.rows == 8


def test_compact_keep_sources_option(settings):
    _seed_hour(settings, NOON_UTC, n=3)
    report = compact.compact(
        settings=settings, lag_h=2, delete_sources=False, now=NOON_UTC + 4 * 3600
    )
    assert report.files_removed == 0
    assert len(list(settings.landing_dir.rglob("states_*.parquet"))) == 3


def test_compact_on_empty_landing_is_a_noop(settings):
    report = compact.compact(settings=settings, lag_h=2, now=NOON_UTC)
    assert report.hours_compacted == 0
    assert report.errors == []


def test_compacted_files_remain_readable_by_pipeline(settings):
    """La compaction ne doit pas rendre l'historique invisible aux lecteurs."""
    from ciel_tranquille.storage.duck import has_raw_data, states_globs

    _seed_hour(settings, NOON_UTC, n=3)
    assert has_raw_data(settings)
    compact.compact(settings=settings, lag_h=2, now=NOON_UTC + 4 * 3600)
    assert has_raw_data(settings)
    assert any("states_hourly" in g for g in states_globs(settings))


def test_include_open_hours_compacts_the_current_hour(settings):
    """Fin de collecte, poller arrêté : l'heure en cours doit pouvoir être fermée."""
    _seed_hour(settings, NOON_UTC, n=4, per_snapshot=2)
    during = NOON_UTC + 600  # 10 min après le début de l'heure : elle n'est pas close

    skipped = compact.compact(settings=settings, lag_h=2, now=during)
    assert skipped.hours_compacted == 0 and skipped.hours_skipped_open == 1

    forced = compact.compact(settings=settings, lag_h=2, now=during, include_open_hours=True)
    assert forced.hours_compacted == 1
    assert forced.rows == 8
    assert list(settings.landing_dir.rglob("states_*.parquet")) == []
