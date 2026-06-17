"""Tests du collecteur bruit Bruitparif + garde-fous crédits OpenSky (offline)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from ciel_tranquille.config import STATIONS
from ciel_tranquille.ingest import noise_collector as nc
from ciel_tranquille.ingest.bruitparif_client import _TOKEN_RE
from ciel_tranquille.ingest.opensky_client import OpenSkyClient


# ----------------------------------------------------------------- Bruitparif
def test_token_regex_extracts_public_token():
    html = "<hardcoded-parameters params=\"merge: true, token: 'AbC123xyz', userAuth: false\">"
    m = _TOKEN_RE.search(html)
    assert m and m.group(1) == "AbC123xyz"


def test_parse_iso_handles_bruitparif_formats():
    assert nc._parse_iso("2026-06-02T17:23:46.400000+0000") is not None
    assert nc._parse_iso("2026-06-02T17:23:46Z") is not None
    assert nc._parse_iso(None) is None
    assert nc._parse_iso("pas une date") is None


def test_normalize_event_target_and_join_key():
    ev = {
        "category": "air", "id": 42,
        "start": "2026-06-02T17:23:21.400000+0000",
        "end": "2026-06-02T17:24:00.500000+0000",
        "max_ts": "2026-06-02T17:23:46.400000+0000",
        "laeq": 69.79, "max_laeq": 77.37, "sel": 85.81, "nrj_laeq": 9530186, "valid": True,
    }
    rec = nc.normalize_event(ev, STATIONS[0], date(2026, 6, 2))
    assert rec is not None
    assert rec["max_laeq"] == 77.37  # cible
    assert rec["max_ts_unix"] and rec["max_ts_unix"] > 1_700_000_000  # clé de jointure
    assert abs(rec["duration_s"] - 39.1) < 0.01
    assert rec["station"] == STATIONS[0].measurement_id


def test_normalize_event_rejects_incomplete():
    assert nc.normalize_event({"category": "air", "id": 1, "max_laeq": 70.0}, STATIONS[0], date(2026, 6, 2)) is None
    assert nc.normalize_event({"category": "air", "id": 1, "max_ts": "2026-06-02T00:00:00Z"}, STATIONS[0], date(2026, 6, 2)) is None


class _StubClient:
    """Client Bruitparif factice : renvoie 2 survols par fenêtre (mêmes ids ->
    la déduplication par `id` doit les fusionner sur les multiples fenêtres)."""

    def __init__(self):
        self.calls = []

    def fetch_window_air_events(self, station, start, end):
        self.calls.append((station, start, end))
        return [
            {"category": "air", "id": 100 + i, "max_ts": f"2026-06-16T08:0{i}:00Z",
             "start": "2026-06-16T08:00:00Z", "end": "2026-06-16T08:01:00Z", "max_laeq": 75.0 + i}
            for i in range(2)
        ]

    def close(self):
        pass


def test_collect_windows_dedup_and_load_duckdb(settings):
    stub = _StubClient()
    report = nc.collect(
        settings=settings, days_back=1, end_day=date(2026, 6, 16),
        stations=(STATIONS[0],), client=stub, request_pause_s=0,
        collected_at=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
    )
    assert len(stub.calls) == 24  # 24 fenêtres horaires sur une journée complète
    assert report["written_total"] == 2  # 48 bruts -> 2 uniques après dédup par id
    assert report["station_days_ok"] == 1
    files = list(settings.noise_events_dir.rglob("*.parquet"))
    assert len(files) == 1
    n = nc.load_to_duckdb(settings)
    assert n == 2


# ------------------------------------------------------------------- OpenSky
def test_capture_rate_limit_reads_header():
    client = OpenSkyClient.__new__(OpenSkyClient)  # sans I/O
    client.last_rate_limit_remaining = None
    resp = httpx.Response(200, headers={"x-rate-limit-remaining": "3997"})
    client._capture_rate_limit(resp)
    assert client.last_rate_limit_remaining == 3997


def test_retry_after_seconds_parses_and_bounds():
    assert OpenSkyClient._retry_after_seconds(httpx.Response(429, headers={"retry-after": "5"})) == 5.0
    # borné à 120 s
    assert OpenSkyClient._retry_after_seconds(httpx.Response(429, headers={"retry-after": "9999"})) == 120.0
    # absent -> défaut
    assert OpenSkyClient._retry_after_seconds(httpx.Response(429)) == 10.0
