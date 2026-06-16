"""Tests du nettoyage (tests unitaires)."""

from __future__ import annotations

import pandas as pd

from ciel_tranquille.transform.clean import (
    LAEQ_MAX_DB,
    clean_flights,
    clean_noise,
    clean_states,
    quality_report,
)


def test_clean_noise_normalises_and_bounds():
    raw = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S2"],
            "station_name": ["a", "a", "b"],
            "timestamp_iso": ["2025-09-01T03:00:00", "2025-09-01T03:00:00", "2025-09-01T14:00:00"],
            "LAeq_dB": [55.0, 55.0, 999.0],  # doublon + aberration
            "Lmax_dB": [60.0, 60.0, 1000.0],
            "latitude": [48.9, 48.9, 48.7],
            "longitude": [2.5, 2.5, 2.3],
            "airport": ["CDG", "CDG", "ORY"],
        }
    )
    out = clean_noise(raw)
    assert "laeq_db" in out.columns and "lmax_db" in out.columns
    assert len(out) == 2  # doublon supprimé
    assert out["laeq_db"].max() <= LAEQ_MAX_DB  # aberration bornée
    # features temporelles
    night_row = out[out["station_id"] == "S1"].iloc[0]
    assert night_row["is_night"] == 1  # 03:00
    day_row = out[out["station_id"] == "S2"].iloc[0]
    assert day_row["is_night"] == 0  # 14:00


def test_clean_states_drops_ground_and_dedup():
    raw = pd.DataFrame(
        {
            "icao24": ["a", "a", "b"],
            "time_position_unix": [1_758_542_819, 1_758_542_819, 1_758_542_819],
            "longitude": [2.4, 2.4, 2.5],
            "latitude": [48.9, 48.9, 49.0],
            "baro_altitude_m": [1000.0, 1000.0, -50.0],
            "velocity_m_s": [100.0, 100.0, 90.0],
            "heading_deg": [90.0, 90.0, 180.0],
            "on_ground": [False, False, True],
            "snapshot_ts": [1_758_542_819, 1_758_542_819, 1_758_542_819],
        }
    )
    out = clean_states(raw)
    assert len(out) == 1  # doublon (a) fusionné, (b) au sol supprimé
    assert (out["baro_altitude_m"] >= 0).all()


def test_clean_flights_duration():
    raw = pd.DataFrame(
        {
            "flight_id": ["F1"],
            "callsign": ["AFR1"],
            "departure_icao": ["LFPG"],
            "arrival_icao": ["LEMD"],
            "first_seen_iso": ["2025-09-01T06:00:00"],
            "last_seen_iso": ["2025-09-01T08:00:00"],
            "avg_altitude_m": [9000.0],
            "avg_speed_knots": [420.0],
            "approx_distance_km": [1000.0],
        }
    )
    out = clean_flights(raw)
    assert abs(out.iloc[0]["duration_min"] - 120.0) < 1e-6


def test_quality_report_keys():
    df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
    rep = quality_report(df, "x")
    assert rep["source"] == "x" and rep["rows"] == 2 and 0 <= rep["null_ratio"] <= 1
