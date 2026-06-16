"""Tests de la jointure spatio-temporelle."""

from __future__ import annotations

import pandas as pd

from ciel_tranquille.transform.join import AIRCRAFT_FEATURES, haversine_km, spatio_temporal_join


def test_haversine_known_distance():
    # Paris (48.8566,2.3522) -> Roissy CDG (49.0097,2.5479) ~ 23 km
    d = haversine_km(48.8566, 2.3522, 49.0097, 2.5479)
    assert 20 < float(d) < 26


def test_join_finds_aligned_neighbour():
    t = pd.Timestamp("2025-09-01T12:00:00", tz="UTC")
    noise = pd.DataFrame({"timestamp": [t], "latitude": [48.90], "longitude": [2.50]})
    states = pd.DataFrame(
        {
            # avion à ~1 km, dans la fenêtre ; + un avion hors rayon
            "timestamp": [t, t],
            "latitude": [48.905, 40.0],
            "longitude": [2.50, 2.0],
            "baro_altitude_m": [800.0, 1000.0],
            "velocity_m_s": [100.0, 120.0],
        }
    )
    out = spatio_temporal_join(noise, states, time_window_min=5, radius_km=20)
    assert out.iloc[0]["num_aircraft"] == 1  # seul le proche compte
    assert out.iloc[0]["min_distance_km"] < 2
    assert out.iloc[0]["avg_altitude_m"] == 800.0


def test_join_respects_time_window():
    t = pd.Timestamp("2025-09-01T12:00:00", tz="UTC")
    noise = pd.DataFrame({"timestamp": [t], "latitude": [48.90], "longitude": [2.50]})
    states = pd.DataFrame(
        {
            "timestamp": [t + pd.Timedelta(minutes=30)],  # hors fenêtre ±5 min
            "latitude": [48.905],
            "longitude": [2.50],
            "baro_altitude_m": [800.0],
            "velocity_m_s": [100.0],
        }
    )
    out = spatio_temporal_join(noise, states, time_window_min=5, radius_km=20)
    assert out.iloc[0]["num_aircraft"] == 0


def test_join_empty_states_returns_zero_features():
    t = pd.Timestamp("2025-09-01T12:00:00", tz="UTC")
    noise = pd.DataFrame({"timestamp": [t], "latitude": [48.9], "longitude": [2.5]})
    out = spatio_temporal_join(noise, pd.DataFrame())
    assert all(f in out.columns for f in AIRCRAFT_FEATURES)
    assert (out[AIRCRAFT_FEATURES].to_numpy() == 0).all()


def test_join_mixed_datetime_resolution():
    # Régression : bruit en [us], états en [s] -> doivent tout de même matcher.
    t = pd.Timestamp("2025-09-01T12:00:00", tz="UTC")
    noise = pd.DataFrame({"timestamp": pd.Series([t]).dt.as_unit("us"),
                          "latitude": [48.90], "longitude": [2.50]})
    states = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([int(t.timestamp())], unit="s", utc=True),
            "latitude": [48.905], "longitude": [2.50],
            "baro_altitude_m": [800.0], "velocity_m_s": [100.0],
        }
    )
    out = spatio_temporal_join(noise, states)
    assert out.iloc[0]["num_aircraft"] == 1
