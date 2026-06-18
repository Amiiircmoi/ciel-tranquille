"""Tests de la co-localisation événement↔aéronef (interpolation, distance, association)."""

from __future__ import annotations

import pandas as pd

from ciel_tranquille.config import STATIONS
from ciel_tranquille.transform.colocate import (
    ang_diff,
    associate_event,
    bearing_deg,
    estimate_position,
    haversine_km,
    haversine_m,
    slant_distance_km,
)

ST = STATIONS[0]  # Gonesse 48.985 / 2.447118


def _states(rows):
    return pd.DataFrame(rows)


def test_estimate_position_linear_interpolation():
    g = _states([
        {"icao24": "a", "snapshot_ts": 1000, "latitude": 49.00, "longitude": 2.45,
         "baro_altitude_m": 600, "velocity_m_s": 100, "heading_deg": 0},
        {"icao24": "a", "snapshot_ts": 1030, "latitude": 49.02, "longitude": 2.45,
         "baro_altitude_m": 660, "velocity_m_s": 100, "heading_deg": 0},
    ])
    lat, lon, alt = estimate_position(g, 1015)  # milieu
    assert abs(lat - 49.01) < 1e-6
    assert abs(alt - 630) < 1e-6


def test_estimate_position_dead_reckoning_one_sided():
    # un seul snapshot avant t -> dead-reckoning vers le nord (cap 0) à 100 m/s
    g = _states([
        {"icao24": "a", "snapshot_ts": 1000, "latitude": 49.00, "longitude": 2.45,
         "baro_altitude_m": 600, "velocity_m_s": 100, "heading_deg": 0},
    ])
    lat, lon, _ = estimate_position(g, 1010)  # +10 s -> ~1 km plus au nord
    assert lat > 49.00  # a avancé vers le nord
    assert abs(lon - 2.45) < 1e-3


def test_slant_distance_combines_horizontal_and_altitude():
    # à l'aplomb (même lat/lon) : slant = altitude
    d = slant_distance_km(ST, ST.latitude, ST.longitude, 1000.0)
    assert abs(d - 1.0) < 1e-6


def test_associate_picks_nearest_in_slant():
    t = 2000
    states = _states([
        # avion proche, à l'aplomb, 500 m
        {"icao24": "near", "snapshot_ts": t, "latitude": ST.latitude, "longitude": ST.longitude,
         "baro_altitude_m": 500, "velocity_m_s": 80, "heading_deg": 90, "callsign": "NEAR"},
        # avion lointain, ~0.1° plus loin (~11 km) et haut
        {"icao24": "far", "snapshot_ts": t, "latitude": ST.latitude + 0.1, "longitude": ST.longitude,
         "baro_altitude_m": 3000, "velocity_m_s": 200, "heading_deg": 90, "callsign": "FAR"},
    ])
    ev = pd.Series({"event_id": 7, "station": ST.measurement_id, "max_laeq": 82.0, "max_ts_unix": t})
    a = associate_event(ev, states, ST, time_tol_s=45)
    assert a.icao24 == "near"
    assert a.n_candidates == 2
    assert a.slant_km is not None and a.slant_km < 1.0


# --- primitives géométriques partagées (utilisées par les scripts de validation) ---

def test_haversine_m_is_km_times_1000():
    args = (48.8566, 2.3522, 49.0097, 2.5479)
    assert abs(haversine_m(*args) - haversine_km(*args) * 1000.0) < 1e-6


def test_bearing_deg_cardinal_directions():
    # plein nord (même longitude, latitude plus haute) -> ~0°
    assert bearing_deg(48.0, 2.0, 49.0, 2.0) == 0.0
    # plein est (même latitude, longitude plus à l'est) -> ~90°
    assert abs(bearing_deg(48.0, 2.0, 48.0, 3.0) - 90.0) < 0.5


def test_ang_diff_wraps_around_north():
    assert ang_diff(350.0, 10.0) == 20.0
    assert ang_diff(10.0, 350.0) == 20.0
    assert ang_diff(0.0, 180.0) == 180.0


def test_associate_no_aircraft_in_window():
    states = _states([
        {"icao24": "x", "snapshot_ts": 1000, "latitude": ST.latitude, "longitude": ST.longitude,
         "baro_altitude_m": 500, "velocity_m_s": 80, "heading_deg": 90},
    ])
    ev = pd.Series({"event_id": 9, "station": ST.measurement_id, "max_laeq": 70.0, "max_ts_unix": 5000})
    a = associate_event(ev, states, ST, time_tol_s=45)
    assert a.icao24 is None and a.n_candidates == 0 and a.slant_km is None
