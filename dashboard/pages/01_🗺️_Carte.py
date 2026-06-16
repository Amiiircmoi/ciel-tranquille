"""Carte des stations de bruit + dernières positions d'aéronefs ingérées."""

from __future__ import annotations

import pydeck as pdk
import streamlit as st
from lib import color_for_db, query, require_curated

st.set_page_config(page_title="Carte", page_icon="🗺️", layout="wide")
st.title("🗺️ Carte des zones bruyantes & du trafic")

if not require_curated():
    st.stop()

# Niveau de bruit moyen par station (agrégat spatial — pas de domicile ciblé).
stations = query(
    """
    SELECT station_id, station_name, airport, latitude, longitude,
           round(avg(laeq_db), 1) AS laeq_moyen,
           round(max(laeq_db), 1) AS laeq_max,
           count(*) AS n
    FROM noise_enriched
    GROUP BY station_id, station_name, airport, latitude, longitude
    """
)
if stations.empty:
    st.info("Pas de données de stations.")
    st.stop()

stations["color"] = stations["laeq_moyen"].apply(color_for_db)
stations["radius"] = 250 + (stations["laeq_moyen"] - 40).clip(lower=0) * 60

view = pdk.ViewState(
    latitude=float(stations["latitude"].mean()),
    longitude=float(stations["longitude"].mean()),
    zoom=9,
    pitch=35,
)

station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=stations,
    get_position=["longitude", "latitude"],
    get_fill_color="color",
    get_radius="radius",
    pickable=True,
    opacity=0.7,
)

layers = [station_layer]

# Dernier snapshot d'avions ingérés (quasi temps réel).
last_states = query(
    """
    SELECT longitude, latitude, baro_altitude_m, callsign
    FROM states
    WHERE snapshot_ts = (SELECT max(snapshot_ts) FROM states)
      AND latitude IS NOT NULL AND longitude IS NOT NULL
    """
)
if not last_states.empty:
    aircraft_layer = pdk.Layer(
        "ScatterplotLayer",
        data=last_states,
        get_position=["longitude", "latitude"],
        get_fill_color=[37, 99, 235, 200],
        get_radius=400,
        pickable=True,
    )
    layers.append(aircraft_layer)

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"text": "{station_name}\nLAeq moyen {laeq_moyen} dB\n{callsign}"},
        map_style=None,
    )
)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Stations (agrégat)")
    st.dataframe(
        stations[["station_name", "airport", "laeq_moyen", "laeq_max", "n"]]
        .sort_values("laeq_moyen", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
with col2:
    st.subheader("Légende")
    st.markdown(
        "🟢 < 50 dB · 🟡 50–60 dB · 🟠 60–68 dB · 🔴 > 68 dB\n\n"
        "🔵 derniers aéronefs ingérés"
    )
    st.metric("Avions au dernier snapshot", len(last_states))
