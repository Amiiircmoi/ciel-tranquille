"""Prévision du LAeq avec intervalle d'incertitude (régression quantile)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lib import model_payload, require_curated

st.set_page_config(page_title="Prévision", page_icon="🤖", layout="wide")
st.title("🤖 Prévision du niveau de bruit")

if not require_curated():
    st.stop()

payload = model_payload()
if payload is None:
    st.warning("Modèle absent. Lancez : `PYTHONPATH=src .venv/bin/python -m ciel_tranquille.ml.train`")
    st.stop()

from ciel_tranquille.ml.predict import predict_with_interval  # noqa: E402

st.caption(
    f"Modèle servi : **{payload['best_model']}** · "
    f"RMSE test {payload['metrics']['rmse']:.2f} dB · R² {payload['metrics']['r2']:.3f}"
)

with st.sidebar:
    st.header("Scénario de trafic")
    airport = st.selectbox("Aéroport", ["CDG", "ORY", "LBG"])
    hour = st.slider("Heure", 0, 23, 18)
    weekend = st.checkbox("Week-end", value=False)
    num_aircraft = st.slider("Avions dans le rayon (20 km)", 0, 30, 12)
    num_close = st.slider("Avions très proches (< 5 km)", 0, 15, 3)
    avg_alt = st.slider("Altitude moyenne (m)", 200, 4000, 1500, 50)
    min_dist = st.slider("Distance min. (km)", 0.0, 20.0, 2.0, 0.5)
    avg_vel = st.slider("Vitesse moyenne (m/s)", 60, 140, 110, 5)


def make_row(n_ac: int) -> dict:
    return {
        "hour": hour,
        "day_of_week": 5 if weekend else 2,
        "is_night": int(hour >= 22 or hour < 6),
        "is_rush_hour": int(hour in {7, 8, 9, 17, 18, 19}),
        "is_weekend": int(weekend),
        "num_aircraft": n_ac,
        "num_close_aircraft": min(num_close, n_ac),
        "avg_altitude_m": avg_alt,
        "min_altitude_m": max(avg_alt - 800, 100),
        "avg_velocity_m_s": avg_vel,
        "min_distance_km": min_dist,
        "avg_distance_km": max(min_dist + 6, 8),
        "airport": airport,
    }


point = predict_with_interval(pd.DataFrame([make_row(num_aircraft)]), payload).iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("LAeq prévu", f"{point['laeq_pred']:.1f} dB")
c2.metric("Intervalle 10–90 %", f"{point['lower']:.1f} – {point['upper']:.1f} dB")
seuil = "🔴 élevé" if point["laeq_pred"] >= 65 else ("🟠 modéré" if point["laeq_pred"] >= 55 else "🟢 faible")
c3.metric("Niveau", seuil)

# Courbe de sensibilité : LAeq prévu en fonction du nombre d'avions proches.
st.subheader("Sensibilité au trafic (toutes choses égales par ailleurs)")
sweep = pd.DataFrame([make_row(n) for n in range(0, 31)])
pred = predict_with_interval(sweep, payload)
pred["num_aircraft"] = sweep["num_aircraft"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=pred["num_aircraft"], y=pred["upper"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=pred["num_aircraft"], y=pred["lower"], fill="tonexty",
                         fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
                         name="Intervalle 10–90 %"))
fig.add_trace(go.Scatter(x=pred["num_aircraft"], y=pred["laeq_pred"],
                         line=dict(color="#2563eb", width=3), name="LAeq prévu"))
fig.update_layout(height=420, xaxis_title="Avions dans le rayon", yaxis_title="LAeq (dB)")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "L'intervalle provient de deux régressions quantiles (10 % / 90 %), plus "
    "honnête qu'un ±MAE symétrique."
)
