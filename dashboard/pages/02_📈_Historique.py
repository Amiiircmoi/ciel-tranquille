"""Historique temporel du bruit + exploration du cube OLAP."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from lib import load_table, query, require_curated

st.set_page_config(page_title="Historique", page_icon="📈", layout="wide")
st.title("📈 Historique du bruit par station")

if not require_curated():
    st.stop()

noise = load_table("noise_enriched")
if noise.empty:
    st.info("Pas de données.")
    st.stop()

stations = sorted(noise["station_id"].unique().tolist())
sel = st.multiselect("Stations", stations, default=stations[:3])
df = noise[noise["station_id"].isin(sel)].sort_values("timestamp") if sel else noise

st.subheader("Série temporelle du LAeq")
fig = px.line(
    df, x="timestamp", y="laeq_db", color="station_id",
    labels={"laeq_db": "LAeq (dB)", "timestamp": "Heure"}, height=420,
)
fig.add_hline(y=55, line_dash="dot", annotation_text="Seuil OMS jour ~55 dB")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Profil horaire moyen")
    prof = df.groupby("hour", as_index=False)["laeq_db"].mean()
    st.plotly_chart(
        px.bar(prof, x="hour", y="laeq_db", labels={"laeq_db": "LAeq moyen (dB)", "hour": "Heure"}),
        use_container_width=True,
    )
with c2:
    st.subheader("Jour vs nuit")
    dn = df.assign(periode=df["is_night"].map({1: "nuit", 0: "jour"})).groupby(
        "periode", as_index=False
    )["laeq_db"].mean()
    st.plotly_chart(
        px.bar(dn, x="periode", y="laeq_db", color="periode",
               labels={"laeq_db": "LAeq moyen (dB)"}),
        use_container_width=True,
    )

st.subheader("🧊 Cube OLAP — bruit par aéroport × période × type de jour")
st.caption(
    "Agrégat multidimensionnel (`GROUP BY CUBE`) : `niveau_agregation` indique "
    "le degré de regroupement (0 = détail, plus élevé = totaux partiels)."
)
cube = query(
    "SELECT airport, periode, type_jour, n_mesures, laeq_moyen_db, laeq_max_db, avions_moyen "
    "FROM cube_noise ORDER BY niveau_agregation, laeq_moyen_db DESC"
)
st.dataframe(cube, use_container_width=True, hide_index=True)
