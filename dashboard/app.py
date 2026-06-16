"""Ciel Tranquille — dashboard (accueil + santé du pipeline)."""

from __future__ import annotations

import streamlit as st
from lib import SETTINGS, load_table, monitoring, require_curated

st.set_page_config(page_title="Ciel Tranquille", page_icon="✈️", layout="wide")

st.title("✈️ Ciel Tranquille — Bruit aérien urbain")
st.caption(
    "Vitrine analytique : ingestion micro-batch OpenSky → DuckDB → features → "
    "modèle ML → ce tableau de bord."
)

with st.sidebar:
    st.header("Configuration")
    st.metric("Mode d'ingestion", SETTINGS.ingest_mode)
    st.caption(
        f"Bounding box : {SETTINGS.bbox_lat_min}–{SETTINGS.bbox_lat_max} N, "
        f"{SETTINGS.bbox_lon_min}–{SETTINGS.bbox_lon_max} E"
    )
    st.caption(f"Cadence micro-batch : {SETTINGS.poll_interval_s}s")
    st.divider()
    st.caption("Pages : carte, historique, prévision, modèles & pipeline →")

if not require_curated():
    st.stop()

noise = load_table("noise_enriched")
states = load_table("states")
mon = monitoring()

st.subheader("Vue d'ensemble")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mesures de bruit", f"{len(noise):,}".replace(",", " "))
c2.metric("Stations", noise["station_id"].nunique() if "station_id" in noise else 0)
c3.metric("États avions ingérés", f"{len(states):,}".replace(",", " "))
c4.metric("LAeq moyen", f"{noise['laeq_db'].mean():.1f} dB" if len(noise) else "—")

st.subheader("🩺 Santé du pipeline (monitoring)")
if mon.get("batches"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Micro-batches", mon["batches"])
    m2.metric("Débit moyen", f"{mon['throughput_rows_per_s_avg']:.0f} lignes/s")
    m3.metric("Latence p50", f"{mon['latency_ms_p50']:.1f} ms")
    m4.metric("Latence max", f"{mon['latency_ms_max']:.0f} ms")
    st.caption(
        f"{mon['batches_ok']}/{mon['batches']} batches OK · "
        f"{mon['rows_total']:,} lignes · {mon['bytes_total'] / 1024:.0f} Ko écrits".replace(
            ",", " "
        )
    )
else:
    st.info("Aucune métrique de pipeline encore. Lancez `ct-poll` ou `ct-synth`.")

st.divider()
st.markdown(
    """
    **Garde-fou d'honnêteté** : l'ingestion est un **polling micro-batch**
    (≈1 req/10 s côté OpenSky gratuit), pas un flux *push* temps réel. Le vrai
    temps réel (broker type Kafka/Kinesis) est une limite assumée et un axe
    d'amélioration documenté.
    """
)
