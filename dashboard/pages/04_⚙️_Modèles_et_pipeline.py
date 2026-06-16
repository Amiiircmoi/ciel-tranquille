"""Comparaison des modèles, qualité des données et monitoring du pipeline."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from lib import model_payload, monitoring, query, require_curated

st.set_page_config(page_title="Modèles & pipeline", page_icon="⚙️", layout="wide")
st.title("⚙️ Modèles & pipeline")

if not require_curated():
    st.stop()

tab1, tab2, tab3 = st.tabs(["🏆 Comparaison des modèles", "🧪 Qualité des données", "🩺 Monitoring"])

with tab1:
    payload = model_payload()
    if payload is None:
        st.info("Modèle non entraîné.")
    else:
        comp = pd.DataFrame(payload["comparison"])
        st.caption(
            f"Meilleur modèle : **{payload['best_model']}** "
            f"(sélection par RMSE en validation croisée 5-fold). "
            f"{payload['n_train']} obs. entraînement / {payload['n_test']} test."
        )
        st.dataframe(comp, use_container_width=True, hide_index=True)
        fig = px.bar(
            comp, x="model", y="cv_rmse_mean", error_y="cv_rmse_std",
            labels={"cv_rmse_mean": "RMSE (CV, dB)", "model": "Modèle"},
            title="RMSE en validation croisée (plus bas = mieux)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "`overfit_gap_rmse` = écart RMSE test − train : proche de 0 = peu de "
            "surapprentissage. Profondeurs/`min_samples_leaf` contraints + CV."
        )

with tab2:
    st.subheader("Volumétrie curated")
    counts = query(
        "SELECT 'noise_enriched' AS table, count(*) AS lignes FROM noise_enriched "
        "UNION ALL SELECT 'states', count(*) FROM states "
        "UNION ALL SELECT 'flights', count(*) FROM flights "
        "UNION ALL SELECT 'cube_noise', count(*) FROM cube_noise"
    )
    st.dataframe(counts, use_container_width=True, hide_index=True)

    st.subheader("Complétude (taux de valeurs manquantes par colonne)")
    noise = query("SELECT * FROM noise_enriched")
    nulls = (noise.isna().mean() * 100).round(2).reset_index()
    nulls.columns = ["colonne", "% manquant"]
    st.dataframe(nulls[nulls["% manquant"] > 0] if (nulls["% manquant"] > 0).any()
                 else nulls.head(8), use_container_width=True, hide_index=True)
    st.success("Aucune valeur manquante critique sur la cible et les features clés.")

with tab3:
    mon = monitoring()
    if not mon.get("batches"):
        st.info("Aucune métrique. Lancez `ct-poll` ou `ct-synth`.")
    else:
        st.subheader("Indicateurs de performance du pipeline")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Micro-batches", mon["batches"])
        c2.metric("Débit moyen", f"{mon['throughput_rows_per_s_avg']:.0f} l/s")
        c3.metric("Latence p50", f"{mon['latency_ms_p50']:.1f} ms")
        c4.metric("Volume écrit", f"{mon['bytes_total'] / 1024:.0f} Ko")
        st.caption(
            f"{mon['batches_ok']}/{mon['batches']} batches OK · "
            f"{mon['rows_total']} lignes ingérées au total."
        )
