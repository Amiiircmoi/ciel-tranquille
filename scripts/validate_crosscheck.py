"""Cross-check indépendant de la jointure bruit↔trafic (2ᵉ méthode d'association).

Validation **complémentaire** au GATE strict `validate_join.py` : là où le GATE
ne retient une paire « propre » que si l'aéronef le plus proche est *overhead*
ET sans 2ᵉ avion ambigu (anti-ambiguïté « unique dominant »), ce cross-check
associe simplement **l'aéronef le plus proche en distance oblique** (via
`colocate.build_pairs`). Deux méthodes indépendantes qui doivent concorder.

Construit la jointure sur les **seuls snapshots forward du jour** (exclut le
replay 2025-09-*) et reporte :
  (a) corrélation distance oblique ↔ LAmax (poolée et **par station**) ;
  (b) confirmation azimut/élévation via /data (direction d'arrivée du son) ;
  (c) taux d'association.

Usage : PYTHONPATH=src .venv/bin/python scripts/validate_crosscheck.py
"""

from __future__ import annotations

import math

import duckdb
import httpx
import pandas as pd
from scipy import stats

from ciel_tranquille.config import STATIONS, get_settings
from ciel_tranquille.ingest.bruitparif_client import BruitparifClient
from ciel_tranquille.transform.colocate import ang_diff, bearing_deg, build_pairs

STATIONS_BY_ID = {s.measurement_id: s for s in STATIONS}
# Fenêtre de recouvrement = bornes des snapshots forward du jour (renseignées au run).
TODAY = "2026-06-17"


def load_data(con):
    states = con.execute(
        f"SELECT icao24, callsign, latitude, longitude, baro_altitude_m, "
        f"velocity_m_s, heading_deg, snapshot_ts "
        f"FROM read_parquet('data/raw/states/date={TODAY}/*.parquet') "
        f"WHERE latitude IS NOT NULL AND on_ground = FALSE"
    ).fetch_df()
    lo, hi = int(states.snapshot_ts.min()), int(states.snapshot_ts.max())
    events = con.execute(
        f"SELECT event_id, station, max_laeq, max_ts_unix, latitude, longitude "
        f"FROM noise_events "
        f"WHERE max_ts_unix BETWEEN {lo} AND {hi}"
    ).fetch_df()
    return events, states, lo, hi


def azimuth_check(pairs: pd.DataFrame, n: int = 20) -> dict:
    """Pour un échantillon d'événements bien associés, compare l'azimut/élévation
    mesuré (/data au pic) au relèvement/élévation géométrique vers l'aéronef."""
    client = BruitparifClient()
    tok = client.token()
    cli = httpx.Client(timeout=30)
    sample = pairs[(pairs.slant_km.notna()) & (pairs.slant_km <= 12)].nlargest(n, "max_laeq")
    rows = []
    for _, p in sample.iterrows():
        st = STATIONS_BY_ID[p.station]
        from datetime import datetime, timezone, timedelta
        t = datetime.fromtimestamp(p.max_ts_unix, tz=timezone.utc)
        a = (t - timedelta(seconds=8)).strftime("%Y-%m-%dT%H:%M:%S")
        b = (t + timedelta(seconds=8)).strftime("%Y-%m-%dT%H:%M:%S")
        url = f"{client.settings.bruitparif_api_url}/data/{p.station}/{a}/{b}"
        try:
            d = cli.get(url, params={"token": tok}).json()
        except Exception:
            continue
        d = [x for x in d if x.get("valid") and x.get("leq") is not None]
        if not d:
            continue
        peak = max(d, key=lambda x: x["leq"])  # échantillon le plus fort = le survol
        brg = bearing_deg(st.latitude, st.longitude, p.aircraft_lat, p.aircraft_lon)
        horiz_m = p.horiz_km * 1000.0
        elev_geom = math.degrees(math.atan2(max(p.altitude_m, 0.0), max(horiz_m, 1.0)))
        rows.append({
            "event_id": p.event_id, "max_laeq": p.max_laeq, "slant_km": p.slant_km,
            "az_measured": peak["azimut"], "az_bearing": round(brg, 1),
            "az_diff": round(ang_diff(peak["azimut"], brg), 1),
            "elev_measured": peak["elevation"], "elev_geom": round(elev_geom, 1),
        })
    df = pd.DataFrame(rows)
    return {"n": len(df), "table": df}


def main():
    settings = get_settings()
    con = duckdb.connect(str(settings.duckdb_path))
    events, states, lo, hi = load_data(con)
    print(f"Fenêtre recouvrement: snapshots {states.snapshot_ts.nunique()} "
          f"({pd.to_datetime(lo, unit='s')} -> {pd.to_datetime(hi, unit='s')} UTC)")
    print(f"Événements bruit dans la fenêtre: {len(events)}")

    pairs = build_pairs(events, states, STATIONS_BY_ID, time_tol_s=45.0)
    pairs.to_csv("outputs/phase1b_pairs.csv", index=False)

    # (c) taux d'association
    with_candidate = pairs[pairs.n_candidates > 0]
    assoc = pairs[pairs.slant_km.notna()]
    print("\n--- (c) ASSOCIATION ---")
    print(f"couverture temporelle (>=1 avion a +-45s): {len(with_candidate)}/{len(pairs)} "
          f"({100*len(with_candidate)/len(pairs):.0f}%)")
    for thr in (5, 8, 12, 20):
        n = (assoc.slant_km <= thr).sum()
        print(f"  slant<= {thr:2d} km : {n}/{len(pairs)} ({100*n/len(pairs):.0f}%)")
    print(f"distance oblique min (km): med={assoc.slant_km.median():.2f} "
          f"p25={assoc.slant_km.quantile(.25):.2f} p75={assoc.slant_km.quantile(.75):.2f}")

    # (a) corrélation distance oblique <-> LAmax
    print("\n--- (a) CORRELATION distance oblique <-> LAmax ---")
    for label, sub in [("tous associés", assoc), ("slant<=12km", assoc[assoc.slant_km <= 12])]:
        if len(sub) >= 5:
            pr = stats.pearsonr(sub.slant_km, sub.max_laeq)
            sp = stats.spearmanr(sub.slant_km, sub.max_laeq)
            print(f"  {label} (n={len(sub)}): Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.1e}) "
                  f"| Spearman rho={sp.statistic:+.3f} (p={sp.pvalue:.1e})")
    print("  par station (slant<=12km, Spearman):")
    for stid, sub in assoc[assoc.slant_km <= 12].groupby("station"):
        if len(sub) >= 5:
            sp = stats.spearmanr(sub.slant_km, sub.max_laeq)
            print(f"    {stid:34s} n={len(sub):3d} rho={sp.statistic:+.3f}")

    # (b) azimut/élévation
    print("\n--- (b) AZIMUT / ELEVATION (/data au pic) ---")
    az = azimuth_check(pairs, n=20)
    if az["n"]:
        t = az["table"]
        print(f"échantillon n={az['n']} ; azimut |mesuré - relèvement| : "
              f"médiane={t.az_diff.median():.0f}° ; <30°: {(t.az_diff<30).sum()}/{len(t)} ; "
              f"<45°: {(t.az_diff<45).sum()}/{len(t)}")
        print(f"élévation mesurée médiane={t.elev_measured.median():.0f}° "
              f"(géométrique médiane={t.elev_geom.median():.0f}°)")
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
