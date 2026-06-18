"""Validation stricte (quality gate) de la jointure bruit↔trafic.

PAS d'entraînement de modèle ici : on valide UNIQUEMENT que la jointure
spatio-temporelle co-localisée est physiquement saine, sur les seuls snapshots
**forward du jour** (date=2026-06-17 ; les partitions 2025-09-* du socle v1 sont
exclues).

Étapes :
  0. CHECK FUSEAU (avant tout) : max_ts (ISO …Z, UTC) vs snapshot_ts OpenSky
     (epoch Unix UTC). 3 paires loguées à la main → confirmer même seconde UTC.
  1. Jointure événement→avion : pour chaque survol, interpoler la position de
     chaque aéronef à l'instant `max_ts` (horizon ≤15 s : interpolation entre
     snapshots encadrants, sinon dead-reckoning velocity+true_track).
  2. Anti-ambiguïté : ne retenir comme paire « propre » qu'un **candidat unique
     dominant** (2ᵉ avion ≥ MARGIN_KM plus loin en distance oblique).
  3. Corrélation distance oblique ↔ LAmax (attendu : nettement négatif).
  4. Recoupement azimut : pour un sous-échantillon, comparer l'azimut Bruitparif
     (/data) au relèvement station→avion calculé.

Sortie : rapport texte + blob JSON (--json) pour la fiche markdown.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys

import duckdb
import numpy as np
import pandas as pd

from ciel_tranquille.transform.colocate import ang_diff, bearing_deg, haversine_m

EARTH_R_M = 6_371_000.0
FORWARD_DATE = "2026-06-17"

# --- paramètres de la jointure (documentés dans le rapport) ---
INTERP_MAX_S = 15.0       # horizon d'interpolation/dead-reckoning autour de max_ts
BRACKET_MAX_GAP_S = 35.0  # écart max entre 2 snapshots encadrants pour interpoler
CAND_HORIZ_KM = 12.0      # un candidat doit être à ≤ ce rayon horizontal de la station
MAX_SLANT_KM = 10.0       # le plus proche doit être à ≤ cette distance oblique (survol audible)
MARGIN_KM = 3.0           # 2e candidat ≥ MARGIN_KM plus loin → « unique dominant »


def move(lat, lon, brg_deg, dist_m):
    """Avance d'une distance le long d'un cap (équirectangulaire, ok à ≤15 s)."""
    brg = math.radians(brg_deg)
    dn = dist_m * math.cos(brg)
    de = dist_m * math.sin(brg)
    dlat = dn / EARTH_R_M
    dlon = de / (EARTH_R_M * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def load_data():
    con = duckdb.connect()
    states = con.execute(
        f"SELECT icao24, callsign, longitude, latitude, baro_altitude_m, "
        f"velocity_m_s, heading_deg, on_ground, snapshot_ts "
        f"FROM read_parquet('data/raw/states/date={FORWARD_DATE}/*.parquet')"
    ).df()
    noise = con.execute(
        f"SELECT station, airport, latitude AS st_lat, longitude AS st_lon, event_id, "
        f"max_ts_unix, max_ts_iso, max_laeq, laeq, sel, duration_s "
        f"FROM read_parquet('data/raw/noise_events/station=*/date={FORWARD_DATE}/*.parquet', "
        f"hive_partitioning=true)"
    ).df()
    return states, noise


# --------------------------------------------------------------- 0. check fuseau
def timezone_check(states, noise, n=3):
    win_a, win_b = states.snapshot_ts.min(), states.snapshot_ts.max()
    snap_times = np.sort(states.snapshot_ts.unique())
    inwin = noise[(noise.max_ts_unix >= win_a) & (noise.max_ts_unix <= win_b)].copy()
    inwin = inwin.sort_values("max_ts_unix")
    lines = []
    picks = inwin.iloc[:: max(1, len(inwin) // n)].head(n)
    for _, ev in picks.iterrows():
        t = float(ev.max_ts_unix)
        nearest = snap_times[np.argmin(np.abs(snap_times - t))]
        back = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        snap_utc = dt.datetime.fromtimestamp(int(nearest), dt.timezone.utc).strftime("%H:%M:%S")
        lines.append({
            "event_id": int(ev.event_id),
            "max_ts_iso_brut": ev.max_ts_iso,
            "max_ts_unix": t,
            "max_ts_unix->UTC": back,
            "snapshot_le_plus_proche_unix": int(nearest),
            "snapshot->UTC": snap_utc,
            "ecart_s": round(abs(nearest - t), 1),
        })
    return {"window_unix": [int(win_a), int(win_b)],
            "window_utc": [dt.datetime.fromtimestamp(int(win_a), dt.timezone.utc).strftime("%H:%M:%S"),
                           dt.datetime.fromtimestamp(int(win_b), dt.timezone.utc).strftime("%H:%M:%S")],
            "pairs": lines}


# ------------------------------------------------- 1+2. jointure + anti-ambiguïté
def interp_aircraft_at(obs, t):
    """Position (lat,lon,alt) d'un aéronef à l'instant t. obs = df trié par snapshot_ts.

    Retourne (lat, lon, alt, method) ou None si non plaçable à ≤ INTERP_MAX_S.
    """
    ts = obs.snapshot_ts.values
    before = obs[obs.snapshot_ts <= t]
    after = obs[obs.snapshot_ts >= t]
    if len(before) and len(after):
        b = before.iloc[-1]
        a = after.iloc[0]
        gap = a.snapshot_ts - b.snapshot_ts
        if gap == 0:
            return float(b.latitude), float(b.longitude), float(b.baro_altitude_m), "snap"
        if gap <= BRACKET_MAX_GAP_S and (t - b.snapshot_ts) <= INTERP_MAX_S and (a.snapshot_ts - t) <= INTERP_MAX_S:
            f = (t - b.snapshot_ts) / gap
            lat = b.latitude + f * (a.latitude - b.latitude)
            lon = b.longitude + f * (a.longitude - b.longitude)
            alt = b.baro_altitude_m + f * (a.baro_altitude_m - b.baro_altitude_m)
            return float(lat), float(lon), float(alt), "bracket"
    # dead-reckoning depuis l'obs la plus proche (|dt| ≤ INTERP_MAX_S)
    nearest = obs.iloc[np.argmin(np.abs(ts - t))]
    dtt = t - nearest.snapshot_ts
    if abs(dtt) > INTERP_MAX_S:
        return None
    lat, lon, alt = float(nearest.latitude), float(nearest.longitude), float(nearest.baro_altitude_m)
    if np.isfinite(nearest.velocity_m_s) and np.isfinite(nearest.heading_deg) and nearest.velocity_m_s > 0:
        lat, lon = move(lat, lon, float(nearest.heading_deg), float(nearest.velocity_m_s) * dtt)
        return lat, lon, alt, "deadreckon"
    return lat, lon, alt, "snap"


def join_events(states, noise):
    win_a, win_b = states.snapshot_ts.min(), states.snapshot_ts.max()
    # candidats : aéronefs en vol, position+altitude valides
    air = states[(~states.on_ground) & states.latitude.notna() & states.longitude.notna()
                 & states.baro_altitude_m.notna()].copy()
    air = air.sort_values("snapshot_ts")
    by_icao = {k: v for k, v in air.groupby("icao24")}
    snap_times = np.sort(states.snapshot_ts.unique())

    inwin = noise[(noise.max_ts_unix >= win_a) & (noise.max_ts_unix <= win_b)].copy()
    rows = []
    for _, ev in inwin.iterrows():
        t = float(ev.max_ts_unix)
        # avions ayant une obs à ≤ INTERP_MAX_S de t (sinon pas plaçables)
        cands = []
        nearby_snaps = snap_times[np.abs(snap_times - t) <= BRACKET_MAX_GAP_S]
        if len(nearby_snaps) == 0:
            rows.append({**ev.to_dict(), "status": "no_snapshot", "n_cand": 0})
            continue
        local = air[(air.snapshot_ts >= t - BRACKET_MAX_GAP_S) & (air.snapshot_ts <= t + BRACKET_MAX_GAP_S)]
        for icao, obs in local.groupby("icao24"):
            placed = interp_aircraft_at(obs.sort_values("snapshot_ts"), t)
            if placed is None:
                continue
            lat, lon, alt, method = placed
            horiz = haversine_m(ev.st_lat, ev.st_lon, lat, lon)
            if horiz / 1000.0 > CAND_HORIZ_KM:
                continue
            slant = math.sqrt(horiz ** 2 + (alt or 0.0) ** 2)
            cands.append({
                "icao24": icao, "callsign": obs.callsign.iloc[-1],
                "horiz_km": horiz / 1000.0, "alt_m": alt, "slant_km": slant / 1000.0,
                "bearing_deg": bearing_deg(ev.st_lat, ev.st_lon, lat, lon),
                "elev_deg": math.degrees(math.atan2(alt or 0.0, max(horiz, 1.0))),
                "method": method,
            })
        cands.sort(key=lambda c: c["slant_km"])
        base = {**ev.to_dict(), "n_cand": len(cands)}
        if not cands:
            rows.append({**base, "status": "no_candidate"})
            continue
        c0 = cands[0]
        second = cands[1]["slant_km"] if len(cands) > 1 else math.inf
        is_overhead = c0["slant_km"] <= MAX_SLANT_KM
        is_dominant = (second - c0["slant_km"]) >= MARGIN_KM
        status = "clean" if (is_overhead and is_dominant) else ("ambiguous" if is_overhead else "far")
        rows.append({**base, "status": status,
                     "m_icao24": c0["icao24"], "m_callsign": c0["callsign"],
                     "m_horiz_km": c0["horiz_km"], "m_alt_m": c0["alt_m"],
                     "m_slant_km": c0["slant_km"], "m_bearing_deg": c0["bearing_deg"],
                     "m_elev_deg": c0["elev_deg"], "m_method": c0["method"],
                     "second_slant_km": second})
    return pd.DataFrame(rows), (int(win_a), int(win_b))


# ------------------------------------------------------- 4. recoupement azimut
def azimuth_crosscheck(clean_df, max_checks=10):
    """Compare l'azimut Bruitparif (/data au pic) au relèvement station→avion."""
    try:
        import httpx
        sys.path.insert(0, "src")
        from ciel_tranquille.ingest.bruitparif_client import BruitparifClient
        from ciel_tranquille.config import get_settings
        s = get_settings()
        client = BruitparifClient(s)
        token = client.token()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"client Bruitparif indisponible: {exc}", "checks": []}

    checks = []
    sample = clean_df.sort_values("m_slant_km").head(max_checks)
    with httpx.Client(timeout=30.0, headers={"User-Agent": "ciel-tranquille/validation"}) as h:
        for _, r in sample.iterrows():
            t = float(r.max_ts_unix)
            frm = dt.datetime.fromtimestamp(t - 20, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            to = dt.datetime.fromtimestamp(t + 20, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            url = f"{s.bruitparif_api_url}/data/{r.station}/{frm}/{to}"
            try:
                resp = h.get(url, params={"token": token})
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                checks.append({"event_id": int(r.event_id), "ok": False, "reason": str(exc)[:80]})
                continue
            series = data if isinstance(data, list) else data.get("data") or data.get("values") or []
            peak = _peak_sample(series)
            if peak is None:
                checks.append({"event_id": int(r.event_id), "ok": False,
                               "reason": "azimut absent", "raw_keys": _keys(series)})
                continue
            az_bp = peak.get("azimuth", peak.get("azimut"))
            el_bp = peak.get("elevation", peak.get("elevation_angle"))
            entry = {"event_id": int(r.event_id), "ok": True,
                     "az_bruitparif": az_bp, "az_calcule": round(r.m_bearing_deg, 1),
                     "el_bruitparif": el_bp, "el_calcule": round(r.m_elev_deg, 1)}
            if az_bp is not None:
                entry["delta_az_deg"] = round(ang_diff(float(az_bp), r.m_bearing_deg), 1)
            checks.append(entry)
    oks = [c for c in checks if c.get("ok") and "delta_az_deg" in c]
    return {"ok": True, "n": len(oks),
            "median_delta_az": round(float(np.median([c["delta_az_deg"] for c in oks])), 1) if oks else None,
            "coherent_pct": round(100 * np.mean([c["delta_az_deg"] <= 45 for c in oks]), 0) if oks else None,
            "checks": checks}


def _keys(series):
    if isinstance(series, list) and series and isinstance(series[0], dict):
        return list(series[0].keys())
    return str(type(series))


def _peak_sample(series):
    if not isinstance(series, list) or not series or not isinstance(series[0], dict):
        return None
    def leq(x):
        return x.get("leq", x.get("laeq", x.get("value", -999))) or -999
    return max(series, key=leq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--no-azimuth", action="store_true")
    args = ap.parse_args()

    states, noise = load_data()
    print(f"states forward {FORWARD_DATE}: {len(states)} lignes, "
          f"{states.snapshot_ts.nunique()} snapshots | noise {FORWARD_DATE}: {len(noise)} évts\n")

    # 0. fuseau
    tz = timezone_check(states, noise)
    print("=== 0. CHECK FUSEAU (max_ts UTC vs snapshot_ts epoch UTC) ===")
    print(f"fenêtre forward UTC : {tz['window_utc'][0]} -> {tz['window_utc'][1]}")
    for p in tz["pairs"]:
        print(f"  évt {p['event_id']}: ISO {p['max_ts_iso_brut']} | epoch {p['max_ts_unix']:.1f} "
              f"-> {p['max_ts_unix->UTC']}Z | snapshot {p['snapshot->UTC']}Z | écart {p['ecart_s']}s")

    # 1+2. jointure
    df, win = join_events(states, noise)
    counts = df.status.value_counts().to_dict()
    denom = len(df)
    clean = df[df.status == "clean"].copy()
    n_clean = len(clean)
    assoc_rate = round(100 * n_clean / denom, 1) if denom else 0.0
    matched = df[df.status.isin(["clean", "ambiguous", "far"])]
    print(f"\n=== 1+2. JOINTURE (évts dans la fenêtre = {denom}) ===")
    print(f"statuts : {counts}")
    print(f"avec ≥1 candidat overhead : {len(df[df.status.isin(['clean','ambiguous'])])}")
    print(f">>> paires PROPRES (unique dominant) : {n_clean}  | taux d'association = {assoc_rate}%")

    # 3. corrélation
    corr = {}
    if n_clean >= 3:
        for col, label in [("m_slant_km", "slant"), ("m_horiz_km", "horiz"), ("m_alt_m", "alt")]:
            pear = float(clean[col].corr(clean.max_laeq, method="pearson"))
            spear = float(clean[col].corr(clean.max_laeq, method="spearman"))
            corr[label] = {"pearson": round(pear, 3), "spearman": round(spear, 3)}
        print(f"\n=== 3. CORRÉLATION (paires propres, n={n_clean}) ===")
        print(f"  distance oblique ↔ LAmax : Pearson {corr['slant']['pearson']} | "
              f"Spearman {corr['slant']['spearman']}  (attendu : nettement NÉGATIF)")
        print(f"  altitude ↔ LAmax         : Pearson {corr['alt']['pearson']}")
        print(f"  LAmax : min {clean.max_laeq.min():.1f} / médiane {clean.max_laeq.median():.1f} / "
              f"max {clean.max_laeq.max():.1f} dB ; slant médian {clean.m_slant_km.median():.2f} km")
    else:
        print(f"\n=== 3. CORRÉLATION : trop peu de paires propres (n={n_clean}) ===")

    # sensibilité (taux selon les seuils)
    sens = []
    for mk in (2.0, 3.0, 5.0):
        for ms in (8.0, 10.0, 15.0):
            c = df[(df.status.isin(["clean", "ambiguous", "far"]))].copy()
            cc = c[(c.m_slant_km <= ms) & ((c.second_slant_km - c.m_slant_km) >= mk)]
            sens.append({"margin_km": mk, "max_slant_km": ms, "n_clean": int(len(cc)),
                         "rate_pct": round(100 * len(cc) / denom, 1) if denom else 0.0})

    # 4. azimut
    az = {"ok": False, "reason": "désactivé"}
    if not args.no_azimuth and n_clean >= 1:
        print("\n=== 4. RECOUPEMENT AZIMUT (/data, sous-échantillon) ===")
        az = azimuth_crosscheck(clean)
        if az.get("ok"):
            print(f"  azimut : delta médian {az['median_delta_az']}° | cohérents (≤45°) {az['coherent_pct']}% "
                  f"sur n={az['n']}")
            for c in az["checks"][:5]:
                if c.get("ok") and "delta_az_deg" in c:
                    print(f"    évt {c['event_id']}: BP {c['az_bruitparif']}° vs calc {c['az_calcule']}° "
                          f"-> Δ {c['delta_az_deg']}°")
                else:
                    print(f"    évt {c['event_id']}: KO ({c.get('reason')}) {c.get('raw_keys','')}")
        else:
            print(f"  indisponible : {az.get('reason')}")

    result = {
        "forward_date": FORWARD_DATE,
        "window_unix": win, "window_utc": tz["window_utc"],
        "params": {"INTERP_MAX_S": INTERP_MAX_S, "CAND_HORIZ_KM": CAND_HORIZ_KM,
                   "MAX_SLANT_KM": MAX_SLANT_KM, "MARGIN_KM": MARGIN_KM},
        "timezone_check": tz,
        "n_events_in_window": denom, "status_counts": counts,
        "n_clean": n_clean, "assoc_rate_pct": assoc_rate,
        "corr": corr, "sensitivity": sens, "azimuth": az,
        "verdict": _verdict(assoc_rate, corr, n_clean),
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[json -> {args.json}]")
    print(f"\n=== VERDICT GATE-mécanisme : {result['verdict']['decision']} ===")
    print(f"  {result['verdict']['rationale']}")


def _verdict(assoc_rate, corr, n_clean):
    neg = corr.get("slant", {}).get("spearman")
    cond_rate = assoc_rate >= 40.0
    cond_corr = neg is not None and neg <= -0.3
    cond_n = n_clean >= 20
    if cond_rate and cond_corr and cond_n:
        d = "PIVOT VALIDÉ"
        r = (f"taux d'association {assoc_rate}% (≥40%), corrélation distance↔LAmax "
             f"Spearman {neg} (nettement négative), {n_clean} paires propres (≥20).")
    else:
        d = "À DIAGNOSTIQUER / PROLONGER"
        manques = []
        if not cond_rate:
            manques.append(f"taux {assoc_rate}% < 40%")
        if not cond_corr:
            manques.append(f"corrélation {neg} non franchement négative")
        if not cond_n:
            manques.append(f"n={n_clean} < 20 paires propres")
        r = "à revoir : " + "; ".join(manques) + " (cf. fuseau / fenêtre / cadence / seuils)."
    return {"decision": d, "rationale": r}


if __name__ == "__main__":
    main()
