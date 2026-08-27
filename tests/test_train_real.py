"""Tests du cadre d'évaluation : découpe par station, anti-fuite, intégrité."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ciel_tranquille.ml import features_real, train_real
from ciel_tranquille.ml.dataset import sha256_file


def _paires(n_par_station=80, stations=("A", "B", "C", "D")):
    """Paires synthétiques obéissant à la propagation, une géométrie par station."""
    rng = np.random.default_rng(11)
    lignes = []
    for i, st in enumerate(stations):
        slant = rng.uniform(0.5, 8.0, n_par_station)
        alt = rng.uniform(300, 3000, n_par_station)
        lignes.append(pd.DataFrame({
            "station": st,
            "airport": ["CDG", "ORY"][i % 2],
            "max_laeq": 95.0 - 20 * np.log10(slant) + rng.normal(0, 1.5, n_par_station),
            "laeq": rng.uniform(50, 70, n_par_station),
            "sel": rng.uniform(70, 95, n_par_station),
            "duration_s": rng.uniform(10, 60, n_par_station),
            "slant_km": slant,
            "horiz_km": slant * 0.8,
            "altitude_m": alt,
            "velocity_m_s": rng.uniform(60, 130, n_par_station),
            "heading_deg": rng.uniform(0, 360, n_par_station),
            "aircraft_lat": 48.9 + rng.normal(0, 0.05, n_par_station),
            "aircraft_lon": 2.4 + rng.normal(0, 0.05, n_par_station),
            "station_lat": 48.9 + i * 0.05,
            "station_lon": 2.4 + i * 0.05,
            "max_ts_unix": 1_787_600_000 + np.arange(n_par_station) * 300 + i,
            "icao24": "abc123",
            "callsign": "AFR123",
            "event_id": np.arange(n_par_station) + i * 1000,
        }))
    return pd.concat(lignes, ignore_index=True)


# ------------------------------------------------------------------- features
def test_features_are_derived_without_touching_disk():
    df = features_real.build_features(_paires())
    for col in ("log_slant", "elevation_deg", "cos_aspect", "hour_utc_num", "is_night"):
        assert col in df.columns
    assert df["elevation_deg"].between(0, 90).all()
    assert df["cos_aspect"].between(-1, 1).all()


def test_feature_derivation_is_deterministic():
    """Geler les données ne sert à rien si la préparation, elle, varie."""
    paires = _paires()
    a = features_real.build_features(paires)[features_real.NUMERIC_FEATURES]
    b = features_real.build_features(paires)[features_real.NUMERIC_FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_target_and_station_never_reach_the_feature_matrix():
    """La cible fuit par ses cousines acoustiques, la station par son identité."""
    for interdit in ("max_laeq", "laeq", "sel", "station", "station_lat", "icao24"):
        assert interdit not in features_real.FEATURES, interdit
    features_real.check_no_leakage(features_real.FEATURES)  # ne lève pas


def test_leakage_check_is_a_loud_failure():
    with pytest.raises(ValueError, match="fuite de cible"):
        features_real.check_no_leakage([*features_real.FEATURES, "sel"])


# --------------------------------------------------------------- découpe
def test_holdout_station_is_chosen_by_a_written_rule():
    """Station médiane en volume : ni la plus flatteuse, ni la plus dure."""
    df = pd.concat([
        _paires(60, ("A",)), _paires(90, ("B",)), _paires(300, ("C",)),
    ], ignore_index=True)
    assert train_real.choose_holdout(df) == "B"


def test_holdout_ignores_stations_too_small_to_test():
    df = pd.concat([_paires(5, ("mini",)), _paires(80, ("A",)), _paires(90, ("B",))],
                   ignore_index=True)
    assert train_real.choose_holdout(df) in ("A", "B")


def test_loso_never_trains_on_the_tested_station(monkeypatch):
    """Le cœur du protocole : aucune ligne de la station de test à l'entraînement."""
    df = train_real.prepare(_paires())
    vus = []
    vrai = train_real._fit_predict

    def espion(nom, estimateur, train, test, features=None):
        vus.append((set(train["station"]), set(test["station"])))
        return vrai(nom, estimateur, train, test, features)

    monkeypatch.setattr(train_real, "_fit_predict", espion)
    train_real.leave_one_station_out(df, stations_test=["A", "B"])
    assert vus, "aucun pli exécuté"
    for stations_train, stations_test in vus:
        assert stations_train.isdisjoint(stations_test)


def test_holdout_is_excluded_from_selection(tmp_path):
    """La réserve ne doit apparaître dans aucun pli de sélection."""
    df = train_real.prepare(_paires())
    reserve = "C"
    selection = df[df[features_real.GROUP_COL] != reserve]
    assert reserve not in set(selection["station"])
    loso = train_real.leave_one_station_out(selection)
    for resultat in loso.values():
        assert reserve not in resultat["par_station"]


# ------------------------------------------------------------- baseline & co
def test_baseline_is_evaluated_by_the_same_harness():
    """Baseline et modèles passent par les mêmes découpes et les mêmes métriques."""
    df = train_real.prepare(_paires())
    loso = train_real.leave_one_station_out(df, stations_test=["A", "B"])
    assert train_real.BASELINE_NAME in loso
    assert "ForetAleatoire" in loso
    for nom in (train_real.BASELINE_NAME, "ForetAleatoire"):
        for cle in ("mae_db", "rmse_db", "r2"):
            assert cle in loso[nom]["moyenne"]


def test_metrics_are_reported_in_decibels():
    y = np.array([80.0, 85.0, 90.0])
    m = train_real.metrics(y, y + 2.0)
    assert m["mae_db"] == pytest.approx(2.0)
    assert m["rmse_db"] == pytest.approx(2.0)
    assert m["n"] == 3


def test_holdout_evaluation_reports_baseline_and_models_side_by_side():
    df = train_real.prepare(_paires())
    final = train_real.evaluate_holdout(df, "D")
    assert final["station_reserve"] == "D"
    assert "D" not in final["stations_entrainement"]
    assert train_real.BASELINE_NAME in final["resultats"]
    assert final["resultats"][train_real.BASELINE_NAME]["parametres"]["n_parametres"] == 1
    num, cat, _ = train_real.active_features()
    assert set(train_real.candidate_models(num, cat)) <= set(final["resultats"])


# ---------------------------------------------------------------- intégrité
def _ecrire_instantane(tmp_path, alterer=False):
    chemin = tmp_path / "pairs_test.parquet"
    _paires(60, ("A", "B")).to_parquet(chemin, index=False)
    manifeste = {"sha256": sha256_file(chemin), "n_paires": 120, "exported_at_iso": "2026-08-27T00:00:00Z"}
    if alterer:
        manifeste["sha256"] = "0" * 64
    (tmp_path / "pairs_test.manifest.json").write_text(json.dumps(manifeste), encoding="utf-8")
    return chemin


def test_frozen_snapshot_loads_when_checksum_matches(tmp_path):
    df = train_real.load_frozen(_ecrire_instantane(tmp_path))
    assert len(df) == 120


def test_tampered_snapshot_is_refused(tmp_path):
    """Une métrique publiée doit porter sur des lignes prouvables."""
    with pytest.raises(train_real.SnapshotIntegrityError, match="Somme de contrôle"):
        train_real.load_frozen(_ecrire_instantane(tmp_path, alterer=True))


def test_snapshot_without_manifest_is_refused(tmp_path):
    chemin = tmp_path / "orphelin.parquet"
    _paires(10, ("A",)).to_parquet(chemin, index=False)
    with pytest.raises(train_real.SnapshotIntegrityError, match="Manifeste introuvable"):
        train_real.load_frozen(chemin)


def test_suffixed_copies_of_the_target_are_also_forbidden():
    """Le gel duplique certaines colonnes en `_evt` : une cible recopiée reste une cible."""
    with pytest.raises(ValueError, match="fuite de cible"):
        features_real.check_no_leakage([*features_real.FEATURES, "max_laeq_evt"])


# -------------------------------------------------- exclusion par configuration
def test_a_feature_can_be_dropped_without_touching_its_code():
    """Écarter par configuration garde la dérivation testée et le choix traçable."""
    num, cat, toutes = train_real.active_features(exclude=["is_weekend"])
    assert "is_weekend" not in toutes
    assert "is_weekend" in features_real.NUMERIC_FEATURES  # le code reste
    assert len(toutes) == len(features_real.FEATURES) - 1
    assert cat == features_real.CATEGORICAL_FEATURES


def test_excluded_feature_never_reaches_the_models(monkeypatch):
    df = train_real.prepare(_paires())
    vus = []
    vrai = train_real._fit_predict

    def espion(nom, estimateur, train, test, features=None):
        if features is not None:
            vus.append(list(features))
        return vrai(nom, estimateur, train, test, features)

    monkeypatch.setattr(train_real, "_fit_predict", espion)
    train_real.evaluate_holdout(df, "D", exclude=["is_weekend"])
    assert vus, "aucun modèle ajusté"
    for feats in vus:
        assert "is_weekend" not in feats


def test_cross_validation_reports_dispersion_not_just_the_mean():
    """Une moyenne flatteuse assise sur une forte dispersion ne se transporte pas."""
    df = train_real.prepare(_paires())
    loso = train_real.leave_one_station_out(df, stations_test=["A", "B", "C"])
    for resultat in loso.values():
        assert set(resultat["ecart_type"]) == {"mae_db", "rmse_db", "r2"}
        assert resultat["pire_station"] in resultat["par_station"]


def test_the_three_requested_families_are_present():
    num, cat, _ = train_real.active_features()
    assert list(train_real.candidate_models(num, cat)) == [
        "RegressionLineaire", "ForetAleatoire", "GradientBoosting",
    ]
