"""Test de l'évaluation réelle (features réelles construites sans erreur)."""

from __future__ import annotations

import shutil

import pandas as pd

from ciel_tranquille.ml.evaluate_real import build_real_features
from ciel_tranquille.ml.features import FEATURES, TARGET
from tests.conftest import REPO_ROOT


def test_build_real_features(settings):
    # Copie les échantillons réels requis dans le data_dir temporaire.
    for name in ("bruit_survol.csv", "opensky_snapshot.csv"):
        shutil.copy(REPO_ROOT / "data" / "samples" / name, settings.samples_dir / name)

    df = build_real_features(settings)
    assert len(df) > 100
    assert all(f in df.columns for f in FEATURES)
    assert TARGET in df.columns
    # La cible réelle reste dans une plage urbaine plausible.
    assert df[TARGET].between(30, 90).all()
    # La jointure produit bien les colonnes avions (même si majoritairement nulles).
    assert (df["num_aircraft"] >= 0).all()
    assert not df[TARGET].isna().any()
    assert isinstance(df, pd.DataFrame)
