"""Tests du générateur synthétique + bout-en-bout (synth -> curated -> ML)."""

from __future__ import annotations

import shutil

import pandas as pd

from ciel_tranquille.ml import synth
from ciel_tranquille.storage.build_curated import build
from tests.conftest import REPO_ROOT


def test_synth_is_reproducible(settings):
    s1 = synth.generate(seed=7, start="2025-09-01", days=2, settings=settings)
    df1 = pd.read_csv(settings.samples_dir / "bruit_synth.csv")
    s2 = synth.generate(seed=7, start="2025-09-01", days=2, settings=settings)
    df2 = pd.read_csv(settings.samples_dir / "bruit_synth.csv")
    assert s1["noise_rows"] == s2["noise_rows"]
    pd.testing.assert_frame_equal(df1, df2)


def test_synth_laeq_in_realistic_range(settings):
    synth.generate(seed=42, start="2025-09-01", days=2, settings=settings)
    df = pd.read_csv(settings.samples_dir / "bruit_synth.csv")
    assert df["LAeq_dB"].between(30, 95).all()
    assert 45 <= df["LAeq_dB"].mean() <= 70  # plage urbaine plausible


def test_end_to_end_synth_to_model(settings):
    # flights nécessaire au build : on copie l'échantillon réel.
    shutil.copy(
        REPO_ROOT / "data" / "samples" / "flights_history.csv",
        settings.samples_dir / "flights_history.csv",
    )
    synth.generate(seed=42, start="2025-09-01", days=3, settings=settings)
    report = build(settings=settings, noise_csv="bruit_synth.csv")
    assert report["tables"]["noise_enriched"] > 100
    assert report["tables"]["cube_noise"] > 0

    # Au moins une partie des mesures doit avoir capté un avion (jointure utile).
    from ciel_tranquille.storage.duck import connect

    with connect(settings, read_only=True) as con:
        matched = con.execute(
            "SELECT count(*) FROM noise_enriched WHERE num_aircraft > 0"
        ).fetchone()[0]
    assert matched > 0
