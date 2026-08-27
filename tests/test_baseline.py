"""Tests de la baseline physique : la pente reste la physique, pas un paramètre."""

from __future__ import annotations

import numpy as np
import pytest

from ciel_tranquille.ml.baseline import (
    SPHERICAL_SLOPE_DB,
    FittedSlopeBaseline,
    SphericalSpreadingBaseline,
)


def _survols(l_ref=95.0, pente=SPHERICAL_SLOPE_DB, n=400, bruit=0.0):
    """Jeu synthétique obéissant exactement à la loi de propagation."""
    rng = np.random.default_rng(7)
    d = rng.uniform(0.3, 9.0, n)
    y = l_ref + pente * np.log10(d) + rng.normal(0, bruit, n)
    return d, y


def test_recovers_the_reference_level():
    d, y = _survols(l_ref=95.0)
    modele = SphericalSpreadingBaseline().fit(d, y)
    assert modele.l_ref_db == pytest.approx(95.0, abs=1e-6)
    assert modele.predict(d) == pytest.approx(y, abs=1e-6)


def test_slope_is_physics_not_a_parameter():
    """Même sur des données à -12 dB/décade, la pente imposée ne bouge pas.

    C'est ce qui distingue une baseline d'un second modèle : elle ne s'adapte
    pas pour gagner, elle dit ce que prédit la divergence sphérique.
    """
    d, y = _survols(l_ref=95.0, pente=-12.0)
    modele = SphericalSpreadingBaseline().fit(d, y)
    assert modele.describe()["pente_db_par_decade"] == -20.0
    assert modele.describe()["pente_ajustee"] is False
    assert modele.describe()["n_parametres"] == 1


def test_six_db_per_doubling():
    """Vérification directe de la loi : -6 dB par doublement de distance."""
    modele = SphericalSpreadingBaseline().fit(*_survols())
    proche, loin = modele.predict([1.0])[0], modele.predict([2.0])[0]
    assert proche - loin == pytest.approx(6.02, abs=0.01)


def test_diagnostic_baseline_measures_the_gap():
    d, y = _survols(l_ref=90.0, pente=-14.0)
    diag = FittedSlopeBaseline().fit(d, y)
    assert diag.slope_db == pytest.approx(-14.0, abs=1e-6)
    assert diag.describe()["ecart_a_la_divergence_spherique_db"] == pytest.approx(6.0, abs=1e-3)


def test_zero_distance_does_not_explode():
    """Un survol quasi vertical ne doit pas produire log10(0)."""
    modele = SphericalSpreadingBaseline().fit(*_survols())
    assert np.isfinite(modele.predict([0.0, 1e-9])).all()


def test_predict_before_fit_is_a_loud_failure():
    with pytest.raises(RuntimeError):
        SphericalSpreadingBaseline().predict([1.0])
    with pytest.raises(RuntimeError):
        FittedSlopeBaseline().predict([1.0])
