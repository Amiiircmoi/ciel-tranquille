"""Baseline physique : atténuation géométrique en fonction de la distance oblique.

Un modèle d'apprentissage ne vaut que par rapport à ce qu'il faut battre. Ici la
référence n'est pas « la moyenne » ni un modèle linéaire quelconque, mais la
**loi de propagation acoustique** : une source ponctuelle rayonnant en champ
libre perd 6 dB à chaque doublement de distance, soit

    LAmax(d) = L_ref - 20 · log10(d / d_ref)

C'est la divergence géométrique sphérique. Un seul paramètre est ajusté sur les
données d'entraînement — `L_ref`, le niveau à la distance de référence — qui
absorbe la puissance acoustique moyenne de la flotte observée. **La pente reste
fixée par la physique**, elle n'est pas apprise : c'est ce qui en fait une
baseline et non un second modèle statistique.

Le rapport dira donc : « à géométrie connue, la physique seule explique X % de
la variance ; le modèle ajoute Y ». Si Y est faible, ce n'est pas un échec, c'est
un résultat — et il est déjà annoncé (`HANDOFF.md`) : à distance quasi constante
sous un couloir d'approche, la variance résiduelle de LAmax vient du **type
d'appareil**, pas de la distance.

`FittedSlopeBaseline` ajuste *aussi* la pente. Elle ne sert pas de référence mais
de **diagnostic** : si la pente estimée s'écarte franchement de -20 dB/décade,
c'est que les données contiennent autre chose que de la divergence sphérique
(absorption atmosphérique, effet de sol, biais de sélection des événements) — un
constat à écrire, pas à masquer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Distance de référence : 1 km. Choix arbitraire mais explicite — elle ne change
# que la valeur de `L_ref`, jamais les prédictions ni les métriques.
D_REF_KM = 1.0

# Plancher de distance : évite log10(0) sur un survol quasi vertical. 50 m est
# très en dessous de toute altitude de survol observée (≥ 150 m), donc inerte en
# pratique ; il n'est là que pour la robustesse numérique.
D_MIN_KM = 0.05

SPHERICAL_SLOPE_DB = -20.0


def _prepare(distance_km) -> np.ndarray:
    d = np.asarray(distance_km, dtype=float)
    return np.log10(np.maximum(d, D_MIN_KM) / D_REF_KM)


@dataclass
class SphericalSpreadingBaseline:
    """Divergence sphérique : pente imposée à -20 dB/décade, offset ajusté.

    Interface volontairement compatible scikit-learn (`fit`/`predict`) pour que
    la baseline traverse exactement le même harnais d'évaluation que les modèles
    — mêmes découpes, mêmes métriques, même code de report. Une baseline évaluée
    autrement que les modèles ne prouve rien.
    """

    l_ref_db: float | None = None

    def fit(self, distance_km, y) -> SphericalSpreadingBaseline:
        residus = np.asarray(y, dtype=float) - SPHERICAL_SLOPE_DB * _prepare(distance_km)
        self.l_ref_db = float(np.mean(residus))
        return self

    def predict(self, distance_km) -> np.ndarray:
        if self.l_ref_db is None:
            raise RuntimeError("Baseline non ajustée : appelez fit() d'abord.")
        return self.l_ref_db + SPHERICAL_SLOPE_DB * _prepare(distance_km)

    def describe(self) -> dict:
        return {
            "forme": "LAmax = L_ref - 20·log10(d/1km)",
            "l_ref_db": None if self.l_ref_db is None else round(self.l_ref_db, 3),
            "pente_db_par_decade": SPHERICAL_SLOPE_DB,
            "pente_ajustee": False,
            "n_parametres": 1,
        }


@dataclass
class FittedSlopeBaseline:
    """Même forme, mais la pente est estimée. **Diagnostic, pas référence.**"""

    l_ref_db: float | None = None
    slope_db: float | None = None

    def fit(self, distance_km, y) -> FittedSlopeBaseline:
        x = _prepare(distance_km)
        slope, intercept = np.polyfit(x, np.asarray(y, dtype=float), 1)
        self.slope_db, self.l_ref_db = float(slope), float(intercept)
        return self

    def predict(self, distance_km) -> np.ndarray:
        if self.slope_db is None:
            raise RuntimeError("Baseline non ajustée : appelez fit() d'abord.")
        return self.l_ref_db + self.slope_db * _prepare(distance_km)

    def describe(self) -> dict:
        return {
            "forme": "LAmax = L_ref + pente·log10(d/1km)",
            "l_ref_db": None if self.l_ref_db is None else round(self.l_ref_db, 3),
            "pente_db_par_decade": None if self.slope_db is None else round(self.slope_db, 3),
            "pente_ajustee": True,
            "n_parametres": 2,
            "ecart_a_la_divergence_spherique_db": (
                None if self.slope_db is None else round(self.slope_db - SPHERICAL_SLOPE_DB, 3)
            ),
        }
