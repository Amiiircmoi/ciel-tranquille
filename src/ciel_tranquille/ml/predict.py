"""Inférence : prédiction du LAeq avec **intervalle d'incertitude**.

L'intervalle vient de deux modèles de régression quantile (10 % / 90 %) entraînés
en même temps que le modèle principal — pas d'un ±MAE symétrique arbitraire.
"""

from __future__ import annotations

import functools

import joblib
import numpy as np
import pandas as pd

from ciel_tranquille.ml.train import MODEL_PATH


@functools.lru_cache(maxsize=1)
def load_payload(path: str | None = None) -> dict:
    return joblib.load(path or MODEL_PATH)


def predict_with_interval(X: pd.DataFrame, payload: dict | None = None) -> pd.DataFrame:
    """Retourne un DataFrame avec `laeq_pred`, `lower`, `upper`.

    L'intervalle est borné de façon cohérente (lower <= pred <= upper).
    """
    payload = payload or load_payload()
    X = X[payload["features"]].copy()
    pred = payload["model"].predict(X)
    lo = payload["quantile_lower"].predict(X)
    hi = payload["quantile_upper"].predict(X)
    lo = np.minimum(lo, pred)
    hi = np.maximum(hi, pred)
    return pd.DataFrame({"laeq_pred": pred, "lower": lo, "upper": hi})
