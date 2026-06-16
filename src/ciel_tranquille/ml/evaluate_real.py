"""Évaluation sur l'échantillon **réel** `bruit_survol.csv`.

Objectif d'honnêteté : le R² élevé obtenu à l'entraînement est mesuré sur des
données **synthétiques**. Ce module reporte, **séparément**, deux choses :

1. *Transfert* : le modèle déployé (entraîné sur synthétique) appliqué tel quel
   aux features réelles → mesure de l'écart de domaine (souvent défavorable).
2. *Baseline réelle* : un modèle entraîné/évalué **uniquement sur le réel** en
   validation croisée (prédictions out-of-fold) → ce qu'on peut honnêtement
   apprendre du peu de données réelles disponibles.

Diagnostic clé : combien de mesures réelles disposent d'un avion proche (la
jointure dépend du recouvrement temporel entre le snapshot OpenSky unique et les
mesures de bruit). C'est le facteur limitant n°1.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ciel_tranquille.config import REPO_ROOT, get_settings
from ciel_tranquille.ml.features import FEATURES, TARGET
from ciel_tranquille.ml.predict import load_payload
from ciel_tranquille.ml.train import RANDOM_STATE, _preprocessor
from ciel_tranquille.transform.clean import clean_noise, clean_states
from ciel_tranquille.transform.join import spatio_temporal_join

logger = logging.getLogger(__name__)

REAL_EVAL_PATH = REPO_ROOT / "models" / "real_evaluation.json"


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "n": int(len(y_true)),
    }


def build_real_features(settings=None) -> pd.DataFrame:
    """Construit la table de features réelle (bruit ↔ snapshot OpenSky réel)."""
    settings = settings or get_settings()
    noise = clean_noise(pd.read_csv(settings.samples_dir / "bruit_survol.csv"))

    snap = pd.read_csv(settings.samples_dir / "opensky_snapshot.csv")
    snap["snapshot_ts"] = snap["time_position_unix"]
    snap["on_ground"] = False
    states = clean_states(snap)

    enriched = spatio_temporal_join(noise, states)
    return enriched


def evaluate(settings=None) -> dict:
    settings = settings or get_settings()
    enriched = build_real_features(settings)
    X = enriched[FEATURES].copy()
    X["airport"] = X["airport"].astype(str)
    y = enriched[TARGET]

    n_with_aircraft = int((enriched["num_aircraft"] > 0).sum())
    diagnostic = {
        "real_rows": int(len(enriched)),
        "rows_with_nearby_aircraft": n_with_aircraft,
        "pct_with_aircraft": round(100 * n_with_aircraft / len(enriched), 1),
        "laeq_real_mean": round(float(y.mean()), 1),
        "laeq_real_min": round(float(y.min()), 1),
        "laeq_real_max": round(float(y.max()), 1),
    }

    # 1) Transfert : modèle synthétique -> réel.
    payload = load_payload()
    transfer = _metrics(y, payload["model"].predict(X))

    # 2) Baseline réelle en CV (out-of-fold).
    pipe = Pipeline(
        [("pre", _preprocessor()),
         ("model", GradientBoostingRegressor(
             n_estimators=200, max_depth=3, learning_rate=0.05,
             min_samples_leaf=5, random_state=RANDOM_STATE))]
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X, y, cv=kf)
    real_cv = _metrics(y, oof)

    result = {
        "diagnostic": diagnostic,
        "transfer_synthetic_to_real": transfer,
        "real_data_cv_baseline": real_cv,
        "deployed_model": payload["best_model"],
    }
    REAL_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REAL_EVAL_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = evaluate()
    d = r["diagnostic"]
    print("\n=== Diagnostic données réelles ===")
    print(f"  mesures réelles : {d['real_rows']} · avec avion proche : "
          f"{d['rows_with_nearby_aircraft']} ({d['pct_with_aircraft']} %)")
    print(f"  LAeq réel : {d['laeq_real_min']}–{d['laeq_real_max']} dB (moy {d['laeq_real_mean']})")
    print("\n=== Performance ===")
    t, c = r["transfer_synthetic_to_real"], r["real_data_cv_baseline"]
    print(f"  Transfert synthétique→réel : RMSE {t['rmse']} · MAE {t['mae']} · R² {t['r2']}")
    print(f"  Baseline réelle (CV oof)   : RMSE {c['rmse']} · MAE {c['mae']} · R² {c['r2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
