"""Entraînement et **comparaison** de modèles.

Pipeline :
1. Chargement des features depuis DuckDB (`features.load_training_frame`).
2. Pré-traitement (one-hot de l'aéroport + passthrough numérique).
3. Comparaison de 4 familles d'algorithmes :
   - Régression linéaire (baseline interprétable),
   - Random Forest,
   - Gradient Boosting,
   - HistGradientBoosting (boosting histogramme, rapide).
4. **Anti-surapprentissage** : validation croisée K-fold + suivi de
   l'écart train/test ; profondeurs et `min_samples_leaf` contraints.
5. Sélection du meilleur modèle (RMSE en CV) et **intervalles d'incertitude**
   par régression quantile (GBR loss="quantile" à 10 % et 90 %) — bien plus
   honnête qu'un ±MAE symétrique.
6. Sauvegarde du payload (modèle + quantiles + métadonnées + comparatif).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ciel_tranquille.config import REPO_ROOT, get_settings
from ciel_tranquille.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    load_training_frame,
    split_xy,
)

logger = logging.getLogger(__name__)

MODEL_PATH = REPO_ROOT / "models" / "noise_model.joblib"
REPORT_PATH = REPO_ROOT / "models" / "model_comparison.json"
RANDOM_STATE = 42


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def _candidate_models() -> dict[str, object]:
    """Modèles candidats. Hyperparamètres contraints (anti-overfit)."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.9, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=400,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
    }


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(settings=None, test_size: float = 0.2) -> dict:
    settings = settings or get_settings()
    df = load_training_frame(settings)
    if len(df) < 50:
        raise RuntimeError(
            f"Trop peu de données ({len(df)}). Lancez d'abord `ct-synth` puis "
            "`build_curated(noise_csv='bruit_synth.csv')`."
        )
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison: list[dict] = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in _candidate_models().items():
        pipe = Pipeline([("pre", _preprocessor()), ("model", estimator)])
        # Validation croisée (anti-overfit) : RMSE moyen + écart-type.
        cv_rmse = -cross_val_score(
            pipe, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error"
        )
        pipe.fit(X_train, y_train)
        train_m = _metrics(y_train, pipe.predict(X_train))
        test_m = _metrics(y_test, pipe.predict(X_test))
        comparison.append(
            {
                "model": name,
                "cv_rmse_mean": round(float(cv_rmse.mean()), 3),
                "cv_rmse_std": round(float(cv_rmse.std()), 3),
                "train_rmse": round(train_m["rmse"], 3),
                "test_rmse": round(test_m["rmse"], 3),
                "test_mae": round(test_m["mae"], 3),
                "test_r2": round(test_m["r2"], 3),
                # Écart train/test = indicateur de surapprentissage.
                "overfit_gap_rmse": round(test_m["rmse"] - train_m["rmse"], 3),
            }
        )
        fitted[name] = pipe
        logger.info(
            "%-22s CV_RMSE=%.3f±%.3f  test_RMSE=%.3f  R2=%.3f  gap=%.3f",
            name, cv_rmse.mean(), cv_rmse.std(), test_m["rmse"], test_m["r2"],
            test_m["rmse"] - train_m["rmse"],
        )

    best = min(comparison, key=lambda c: c["cv_rmse_mean"])
    best_name = best["model"]
    best_pipe = fitted[best_name]

    # Intervalles d'incertitude par régression quantile (10 % / 90 %).
    q_lower = Pipeline(
        [("pre", _preprocessor()),
         ("model", GradientBoostingRegressor(
             loss="quantile", alpha=0.1, n_estimators=300, max_depth=3,
             learning_rate=0.05, random_state=RANDOM_STATE))]
    ).fit(X_train, y_train)
    q_upper = Pipeline(
        [("pre", _preprocessor()),
         ("model", GradientBoostingRegressor(
             loss="quantile", alpha=0.9, n_estimators=300, max_depth=3,
             learning_rate=0.05, random_state=RANDOM_STATE))]
    ).fit(X_train, y_train)

    payload = {
        "model": best_pipe,
        "quantile_lower": q_lower,
        "quantile_upper": q_upper,
        "features": FEATURES,
        "target": "laeq_db",
        "best_model": best_name,
        "metrics": {
            "rmse": best["test_rmse"],
            "mae": best["test_mae"],
            "r2": best["test_r2"],
        },
        "comparison": comparison,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)
    Path(REPORT_PATH).write_text(
        json.dumps(
            {"best_model": best_name, "comparison": comparison,
             "n_train": payload["n_train"], "n_test": payload["n_test"]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Meilleur modèle : %s (sauvegardé dans %s)", best_name, MODEL_PATH)
    return {"best_model": best_name, "comparison": comparison, "model_path": str(MODEL_PATH)}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description="Entraîne et compare les modèles.").parse_args(argv)
    result = train()
    print(f"\nMeilleur modèle : {result['best_model']}\n")
    print(f"{'Modèle':<22}{'CV_RMSE':>10}{'test_RMSE':>11}{'test_MAE':>10}{'R2':>8}{'gap':>8}")
    for c in result["comparison"]:
        print(
            f"{c['model']:<22}{c['cv_rmse_mean']:>10.3f}{c['test_rmse']:>11.3f}"
            f"{c['test_mae']:>10.3f}{c['test_r2']:>8.3f}{c['overfit_gap_rmse']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
