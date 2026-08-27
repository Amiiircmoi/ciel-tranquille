"""Cadre d'évaluation du modèle réel co-localisé — **découpe par station**.

Trois principes, chacun motivé par une façon précise de se tromper.

**1. On ne découpe jamais au hasard.** Deux survols séparés de quelques minutes
partagent l'appareil, la trajectoire, la météo et l'état du sol. Un découpage
aléatoire met ces quasi-doublons de part et d'autre de la frontière : le modèle
retrouve en test ce qu'il a mémorisé en entraînement, et le score obtenu ne dit
rien de ce qui nous intéresse — la capacité à prédire le bruit **là où l'on n'a
pas de sonomètre**. C'est précisément l'usage visé. On découpe donc **par
station**, chaque station étant un site avec sa géométrie, son couloir et son
environnement propres.

**2. Une station est tenue entièrement à l'écart.** La validation
leave-one-station-out sert à *choisir* entre familles de modèles ; s'en servir
aussi pour annoncer la performance finale reviendrait à publier le score du
gagnant d'un concours auquel on a participé. La station de réserve n'entre dans
aucun choix : ni sélection, ni réglage, ni arrêt anticipé. Elle est ouverte une
fois, à la fin.

**3. La baseline physique traverse exactement le même harnais.** Mêmes découpes,
mêmes métriques, même code. Une baseline évaluée à part ne prouve rien. Elle
n'est pas là pour faire joli : si le modèle ne la bat pas, c'est le résultat, et
il se rapporte tel quel.

Toutes les métriques sont calculées sur l'**instantané figé** (`ml/dataset.py`),
dont la somme de contrôle est vérifiée au chargement. Lire la collecte vive
donnerait des chiffres non reproductibles.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ciel_tranquille.ml.baseline import FittedSlopeBaseline, SphericalSpreadingBaseline
from ciel_tranquille.ml.dataset import sha256_file
from ciel_tranquille.ml.features_real import (
    CATEGORICAL_FEATURES,
    DISTANCE_COL,
    FEATURES,
    GROUP_COL,
    LEAKAGE_COLS,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
    check_no_leakage,
    usable_rows,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
BASELINE_NAME = "BaselinePhysique(-20dB/décade)"
DIAGNOSTIC_NAME = "BaselinePenteAjustée(diagnostic)"

# Une station trop peu fournie donne un score de test dominé par le bruit
# d'échantillonnage : on l'exclut du rôle de station de test, sans retirer ses
# lignes de l'entraînement.
MIN_PAIRS_PAR_STATION_TEST = 50


class SnapshotIntegrityError(RuntimeError):
    """L'instantané ne correspond pas à son manifeste : évaluation refusée."""


def load_frozen(dataset_path: Path | str, manifest_path: Path | str | None = None) -> pd.DataFrame:
    """Charge l'instantané figé après vérification de sa somme de contrôle.

    Refuser un fichier qui ne correspond pas à son manifeste n'est pas de la
    paranoïa : c'est ce qui rend une métrique citable. Sans cette vérification,
    « MAE = 3,1 dB sur 5 024 paires » n'est qu'une affirmation.
    """
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path) if manifest_path else dataset_path.with_suffix(
        ".manifest.json"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Instantané introuvable : {dataset_path}")
    if not manifest_path.exists():
        raise SnapshotIntegrityError(
            f"Manifeste introuvable ({manifest_path}) — un instantané sans manifeste "
            "n'est pas vérifiable, donc pas utilisable pour une métrique publiée."
        )
    manifeste = json.loads(manifest_path.read_text(encoding="utf-8"))
    attendu = manifeste.get("sha256")
    obtenu = sha256_file(dataset_path)
    if attendu != obtenu:
        raise SnapshotIntegrityError(
            f"Somme de contrôle divergente pour {dataset_path.name} : "
            f"manifeste={attendu} fichier={obtenu}"
        )
    logger.info(
        "Instantané vérifié : %s (%s paires, %s)",
        dataset_path.name, manifeste.get("n_paires"), manifeste.get("exported_at_iso"),
    )
    return pd.read_parquet(dataset_path)


def active_features(exclude: Iterable[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """(numériques, catégorielles, toutes) après retrait explicite de variables.

    Écarter une variable par **configuration** plutôt qu'en supprimant son code
    laisse la trace du choix : la dérivation reste testée, la variable reste
    disponible pour une fenêtre de collecte où elle aurait du sens, et le
    fichier de résultats peut nommer ce qui a été retiré et pourquoi. Supprimer
    le code rendrait l'exécution irreproductible six mois plus tard.
    """
    retire = set(exclude or ())
    num = [c for c in NUMERIC_FEATURES if c not in retire]
    cat = [c for c in CATEGORICAL_FEATURES if c not in retire]
    return num, cat, num + cat


def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )


def candidate_models(num: list[str], cat: list[str]) -> dict[str, Pipeline]:
    """Trois familles, du plus simple au plus complexe.

    Hyperparamètres **contraints et non optimisés** : le point n'est pas le
    dernier point de R². Sur un découpage par station, le surapprentissage ne se
    manifeste pas comme d'habitude — le modèle n'apprend pas le bruit
    d'échantillonnage, il apprend *les stations d'entraînement*. Des arbres
    profonds sculptent la géométrie propre à chaque site et transfèrent mal.
    D'où des profondeurs bornées et des feuilles peuplées, plus prudentes que ce
    qu'un découpage aléatoire suggérerait.
    """
    return {
        "RegressionLineaire": Pipeline([
            ("pre", _preprocessor(num, cat)),
            ("model", LinearRegression()),
        ]),
        "ForetAleatoire": Pipeline([
            ("pre", _preprocessor(num, cat)),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=10,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("pre", _preprocessor(num, cat)),
            ("model", GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                min_samples_leaf=10, subsample=0.9, random_state=RANDOM_STATE)),
        ]),
    }


def metrics(y_true, y_pred) -> dict:
    """MAE et RMSE en décibels, R² sans unité."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae_db": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_db": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }


def _fit_predict(nom: str, estimateur, train: pd.DataFrame, test: pd.DataFrame,
                 features: list[str] | None = None):
    """Ajuste puis prédit. La baseline ne consomme que la distance oblique."""
    if nom in (BASELINE_NAME, DIAGNOSTIC_NAME):
        estimateur.fit(train[DISTANCE_COL], train[TARGET])
        return estimateur.predict(test[DISTANCE_COL])
    feats = FEATURES if features is None else features
    check_no_leakage(feats)
    estimateur.fit(train[feats], train[TARGET])
    return estimateur.predict(test[feats])


def _tous_les_estimateurs(exclude: Iterable[str] | None = None) -> dict:
    num, cat, _ = active_features(exclude)
    estimateurs = {
        BASELINE_NAME: SphericalSpreadingBaseline(),
        DIAGNOSTIC_NAME: FittedSlopeBaseline(),
    }
    estimateurs.update(candidate_models(num, cat))
    return estimateurs


def leave_one_station_out(
    df: pd.DataFrame,
    stations_test: list[str] | None = None,
    exclude: Iterable[str] | None = None,
    collect_predictions: bool = False,
    only: Iterable[str] | None = None,
) -> dict:
    """Évalue chaque famille en tenant une station à l'écart, tour à tour.

    Le score agrégé est la **moyenne des scores par station**, pas le score
    calculé sur l'ensemble des prédictions concaténées : sans cela une station
    très productive écraserait les autres, et l'on mesurerait sa géométrie
    particulière plutôt que la capacité à généraliser. L'écart type entre plis
    est rapporté avec la moyenne — une moyenne flatteuse assise sur une forte
    dispersion ne se transporte pas.
    """
    _, _, features = active_features(exclude)
    stations = stations_test or sorted(
        s for s, n in df[GROUP_COL].value_counts().items() if n >= MIN_PAIRS_PAR_STATION_TEST
    )
    resultats: dict[str, dict] = {}

    retenus = set(only) if only else None
    hors_pli: dict[str, pd.Series] = {}
    for nom in _tous_les_estimateurs(exclude):
        if retenus is not None and nom not in retenus:
            continue
        par_station, morceaux = {}, []
        for station in stations:
            train = df[df[GROUP_COL] != station]
            test = df[df[GROUP_COL] == station]
            if train.empty or test.empty:
                continue
            estimateur = _tous_les_estimateurs(exclude)[nom]  # instance neuve par pli
            y_pred = _fit_predict(nom, estimateur, train, test, features)
            par_station[station] = metrics(test[TARGET], y_pred)
            if collect_predictions:
                morceaux.append(pd.Series(np.asarray(y_pred), index=test.index))
        if not par_station:
            continue
        if collect_predictions and morceaux:
            hors_pli[nom] = pd.concat(morceaux).sort_index()
        resultats[nom] = {
            "par_station": par_station,
            "moyenne": {
                cle: round(float(np.mean([m[cle] for m in par_station.values()])), 3)
                for cle in ("mae_db", "rmse_db", "r2")
            },
            "ecart_type": {
                cle: round(float(np.std([m[cle] for m in par_station.values()])), 3)
                for cle in ("mae_db", "rmse_db", "r2")
            },
            "pire_station": min(par_station.items(), key=lambda kv: kv[1]["r2"])[0],
        }
    if collect_predictions:
        resultats["_predictions_hors_pli"] = hors_pli
    return resultats


def evaluate_holdout(
    df: pd.DataFrame, holdout_station: str, exclude: Iterable[str] | None = None
) -> dict:
    """Évaluation finale : entraînement sur toutes les autres, test sur la réserve."""
    _, _, features = active_features(exclude)
    train = df[df[GROUP_COL] != holdout_station]
    test = df[df[GROUP_COL] == holdout_station]
    if test.empty:
        raise ValueError(f"Station de réserve absente du jeu : {holdout_station}")

    sortie = {}
    modeles: dict[str, object] = {}
    for nom, estimateur in _tous_les_estimateurs(exclude).items():
        y_pred = _fit_predict(nom, estimateur, train, test, features)
        entree = metrics(test[TARGET], y_pred)
        if hasattr(estimateur, "describe"):
            entree["parametres"] = estimateur.describe()
        sortie[nom] = entree
        modeles[nom] = estimateur
    return {
        "station_reserve": holdout_station,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "stations_entrainement": sorted(train[GROUP_COL].unique()),
        "resultats": sortie,
        "_modeles": modeles,   # non sérialisé : sert aux figures et à SHAP
        "_train": train,
        "_test": test,
    }


def choose_holdout(df: pd.DataFrame) -> str:
    """Station de réserve choisie **par une règle**, pas à la main.

    Règle : parmi les stations suffisamment fournies, celle dont le nombre de
    paires est le plus proche de la médiane. Prendre la plus productive
    flatterait le résultat ; prendre la moins productive le noircirait. Une
    station médiane est le cas représentatif — et le critère est écrit d'avance,
    donc non ajustable après avoir vu les scores.
    """
    comptes = df[GROUP_COL].value_counts()
    eligibles = comptes[comptes >= MIN_PAIRS_PAR_STATION_TEST]
    if eligibles.empty:
        raise ValueError("Aucune station n'atteint le minimum de paires pour servir de test.")
    mediane = eligibles.median()
    return str((eligibles - mediane).abs().idxmin())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Paires figées → matrice prête, sans jamais toucher au disque."""
    features = build_features(df)
    prepared = usable_rows(features)
    logger.info(
        "%d paires exploitables sur %d (%d écartées pour valeur manquante).",
        len(prepared), len(df), len(df) - len(prepared),
    )
    return prepared


def run(
    dataset_path: Path | str,
    manifest_path=None,
    holdout_station: str | None = None,
    exclude_features: Iterable[str] | None = None,
) -> dict:
    """Chaîne complète : chargement vérifié → réserve → validation croisée.

    Deux niveaux, volontairement distincts :

    - **(a) chiffre principal** : la station de réserve, jamais vue, ouverte une
      seule fois. C'est le chiffre qu'on publie.
    - **(b) validation croisée par station** sur les 10 stations, chacune servant
      de test à son tour. Elle dit à quel point le chiffre principal dépend de
      la station qu'on a tirée — un écart type élevé entre plis signifie que le
      résultat ne se transporte pas, quelle que soit la moyenne.
    """
    exclude = list(exclude_features or ())
    brut = load_frozen(dataset_path, manifest_path)
    df = prepare(brut)
    reserve = holdout_station or choose_holdout(df)
    num, cat, features = active_features(exclude)
    logger.info("Station de réserve : %s | %d variables actives", reserve, len(features))

    final = evaluate_holdout(df, reserve, exclude)
    loso_complet = leave_one_station_out(df, exclude=exclude, collect_predictions=True)
    # Les prédictions hors pli sortent du dictionnaire de résultats : elles ne
    # sont ni sérialisables ni comparables aux entrées « un modèle = une ligne ».
    predictions = loso_complet.pop("_predictions_hors_pli", {})
    loso_selection = leave_one_station_out(df[df[GROUP_COL] != reserve], exclude=exclude)

    return {
        "instantane": str(dataset_path),
        "n_paires": int(len(df)),
        "stations": sorted(df[GROUP_COL].unique()),
        "paires_par_station": {k: int(v) for k, v in df[GROUP_COL].value_counts().items()},
        "protocole": {
            "decoupe": "par station (leave-one-station-out), jamais aléatoire",
            "station_reserve": reserve,
            "reserve_utilisee_pour_selection": False,
            "cible": TARGET,
            "features_actives": features,
            "features_numeriques": num,
            "features_categorielles": cat,
            "features_ecartees": exclude,
            "features_interdites": LEAKAGE_COLS,
            "hyperparametres_optimises": False,
        },
        "evaluation_finale": final,
        "validation_croisee_10_stations": loso_complet,
        "_predictions_hors_pli": predictions,
        "_frame": df,
        "validation_croisee_hors_reserve": loso_selection,
    }


def _tableau(titre: str, lignes: dict, cle_metriques=lambda v: v) -> str:
    out = [f"\n{titre}", f"{'Modèle':<34}{'MAE dB':>9}{'RMSE dB':>10}{'R²':>9}{'n':>8}"]
    for nom, valeurs in lignes.items():
        m = cle_metriques(valeurs)
        out.append(
            f"{nom:<34}{m['mae_db']:>9.3f}{m['rmse_db']:>10.3f}{m['r2']:>9.4f}"
            f"{m.get('n', ''):>8}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Évalue modèles et baseline physique sur un instantané figé."
    )
    parser.add_argument("dataset", help="chemin du Parquet figé (jamais la collecte vive)")
    parser.add_argument("--manifest", default=None,
                        help="manifeste (défaut : <dataset>.manifest.json)")
    parser.add_argument("--holdout-station", default=None, help="station de réserve imposée")
    parser.add_argument("--exclude-feature", action="append", default=[],
                        help="variable écartée de cette exécution (répétable)")
    parser.add_argument("--report", default=None, help="chemin du rapport JSON à écrire")
    args = parser.parse_args(argv)

    resultat = run(args.dataset, args.manifest, args.holdout_station, args.exclude_feature)

    final = resultat["evaluation_finale"]
    print(_tableau(
        f"Évaluation finale — réserve {final['station_reserve']} "
        f"({final['n_test']} paires, jamais vue)",
        final["resultats"],
    ))
    print(_tableau(
        "Validation croisée par station (10 plis, moyenne)",
        resultat["validation_croisee_10_stations"],
        lambda v: {**v["moyenne"], "n": len(v["par_station"])},
    ))

    if args.report:
        Path(args.report).write_text(
            json.dumps(_serialisable(resultat), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nRapport écrit : {args.report}")
    return 0


def _serialisable(resultat: dict) -> dict:
    """Retire les objets non JSON (modèles ajustés, trames) avant écriture."""
    sans_prive = {k: v for k, v in resultat.items() if not k.startswith("_")}
    copie = json.loads(json.dumps(sans_prive, default=lambda o: None))
    finale = copie.get("evaluation_finale", {})
    for cle in ("_modeles", "_train", "_test", "_exclude"):
        finale.pop(cle, None)
    return copie


if __name__ == "__main__":
    raise SystemExit(main())
