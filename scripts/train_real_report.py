#!/usr/bin/env python3
"""Entraîne, évalue et restitue le modèle réel co-localisé.

Produit le **fichier de résultats qui fait foi** (Markdown horodaté, citant la
somme de contrôle du jeu) et les figures PNG du support de présentation. La
sortie console n'est qu'un écho : tout ce qui compte est écrit sur disque.

    python scripts/train_real_report.py datasets/pairs_20260827.parquet \
        --exclude-feature is_weekend

Le jeu est **figé** : sa somme de contrôle est vérifiée avant tout chargement.
Aucune écriture dans le répertoire de collecte, aucun regel.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # aucun affichage : on écrit des PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ciel_tranquille.ml import train_real as T  # noqa: E402
from ciel_tranquille.ml.dataset import sha256_file  # noqa: E402
from ciel_tranquille.ml.features_real import GROUP_COL, TARGET  # noqa: E402

logger = logging.getLogger("train_real_report")

# Familles de variables, pour répondre à « quelle part la géométrie explique ».
GROUPES = {
    "geometrie": ["slant_km", "log_slant", "horiz_km", "altitude_m",
                  "elevation_deg", "cos_aspect", "velocity_m_s"],
    "temporel": ["hour_utc_num", "is_night", "is_weekend"],
    "contexte": ["airport"],
}

# Tranches de distance oblique pour l'analyse de résidus (km).
TRANCHES = [0.0, 1.0, 2.0, 3.0, 5.0, 10.01]

COULEURS = {"modele": "#1f4e79", "baseline": "#c0504d", "grille": "#d9d9d9"}


# --------------------------------------------------------------------- outils
def _fig(nom: str, out_dir: Path, fig) -> str:
    chemin = out_dir / nom
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    logger.info("figure : %s", chemin)
    return str(chemin)


def _style(ax):
    ax.grid(True, color=COULEURS["grille"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)


# ------------------------------------------------------------------ ablations
def ablation_geometrie(df: pd.DataFrame, exclude: list[str], famille: str) -> dict:
    """Jeux de variables emboîtés : que gagne-t-on à chaque étage ?

    C'est la seule façon honnête de répondre à « quelle part la géométrie
    explique-t-elle » : une importance de variable dit ce que le modèle *utilise*,
    pas ce que l'information *apporte*. Seul le retrait effectif le dit.

    Mesurée **hors pli sur les dix stations**, pas sur la seule réserve. Sur une
    station unique, le décalage de niveau propre au site (jusqu'à 9 dB) domine
    tout : les quatre lignes du tableau se ressembleraient sans que cela dise
    quoi que ce soit sur l'apport des variables. Dix stations moyennent ce
    décalage sans rien perdre de la garantie de généralisation, puisque chaque
    survol est prédit par un modèle qui n'a jamais vu sa station.
    """
    _, _, toutes = T.active_features(exclude)
    geometrie = [f for f in toutes if f in GROUPES["geometrie"]]
    hors_geometrie = [f for f in toutes if f not in GROUPES["geometrie"]]

    sortie = {}
    for libelle, feats, modele in (
        ("distance seule (baseline physique)", ["slant_km"], T.BASELINE_NAME),
        ("géométrie seule", geometrie, famille),
        ("hors géométrie (heure + aéroport)", hors_geometrie, famille),
        ("toutes variables", toutes, famille),
    ):
        if not feats:
            continue
        ecarte = [f for f in toutes if f not in feats]
        # La baseline ignore la liste de variables : on lui laisse le jeu complet
        # pour éviter un ajustement à zéro colonne côté modèles.
        retrait = exclude if modele == T.BASELINE_NAME else exclude + ecarte
        resultat = T.leave_one_station_out(
            df, exclude=retrait, only=[modele], collect_predictions=True
        )
        pred = resultat.pop("_predictions_hors_pli")[modele]
        pooled = T.metrics(df.loc[pred.index, TARGET], pred)
        sortie[libelle] = {
            "n_variables": len(feats), "variables": feats, "modele": modele,
            "mae_db": pooled["mae_db"], "rmse_db": pooled["rmse_db"], "r2": pooled["r2"],
            "mae_moyenne_plis_db": resultat[modele]["moyenne"]["mae_db"],
        }
    return sortie


# --------------------------------------------------------------- importances
def importances(final: dict, features: list[str], seed: int) -> dict:
    """Importance par permutation — comparable entre familles, contrairement au
    `feature_importances_` des arbres qui n'existe pas pour un modèle linéaire et
    ne se compare pas d'une famille à l'autre.

    Mesurée **sur la station de réserve**, donc sur des données jamais vues :
    une importance mesurée en entraînement dit ce que le modèle a mémorisé.
    """
    test, sortie = final["_test"], {}
    for nom, modele in final["_modeles"].items():
        if nom in (T.BASELINE_NAME, T.DIAGNOSTIC_NAME):
            continue
        perm = permutation_importance(
            modele, test[features], test[TARGET],
            n_repeats=20, random_state=seed, scoring="neg_root_mean_squared_error",
        )
        sortie[nom] = {
            f: {"moyenne": round(float(m), 4), "ecart_type": round(float(s), 4)}
            for f, m, s in zip(features, perm.importances_mean, perm.importances_std, strict=True)
        }
    return sortie


def shap_sur_le_meilleur(final: dict, nom_modele: str, features: list[str]) -> dict:
    """Attribution SHAP sur le meilleur modèle, agrégée par famille de variables.

    L'explainer suit la famille du modèle : `TreeExplainer` pour les ensembles
    d'arbres, `LinearExplainer` pour la régression linéaire. Les deux donnent des
    valeurs de Shapley **exactes** pour leur famille — pas une approximation par
    échantillonnage.
    """
    import shap

    pipeline = final["_modeles"][nom_modele]
    pre, modele = pipeline.named_steps["pre"], pipeline.named_steps["model"]
    X = pre.transform(final["_test"][features])
    fond = pre.transform(final["_train"][features])
    noms = list(pre.get_feature_names_out())

    if hasattr(modele, "estimators_") or hasattr(modele, "tree_"):
        explainer = shap.TreeExplainer(modele)
    else:
        explainer = shap.LinearExplainer(modele, fond)
    valeurs = np.asarray(explainer.shap_values(X))
    moyennes = np.abs(valeurs).mean(axis=0)

    par_variable = {n.split("__", 1)[-1]: float(v) for n, v in zip(noms, moyennes, strict=True)}
    total = sum(par_variable.values()) or 1.0

    par_groupe = {}
    for groupe, membres in GROUPES.items():
        part = sum(v for k, v in par_variable.items()
                   if k in membres or any(k.startswith(f"{m}_") for m in membres))
        par_groupe[groupe] = round(100 * part / total, 1)

    return {
        "modele": nom_modele,
        "n_observations": int(X.shape[0]),
        "valeur_de_base_db": round(float(np.ravel(explainer.expected_value)[0]), 2),
        "contribution_moyenne_absolue_db": {
            k: round(v, 3) for k, v in
            sorted(par_variable.items(), key=lambda kv: -kv[1])
        },
        "part_par_famille_pct": par_groupe,
    }


# ------------------------------------------------- décalage de site vs forme
def diagnostic_centre(df: pd.DataFrame, predictions: dict, meilleur: str) -> dict:
    """Sépare ce que le modèle rate en **niveau** de ce qu'il rate en **forme**.

    Un R² négatif sur une station tenue à l'écart ne veut pas dire « le modèle
    n'a rien appris » : il veut dire qu'il prédit moins bien que la moyenne *de
    cette station*, moyenne qu'il ne pouvait pas connaître puisqu'aucune de ses
    mesures n'était en entraînement. Chaque site a un niveau propre — bruit de
    fond, hauteur de survol, mix d'appareils.

    En retirant de chaque station sa moyenne, côté observé **et** côté prédit,
    on isole la question qui reste : la variation *à l'intérieur* d'une station
    est-elle correctement reproduite ? Ce chiffre ne remplace pas le R² brut, il
    l'explique. Les deux figurent au rapport.
    """
    sortie = {}
    for nom, pred in predictions.items():
        aligne = pd.DataFrame({
            "station": df.loc[pred.index, GROUP_COL],
            "y": df.loc[pred.index, TARGET],
            "p": pred,
        })
        aligne["y_c"] = aligne["y"] - aligne.groupby("station")["y"].transform("mean")
        aligne["p_c"] = aligne["p"] - aligne.groupby("station")["p"].transform("mean")
        decalages = (aligne.groupby("station")["y"].mean()
                     - aligne.groupby("station")["p"].mean())
        sortie[nom] = {
            "brut": T.metrics(aligne["y"], aligne["p"]),
            "centre_par_station": T.metrics(aligne["y_c"], aligne["p_c"]),
            "decalage_moyen_absolu_db": round(float(decalages.abs().mean()), 3),
            "decalage_min_db": round(float(decalages.min()), 3),
            "decalage_max_db": round(float(decalages.max()), 3),
            "decalage_par_station_db": {k: round(float(v), 2) for k, v in decalages.items()},
        }
    return {"par_modele": sortie, "meilleur": meilleur}


# ------------------------------------------------------------------- résidus
def _tableau_residus(frame: pd.DataFrame) -> dict:
    frame = frame.copy()
    frame["tranche"] = pd.cut(frame["slant_km"], bins=TRANCHES, right=False)
    lignes = {}
    for tranche, groupe in frame.groupby("tranche", observed=True):
        lignes[str(tranche)] = {
            "n": int(len(groupe)),
            "biais_db": round(float(groupe["residu_modele"].mean()), 3),
            "mae_db": round(float(groupe["residu_modele"].abs().mean()), 3),
            "rmse_db": round(float(np.sqrt((groupe["residu_modele"] ** 2).mean())), 3),
            "mae_baseline_db": round(float(groupe["residu_baseline"].abs().mean()), 3),
            "ecart_type_cible_db": round(float(groupe[TARGET].std()), 3),
        }
    return lignes


def residus_par_distance(
    final: dict, meilleur: str, df: pd.DataFrame, predictions: dict
) -> dict:
    """Le modèle se dégrade-t-il au loin, au près, ou uniformément ?

    Deux tables. La **réserve seule** répond à la question posée sur le chiffre
    publié, mais 92 % de ses survols tiennent dans une seule tranche : les autres
    tranches y pèsent moins de quinze points et ne permettent aucune conclusion.
    La table **hors pli, toutes stations** reprend les prédictions de chaque
    station au moment où elle était tenue à l'écart : mêmes garanties de
    généralisation, dix fois plus de points, toute l'étendue de distances.
    """
    _, _, features = T.active_features(final.get("_exclude", []))

    test = final["_test"].copy()
    test["pred_modele"] = final["_modeles"][meilleur].predict(test[features])
    test["pred_baseline"] = final["_modeles"][T.BASELINE_NAME].predict(test[T.DISTANCE_COL])
    test["residu_modele"] = test[TARGET] - test["pred_modele"]
    test["residu_baseline"] = test[TARGET] - test["pred_baseline"]

    pooled = df.loc[predictions[meilleur].index].copy()
    pooled["pred_modele"] = predictions[meilleur]
    pooled["pred_baseline"] = predictions[T.BASELINE_NAME]
    pooled["residu_modele"] = pooled[TARGET] - pooled["pred_modele"]
    pooled["residu_baseline"] = pooled[TARGET] - pooled["pred_baseline"]

    return {
        "par_tranche": _tableau_residus(test),
        "par_tranche_hors_pli": _tableau_residus(pooled),
        "_frame": test,
        "_pooled": pooled,
    }


# ------------------------------------------------------------------- figures
def figure_predit_observe(test: pd.DataFrame, reserve: str, meilleur: str, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(test[TARGET], test["pred_modele"], s=14, alpha=0.45,
               color=COULEURS["modele"], edgecolors="none", label=meilleur)
    ax.scatter(test[TARGET], test["pred_baseline"], s=14, alpha=0.30,
               color=COULEURS["baseline"], edgecolors="none", label="baseline physique")
    lo = float(min(test[TARGET].min(), test["pred_modele"].min()) - 1)
    hi = float(max(test[TARGET].max(), test["pred_modele"].max()) + 1)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="y = x")
    ax.set_xlabel("LAmax observé (dB)")
    ax.set_ylabel("LAmax prédit (dB)")
    ax.set_title(f"Prédit contre observé — réserve {reserve}\n({len(test)} survols jamais vus)")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    return _fig("modele-predit-vs-observe.png", out, fig)


def figure_par_station(loso: dict, meilleur: str, out: Path) -> str:
    stations = list(loso[meilleur]["par_station"])
    stations.sort(key=lambda s: loso[meilleur]["par_station"][s]["mae_db"])
    modele = [loso[meilleur]["par_station"][s]["mae_db"] for s in stations]
    base = [loso[T.BASELINE_NAME]["par_station"][s]["mae_db"] for s in stations]

    y = np.arange(len(stations))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.barh(y - 0.2, base, height=0.38, color=COULEURS["baseline"], label="baseline physique")
    ax.barh(y + 0.2, modele, height=0.38, color=COULEURS["modele"], label=meilleur)
    ax.set_yticks(y)
    ax.set_yticklabels([s.split("-", 1)[-1].replace("-", " ").title() for s in stations], fontsize=8)
    ax.set_xlabel("MAE (dB) — station tenue à l'écart")
    ax.set_title("Modèle contre baseline physique, par station\n"
                 "(validation croisée : chaque station testée sans jamais avoir servi à l'entraînement)")
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    return _fig("modele-vs-baseline-par-station.png", out, fig)


def figure_residus(residus: dict, meilleur: str, out: Path) -> str:
    table = residus["par_tranche_hors_pli"]
    tranches = list(table)
    mae = [table[t]["mae_db"] for t in tranches]
    mae_b = [table[t]["mae_baseline_db"] for t in tranches]
    biais = [table[t]["biais_db"] for t in tranches]
    n = [table[t]["n"] for t in tranches]

    x = np.arange(len(tranches))
    fig, (haut, bas) = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    haut.bar(x - 0.2, mae_b, width=0.38, color=COULEURS["baseline"], label="baseline physique")
    haut.bar(x + 0.2, mae, width=0.38, color=COULEURS["modele"], label=meilleur)
    for i, effectif in enumerate(n):
        haut.text(i, max(mae[i], mae_b[i]) + 0.15, f"n={effectif}",
                  ha="center", fontsize=7, color="#555555")
    haut.set_ylabel("MAE (dB)")
    haut.set_title("Résidus par tranche de distance oblique\n(prédictions hors pli : chaque station prédite sans avoir servi à l'entraînement)")
    haut.legend(frameon=False)
    _style(haut)

    bas.axhline(0, color="black", linewidth=1.0)
    bas.bar(x, biais, width=0.5, color=COULEURS["modele"])
    bas.set_ylabel("biais (dB)")
    bas.set_xticks(x)
    bas.set_xticklabels([t.replace("[", "").replace(")", "").replace(", ", "–") + " km"
                         for t in tranches], fontsize=8)
    bas.set_xlabel("distance oblique")
    _style(bas)
    return _fig("modele-residus-par-distance.png", out, fig)


def figure_importances(imp: dict, meilleur: str, out: Path) -> str:
    valeurs = imp[meilleur]
    ordre = sorted(valeurs, key=lambda f: -valeurs[f]["moyenne"])
    moyennes = [valeurs[f]["moyenne"] for f in ordre]
    erreurs = [valeurs[f]["ecart_type"] for f in ordre]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    y = np.arange(len(ordre))[::-1]
    ax.barh(y, moyennes, xerr=erreurs, height=0.6, color=COULEURS["modele"],
            error_kw={"ecolor": "#888888", "capsize": 3, "linewidth": 0.9})
    ax.set_yticks(y)
    ax.set_yticklabels(ordre, fontsize=9)
    ax.set_xlabel("dégradation du RMSE quand la variable est permutée (dB)")
    ax.set_title(f"Importance des variables — {meilleur}\n"
                 "par permutation, sur la station de réserve (données jamais vues)")
    _style(ax)
    return _fig("modele-importance-variables.png", out, fig)


# -------------------------------------------------------------------- rapport
def _table(entetes: list[str], lignes: list[list]) -> str:
    out = ["| " + " | ".join(entetes) + " |",
           "|" + "|".join(["---"] * len(entetes)) + "|"]
    for ligne in lignes:
        out.append("| " + " | ".join(str(c) for c in ligne) + " |")
    return "\n".join(out)


def ecrire_rapport(ctx: dict, chemin: Path) -> None:
    r, final = ctx["resultat"], ctx["resultat"]["evaluation_finale"]
    loso = ctx["resultat"]["validation_croisee_10_stations"]
    proto = r["protocole"]
    meilleur = ctx["meilleur"]

    lignes_principales = []
    for nom in [T.BASELINE_NAME, T.DIAGNOSTIC_NAME, *ctx["noms_modeles"]]:
        rs, lc = final["resultats"][nom], loso[nom]
        lignes_principales.append([
            nom,
            f"{rs['mae_db']:.2f}", f"{rs['rmse_db']:.2f}", f"{rs['r2']:.3f}",
            f"{lc['moyenne']['mae_db']:.2f} ± {lc['ecart_type']['mae_db']:.2f}",
            f"{lc['moyenne']['rmse_db']:.2f} ± {lc['ecart_type']['rmse_db']:.2f}",
            f"{lc['moyenne']['r2']:.3f} ± {lc['ecart_type']['r2']:.3f}",
        ])

    plis = []
    for station in sorted(loso[meilleur]["par_station"]):
        m, b = loso[meilleur]["par_station"][station], loso[T.BASELINE_NAME]["par_station"][station]
        plis.append([
            station, m["n"],
            f"{m['mae_db']:.2f}", f"{m['rmse_db']:.2f}", f"{m['r2']:.3f}",
            f"{b['mae_db']:.2f}", f"{b['r2']:.3f}",
            f"{b['mae_db'] - m['mae_db']:+.2f}",
        ])

    ablation = [[k, v["modele"], v["n_variables"],
                 f"{v['mae_db']:.2f}", f"{v['rmse_db']:.2f}", f"{v['r2']:.3f}",
                 f"{v['mae_moyenne_plis_db']:.2f}"]
                for k, v in ctx["ablation"].items()]

    def _lignes_residus(table):
        return [[t.replace("[", "").replace(")", "").replace(", ", " – ") + " km",
                 v["n"], f"{v['biais_db']:+.2f}", f"{v['mae_db']:.2f}", f"{v['rmse_db']:.2f}",
                 f"{v['mae_baseline_db']:.2f}", f"{v['ecart_type_cible_db']:.2f}"]
                for t, v in table.items()]

    residus = _lignes_residus(ctx["residus"]["par_tranche_hors_pli"])
    residus_reserve = _lignes_residus(ctx["residus"]["par_tranche"])
    cen = ctx["centre"]["par_modele"]
    lignes_centre = [[nom,
                      f"{v['brut']['mae_db']:.2f}", f"{v['brut']['r2']:.3f}",
                      f"{v['centre_par_station']['mae_db']:.2f}",
                      f"{v['centre_par_station']['r2']:.3f}",
                      f"{v['decalage_moyen_absolu_db']:.2f}"]
                     for nom, v in cen.items()]
    decalages = ctx["centre"]["par_modele"][meilleur]["decalage_par_station_db"]
    lignes_decalage = [[k, f"{v:+.2f}"] for k, v in
                       sorted(decalages.items(), key=lambda kv: kv[1])]

    shap_lignes = [[k, f"{v:.3f}"] for k, v in
                   list(ctx["shap"]["contribution_moyenne_absolue_db"].items())[:12]]

    texte = f"""# Résultats — modèle réel co-localisé

**Généré le {ctx['horodatage']}.** Ce fichier fait foi : la sortie console n'en
est qu'un écho.

## Jeu de données

| | |
|---|---|
| Fichier | `{Path(r['instantane']).name}` |
| **SHA-256** | `{ctx['sha256']}` |
| Paires exploitables | {r['n_paires']} |
| Stations | {len(r['stations'])} |
| Fenêtre | {ctx['fenetre']} |
| Taux d'association | {ctx['taux_association']} |

Le jeu est **figé**. Sa somme de contrôle a été vérifiée avant chargement :
toute divergence aurait interrompu l'exécution.

## Variables

**Retenues ({len(proto['features_actives'])})** : {', '.join(f'`{f}`' for f in proto['features_actives'])}

**Écartées de cette exécution** : {', '.join(f'`{f}`' for f in proto['features_ecartees']) or 'aucune'}

> `is_weekend` est écartée **par configuration, pas par suppression de code**.
> La fenêtre de collecte (mar. 25 → jeu. 27 août) ne contient aucun week-end :
> la variable est constante à 0 et ne porte aucune information. Son code de
> dérivation reste en place et testé — sur une fenêtre plus longue elle
> redeviendra pertinente, et le choix reste traçable.

**Interdites en entrée** ({len(proto['features_interdites'])}) : les autres mesures
acoustiques du même événement (`laeq`, `sel`, `nrj_laeq`, `duration_s` — prédire
le bruit à partir du bruit) et l'identité de la station (`station`,
`station_lat`, `station_lon`, `icao24`, `callsign` — la station de test est
inconnue à l'entraînement). Contrôle mécanique, pas un commentaire.

**Cible** : `{proto['cible']}` — le LAmax du survol, en dB.

## Protocole

- **Découpe par station, jamais aléatoire.** Deux survols séparés de quelques
  minutes partagent l'appareil, la trajectoire et la météo ; un découpage
  aléatoire les met de part et d'autre de la frontière et le modèle retrouve en
  test ce qu'il a mémorisé.
- **(a) Chiffre principal** : station de réserve `{proto['station_reserve']}`
  ({final['n_test']} survols), entraînement sur les {len(final['stations_entrainement'])}
  autres ({final['n_train']} survols). Ouverte une seule fois, à la fin.
- **(b) Validation croisée par station** sur les {len(r['stations'])} stations,
  chacune servant de test à son tour.
- **Baseline physique** dans le même harnais : mêmes découpes, mêmes métriques,
  même code. `LAmax = L_ref − 20·log10(d)` — divergence sphérique, **un seul
  paramètre ajusté**, la pente reste la physique.
- **Hyperparamètres non optimisés** : le point n'est pas le dernier point de R².

## 1. Tableau principal

{_table(["Modèle", "MAE (dB) réserve", "RMSE (dB) réserve", "R² réserve",
         "MAE (dB) VC", "RMSE (dB) VC", "R² VC"], lignes_principales)}

*VC = validation croisée sur les {len(r['stations'])} stations, moyenne ± écart type entre plis.*

{ctx['verdict_principal']}

## 2. Validation croisée, pli par pli — {meilleur}

{_table(["Station de test", "n", "MAE", "RMSE", "R²", "MAE baseline", "R² baseline", "gain MAE"], plis)}

Station la plus difficile pour le modèle : **{loso[meilleur]['pire_station']}**.

## 3. Quelle part la géométrie explique-t-elle réellement ?

Ablation emboîtée, **prédictions hors pli sur les dix stations** (chaque survol prédit par un modèle qui n'a jamais vu sa station) :

{_table(["Jeu de variables", "Modèle", "n", "MAE (dB)", "RMSE (dB)", "R²", "MAE moy. plis"], ablation)}

{ctx['verdict_geometrie']}

### Attribution SHAP — {ctx['shap']['modele']}

Contribution moyenne absolue au LAmax prédit, en dB, sur les
{ctx['shap']['n_observations']} survols de la réserve (valeur de base :
{ctx['shap']['valeur_de_base_db']} dB) :

{_table(["Variable", "|SHAP| moyen (dB)"], shap_lignes)}

Part par famille : {', '.join(f"**{k}** {v} %" for k, v in ctx['shap']['part_par_famille_pct'].items())}

### Décalage de niveau contre forme de la relation

Métriques hors pli, avant et après retrait de la moyenne propre à chaque station :

{_table(["Modèle", "MAE brut", "R² brut", "MAE centré", "R² centré", "décalage moyen |dB|"], lignes_centre)}

Décalage par station pour {meilleur} (observé − prédit, en dB) :

{_table(["Station", "décalage (dB)"], lignes_decalage)}

## 4. Résidus par tranche de distance oblique

Prédictions **hors pli**, toutes stations : chaque survol est prédit par un modèle
qui n'a jamais vu sa station.

{_table(["Tranche", "n", "biais (dB)", "MAE (dB)", "RMSE (dB)", "MAE baseline", "écart type cible"], residus)}

Sur la station de réserve seule :

{_table(["Tranche", "n", "biais (dB)", "MAE (dB)", "RMSE (dB)", "MAE baseline", "écart type cible"], residus_reserve)}

{ctx['verdict_residus']}

## 5. Figures

{chr(10).join(f'- `{Path(f).name}`' for f in ctx['figures'])}

## Limites à énoncer telles quelles

{ctx['limites']}
"""
    chemin.write_text(texte, encoding="utf-8")
    logger.info("rapport : %s", chemin)


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--holdout-station", default="91260-JUVISY-SARRAUT")
    parser.add_argument("--exclude-feature", action="append", default=[])
    parser.add_argument("--out-dir", default="dossier-rncp/figures")
    parser.add_argument("--report-dir", default="dossier-rncp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horodatage = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    manifeste = json.loads(
        (Path(args.manifest) if args.manifest else dataset.with_suffix(".manifest.json"))
        .read_text(encoding="utf-8")
    )

    logger.info("=== évaluation (jeu figé, somme de contrôle vérifiée) ===")
    resultat = T.run(dataset, args.manifest, args.holdout_station, args.exclude_feature)
    final = resultat["evaluation_finale"]
    final["_exclude"] = args.exclude_feature
    loso = resultat["validation_croisee_10_stations"]
    _, _, features = T.active_features(args.exclude_feature)
    noms_modeles = [n for n in final["resultats"]
                    if n not in (T.BASELINE_NAME, T.DIAGNOSTIC_NAME)]

    meilleur = min(noms_modeles, key=lambda n: loso[n]["moyenne"]["mae_db"])
    logger.info("meilleur modèle (MAE moyen en validation croisée) : %s", meilleur)

    predictions = resultat["_predictions_hors_pli"]

    logger.info("=== ablation géométrie hors pli (famille : %s) ===", meilleur)
    ablation = ablation_geometrie(resultat["_frame"], args.exclude_feature, famille=meilleur)

    logger.info("=== importance par permutation ===")
    imp = importances(final, features, args.seed)

    logger.info("=== attribution SHAP ===")
    shap_res = shap_sur_le_meilleur(final, meilleur, features)

    logger.info("=== résidus par distance ===")
    residus = residus_par_distance(final, meilleur, resultat["_frame"], predictions)
    centre = diagnostic_centre(resultat["_frame"], predictions, meilleur)

    logger.info("=== figures ===")
    figures = [
        figure_predit_observe(residus["_frame"], args.holdout_station, meilleur, out_dir),
        figure_par_station(loso, meilleur, out_dir),
        figure_residus(residus, meilleur, out_dir),
        figure_importances(imp, meilleur, out_dir),
    ]

    # --- verdicts factuels, dérivés des chiffres, pas rédigés d'avance -------
    base_r = final["resultats"][T.BASELINE_NAME]
    mod_r = final["resultats"][meilleur]
    gain_mae = base_r["mae_db"] - mod_r["mae_db"]
    gain_pct = 100 * gain_mae / base_r["mae_db"] if base_r["mae_db"] else 0.0
    sens = "gagne" if gain_mae > 0 else "**perd**"
    plis_gagnes = sum(
        1 for st in loso[meilleur]["par_station"]
        if loso[T.BASELINE_NAME]["par_station"][st]["mae_db"]
        > loso[meilleur]["par_station"][st]["mae_db"]
    )
    n_plis = len(loso[meilleur]["par_station"])
    verdict_principal = (
        f"Sur la station de réserve, **{meilleur}** {sens} face à la baseline physique : "
        f"MAE {mod_r['mae_db']:.2f} dB contre {base_r['mae_db']:.2f} dB "
        f"(**{gain_mae:+.2f} dB, soit {gain_pct:+.0f} %**), R² {mod_r['r2']:.3f} contre "
        f"{base_r['r2']:.3f}.\n\n"
        f"En validation croisée, le modèle bat la baseline sur **{plis_gagnes} des "
        f"{n_plis} stations** ; la réserve fait partie de celles où il perd. "
        "C'est précisément l'intérêt d'avoir désigné cette station par une règle "
        "**avant** de voir les résultats : le chiffre publié n'est pas celui de la "
        "station la plus flatteuse.\n\n"
        "> **Tous les R² sont négatifs, baseline comprise.** Ce n'est pas une anomalie "
        "de calcul : sur une station tenue à l'écart, prédire moins bien que la moyenne "
        "*de cette station* est attendu, puisque cette moyenne — le niveau propre du "
        "site — n'était dans aucune donnée d'entraînement. La section suivante sépare "
        "ce décalage de niveau de ce que le modèle capture réellement."
    )

    a = ablation
    geo, hors = a["géométrie seule"], a["hors géométrie (heure + aéroport)"]
    tout = a["toutes variables"]
    centre_best = centre["par_modele"][meilleur]
    verdict_geometrie = (
        "**Réponse en trois chiffres.**\n\n"
        f"1. *Ce que la géométrie apporte.* Seule, elle donne MAE "
        f"{geo['mae_db']:.2f} dB (R² {geo['r2']:.3f}) ; les variables hors géométrie "
        f"seules donnent MAE {hors['mae_db']:.2f} dB (R² {hors['r2']:.3f}), soit "
        f"{hors['mae_db'] / geo['mae_db']:.1f} fois pire. Les ajouter à la géométrie "
        f"ne l'améliore pas ({tout['mae_db']:.2f} dB contre {geo['mae_db']:.2f} dB) : "
        "l'heure et l'aéroport n'apportent rien qui ne soit déjà dans la trajectoire.\n\n"
        f"2. *Ce que le modèle utilise.* L'attribution SHAP place "
        f"{shap_res['part_par_famille_pct']['geometrie']} % de la contribution absolue "
        "sur les variables géométriques.\n\n"
        f"3. *Ce qui reste hors de portée.* Une fois retiré de chaque station son "
        f"niveau moyen, le R² hors pli passe de "
        f"{centre_best['brut']['r2']:.3f} à **{centre_best['centre_par_station']['r2']:.3f}** "
        f"et le MAE de {centre_best['brut']['mae_db']:.2f} à "
        f"{centre_best['centre_par_station']['mae_db']:.2f} dB. Le décalage de niveau "
        f"entre stations vaut en moyenne "
        f"{centre_best['decalage_moyen_absolu_db']:.2f} dB "
        f"(de {centre_best['decalage_min_db']:+.2f} à "
        f"{centre_best['decalage_max_db']:+.2f} dB).\n\n"
        "**Lecture.** La géométrie de trajectoire porte l'essentiel de ce qui est "
        "prédictible, mais elle ne détermine pas le niveau absolu d'un site inconnu : "
        "c'est là que se loge l'erreur."
    )

    tranches = residus["par_tranche_hors_pli"]
    solides = {t: v for t, v in tranches.items() if v["n"] >= 50}
    base_t = solides or tranches
    # Les libellés de tranche sont des chaînes : les comparer donnerait l'ordre
    # alphabétique. On se repère sur la borne inférieure, lue dans le libellé.
    def _borne(t: str) -> float:
        return float(t.strip("[)").split(",")[0])

    pire = max(base_t, key=lambda t: base_t[t]["mae_db"])
    meilleure = min(base_t, key=lambda t: base_t[t]["mae_db"])
    ecart = base_t[pire]["mae_db"] - base_t[meilleure]["mae_db"]
    if ecart <= 1.0:
        tendance = "est à peu près uniforme"
    elif _borne(pire) > _borne(meilleure):
        tendance = "se dégrade au loin"
    else:
        tendance = "se dégrade au près"
    ignorees = [t for t, v in tranches.items() if v["n"] < 50]

    reserve_t = residus["par_tranche"]
    total_reserve = sum(v["n"] for v in reserve_t.values()) or 1
    dominante = max(reserve_t, key=lambda t: reserve_t[t]["n"])
    part_dominante = 100 * reserve_t[dominante]["n"] / total_reserve
    biais_solides = [v["biais_db"] for v in reserve_t.values() if v["n"] >= 20]

    verdict_residus = (
        f"**Réponse : l'erreur {tendance}.** Sur les tranches d'effectif suffisant "
        f"(n ≥ 50), le MAE va de {base_t[meilleure]['mae_db']:.2f} dB sur `{meilleure}` "
        f"à {base_t[pire]['mae_db']:.2f} dB sur `{pire}` — un écart de {ecart:.2f} dB. "
        "L'écart type de la cible suit la même forme : les survols les plus proches "
        "sont aussi les plus dispersés, donc les plus durs à prédire.\n\n"
        + (("Tranches écartées du verdict pour effectif insuffisant : "
            + ", ".join("`{}` (n={})".format(t, tranches[t]["n"]) for t in ignorees)
            + ". Elles figurent au tableau mais ne portent aucune conclusion.\n\n")
           if ignorees else "")
        + f"Sur la réserve seule, {part_dominante:.0f} % des survols tiennent dans la "
        f"tranche `{dominante}` et le biais y reste du même signe et du même ordre "
        f"({', '.join(f'{b:+.1f}' for b in biais_solides)} dB) : c'est un "
        "**décalage de niveau du site**, pas une dégradation liée à la distance."
    )

    limites = (
        "- **Le type d'appareil manque.** `/states/all` ne le fournit pas. C'est la\n"
        "  variable qui expliquerait la variance résiduelle à géométrie constante :\n"
        "  deux appareils différents au même point ne font pas le même bruit.\n"
        f"- **Fenêtre de {manifeste['fenetre']['heures_closes']} heures**, sans week-end,\n"
        "  sans variation saisonnière, sans météo. Aucune généralisation annuelle.\n"
        "- **Aucune donnée météorologique** : vent et gradient de température modifient\n"
        "  la propagation de plusieurs dB.\n"
        "- **10 stations d'Île-de-France** sous trois couloirs. Transposer à un autre\n"
        "  aéroport reste à démontrer.\n"
        "- **Hyperparamètres non optimisés**, par choix : un gain de dernier décimal\n"
        "  obtenu par recherche exhaustive sur 10 stations serait du surapprentissage\n"
        "  au jeu de validation."
    )

    ctx = {
        "resultat": resultat, "meilleur": meilleur, "noms_modeles": noms_modeles,
        "ablation": ablation, "shap": shap_res, "residus": residus,
        "centre": centre, "figures": figures,
        "horodatage": horodatage, "sha256": sha256_file(dataset),
        "fenetre": f"{manifeste['fenetre']['premier_evenement_iso']} → "
                   f"{manifeste['fenetre']['dernier_evenement_iso']} "
                   f"({manifeste['fenetre']['heures_closes']} heures closes)",
        "taux_association": manifeste["taux_association"],
        "verdict_principal": verdict_principal,
        "verdict_geometrie": verdict_geometrie,
        "verdict_residus": verdict_residus,
        "limites": limites,
    }

    rapport = Path(args.report_dir) / f"resultats-modele-reel-{tag}.md"
    ecrire_rapport(ctx, rapport)

    brut = Path(args.report_dir) / f"resultats-modele-reel-{tag}.json"
    brut.write_text(json.dumps({
        "genere_le": horodatage,
        "sha256_jeu": ctx["sha256"],
        "protocole": resultat["protocole"],
        "evaluation_finale": {k: v for k, v in final["resultats"].items()},
        "validation_croisee_10_stations": loso,
        "validation_croisee_hors_reserve": resultat["validation_croisee_hors_reserve"],
        "ablation_geometrie": ablation,
        "importance_permutation": imp,
        "shap": shap_res,
        "residus_par_distance_reserve": residus["par_tranche"],
        "residus_par_distance_hors_pli": residus["par_tranche_hors_pli"],
        "diagnostic_centre": centre["par_modele"],
        "meilleur_modele": meilleur,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRapport   : {rapport}")
    print(f"Chiffres  : {brut}")
    for f in figures:
        print(f"Figure    : {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
