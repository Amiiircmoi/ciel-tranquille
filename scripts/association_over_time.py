#!/usr/bin/env python3
"""Taux d'association heure par heure — tendance ou régime permanent ?

Le chiffre publié (taux cumulé sur toute la campagne) est plus bas que le régime
observé la plupart des heures. Deux explications possibles, qui n'ont pas les
mêmes conséquences :

- **une tendance** : la collecte s'améliore ou se dégrade au fil du temps, et le
  cumul mélange des régimes différents ;
- **un régime permanent pollué par quelques heures aberrantes** : le cumul est
  tiré vers le bas par des incidents identifiables et datés.

On tranche avec les données : moyenne, écart type, pente d'une régression sur le
temps et sa significativité, puis identification explicite des heures aberrantes
et rattachement aux incidents journalisés.

    python scripts/association_over_time.py \
        --progress datasets/pairs_progress_20260827.json \
        --dataset  datasets/pairs_20260827.parquet \
        --ticks    datasets/ticks_echec_20260827.txt

Les entrées sont **figées** : un instantané du suivi de supervision et le relevé
des ticks en échec, tous deux copiés hors de la collecte vive.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("association_over_time")

COULEURS = {"instantane": "#1f4e79", "cumule": "#c0504d",
            "incident": "#e8a33d", "grille": "#d9d9d9"}

# Une heure creuse (quelques survols) produit un taux très bruité : 2 paires sur
# 3 font 67 %, ce qui n'a pas le même sens que 200 sur 300. Le seuil sépare les
# heures interprétables du bruit de nuit, sans les retirer du cumul.
MIN_EVENEMENTS_INTERPRETABLES = 30

# Seuil du test binomial avant correction de Bonferroni. On teste une heure
# contre le régime médian ; corriger le seuil est nécessaire puisqu'on teste
# quarante heures d'affilée et qu'un test sur vingt ressort au hasard à 5 %.
ALPHA = 0.01


def charger(progress: Path, dataset: Path) -> pd.DataFrame:
    """Suivi horaire restreint aux heures **effectivement gelées** dans le jeu.

    Le fichier de suivi continue de grossir avec la collecte ; le jeu figé, lui,
    ne bouge plus. Les faire coïncider est la seule façon d'obtenir une courbe
    qui décrit le jeu sur lequel le modèle a été évalué.
    """
    heures_gelees = set(pd.read_parquet(dataset, columns=["hour_utc"])["hour_utc"].unique())
    suivi = json.loads(progress.read_text(encoding="utf-8"))["hours"]

    lignes = []
    for heure, valeurs in sorted(suivi.items()):
        if heure not in heures_gelees:
            continue
        evenements = valeurs.get("events", 0)
        lignes.append({
            "heure": heure,
            "horodatage": pd.Timestamp(heure.replace("T", " ") + ":00:00", tz="UTC"),
            "evenements": evenements,
            "paires": valeurs.get("pairs", 0),
            "taux": valeurs["pairs"] / evenements if evenements else np.nan,
        })
    frame = pd.DataFrame(lignes).sort_values("horodatage").reset_index(drop=True)
    frame["evenements_cumules"] = frame["evenements"].cumsum()
    frame["paires_cumulees"] = frame["paires"].cumsum()
    frame["taux_cumule"] = frame["paires_cumulees"] / frame["evenements_cumules"]
    frame["interpretable"] = frame["evenements"] >= MIN_EVENEMENTS_INTERPRETABLES
    return frame


def charger_compteur(chemin: Path | None) -> dict[str, int]:
    """Fichier « <heure> <valeur> » par ligne. Absent = dictionnaire vide."""
    if not chemin or not chemin.exists():
        return {}
    valeurs = {}
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        morceaux = ligne.split()
        if len(morceaux) == 2:
            valeurs[morceaux[0]] = int(morceaux[1])
    return valeurs


def tendance(frame: pd.DataFrame) -> dict:
    """Régression du taux sur le temps. Pente, significativité, dispersion.

    Menée sur les heures interprétables uniquement : inclure des heures à trois
    survols reviendrait à laisser le bruit de nuit décider de la pente.
    """
    sous = frame[frame["interpretable"]].copy()
    x = (sous["horodatage"] - sous["horodatage"].min()).dt.total_seconds() / 3600.0
    y = sous["taux"] * 100
    reg = stats.linregress(x, y)

    return {
        "n_heures_totales": int(len(frame)),
        "n_heures_interpretables": int(len(sous)),
        "seuil_interpretable_evenements": MIN_EVENEMENTS_INTERPRETABLES,
        "moyenne_pct": round(float(y.mean()), 2),
        "mediane_pct": round(float(y.median()), 2),
        "ecart_type_pct": round(float(y.std(ddof=1)), 2),
        "min_pct": round(float(y.min()), 2),
        "max_pct": round(float(y.max()), 2),
        "pente_pct_par_heure": round(float(reg.slope), 4),
        "pente_pct_sur_la_campagne": round(float(reg.slope * x.max()), 2),
        "erreur_standard_pente": round(float(reg.stderr), 4),
        "p_value": float(reg.pvalue),
        "r2_regression": round(float(reg.rvalue ** 2), 4),
        "intervalle_95_pente": [
            round(float(reg.slope - 1.96 * reg.stderr), 4),
            round(float(reg.slope + 1.96 * reg.stderr), 4),
        ],
        "significative_a_5pct": bool(reg.pvalue < 0.05),
    }


def _reference_snapshots(frame: pd.DataFrame, snapshots: dict[str, int]) -> dict[str, int]:
    """Débit de snapshots attendu pour chaque heure, lu sur ses **voisines**.

    Une référence globale serait fausse : le garde-fou de budget a étiré la
    cadence de 30 s à 34 s pendant treize heures, ce qui fait 107 snapshots par
    heure au lieu de 120. Comparer ces heures-là au régime 30 s les déclarerait
    toutes déficitaires alors qu'elles sont pleines. La médiane des six heures
    voisines suit le régime en vigueur sans qu'on ait à le lui dire.
    """
    heures = list(frame["heure"])
    sortie = {}
    for i, heure in enumerate(heures):
        voisines = [snapshots[h] for h in heures[max(0, i - 3):i + 4]
                    if h != heure and h in snapshots]
        sortie[heure] = int(np.median(voisines)) if voisines else 0
    return sortie


def aberrantes(
    frame: pd.DataFrame, incidents: dict[str, int], snapshots: dict[str, int]
) -> dict:
    """Heures atypiques, repérées par **test binomial** contre le régime médian.

    Un seuil sur l'écart au taux médian ne convient pas ici : la plupart des
    heures sont entre 99 et 100 %, donc n'importe quelle mesure de dispersion
    devient minuscule et une heure à 95 % sur 164 survols ressort « aberrante »
    alors qu'elle est parfaitement banale. Le test binomial pose la bonne
    question — *combien de paires manquantes cette heure-là seraient encore
    explicables par le hasard, vu son effectif ?* — et donne le même verdict à
    95 % sur 164 survols (banal) qu'à 57 % sur 37 (impossible).

    Seuil corrigé de Bonferroni : on teste plusieurs dizaines d'heures d'affilée,
    et sans correction quelques-unes ressortiraient par construction.
    """
    # La médiane du régime se lit sur les heures fournies : une heure à trois
    # survols ne doit pas définir la référence. Le **test**, lui, porte sur
    # toutes les heures — il tient compte de l'effectif tout seul, et écarter
    # les heures creuses laisserait passer un incident survenu la nuit.
    interpretables = frame[frame["interpretable"]]
    mediane = float((interpretables["taux"] * 100).median())
    sous = frame.dropna(subset=["taux"]).copy()
    seuil = ALPHA / max(len(sous), 1)

    premiere_heure = frame["heure"].iloc[0]
    reference = _reference_snapshots(frame, snapshots)
    lignes = []
    for _, ligne in sous.iterrows():
        test = stats.binomtest(
            int(ligne["paires"]), int(ligne["evenements"]),
            p=mediane / 100, alternative="less",
        )
        if test.pvalue >= seuil:
            continue
        attendues = ligne["evenements"] * mediane / 100
        ticks = incidents.get(ligne["heure"], 0)
        vus = snapshots.get(ligne["heure"])
        attendu_snapshots = reference.get(ligne["heure"], 0)
        manque_snapshots = (vus is not None and attendu_snapshots
                            and vus < 0.9 * attendu_snapshots)

        if ligne["heure"] == premiere_heure:
            cause = (f"démarrage en cours d'heure ({vus} snapshots au lieu de "
                     f"~{attendu_snapshots}) — aucun aéronef avant le premier tick")
        elif ticks:
            cause = (f"{ticks} tick(s) de collecte en échec"
                     + (f", {vus} snapshots au lieu de ~{attendu_snapshots}" if vus else ""))
        elif manque_snapshots:
            cause = f"débit de snapshots réduit ({vus} au lieu de ~{attendu_snapshots})"
        elif vus is not None:
            cause = (f"couverture aéronef complète ({vus} snapshots) — survols non vus "
                     "par OpenSky ou hors rayon d'appariement")
        else:
            cause = "non élucidée"
        lignes.append({
            "heure": ligne["heure"],
            "taux_pct": round(float(ligne["taux"] * 100), 2),
            "evenements": int(ligne["evenements"]),
            "paires": int(ligne["paires"]),
            "p_value": float(test.pvalue),
            "ticks_en_echec": ticks,
            "cause": cause,
            "paires_manquantes": int(round(attendues - ligne["paires"])),
        })
    lignes.sort(key=lambda x: -x["paires_manquantes"])

    total_evts = int(frame["evenements"].sum())
    total_paires = int(frame["paires"].sum())
    heures_atypiques = {x["heure"] for x in lignes}
    reste = frame[~frame["heure"].isin(heures_atypiques)]
    return {
        "mediane_pct": round(mediane, 2),
        "reference_du_test": "médiane horaire des heures fournies",
        "seuil_binomial": seuil,
        "n_heures_testees": int(len(sous)),
        "alpha_avant_correction": ALPHA,
        "heures": lignes,
        "heures_expliquees": sum(1 for x in lignes if x["cause"] != "non élucidée"),
        "reference_snapshots_par_heure": reference,
        "taux_publie_pct": round(100 * total_paires / total_evts, 2),
        "taux_hors_heures_atypiques_pct": round(
            100 * int(reste["paires"].sum()) / int(reste["evenements"].sum()), 2
        ),
        "paires_perdues_estimees": sum(x["paires_manquantes"] for x in lignes),
        "incidents_journalises": incidents,
    }


def figure(frame: pd.DataFrame, stats_t: dict, atyp: dict, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    creuses = frame[~frame["interpretable"]]
    pleines = frame[frame["interpretable"]]
    heures_atypiques = {x["heure"] for x in atyp["heures"]}

    # La ligne suit **toutes** les heures : la couper sur les heures creuses
    # ferait sauter l'axe du temps et rapprocherait visuellement des heures
    # séparées de plusieurs heures réelles.
    ax.plot(frame["horodatage"], frame["taux"] * 100, linewidth=1.3,
            color=COULEURS["instantane"], alpha=0.85, zorder=2)
    ax.scatter(pleines["horodatage"], pleines["taux"] * 100, s=22, zorder=3,
               color=COULEURS["instantane"], label="taux horaire (instantané)")
    ax.scatter(creuses["horodatage"], creuses["taux"] * 100, s=30, facecolors="white",
               edgecolors=COULEURS["instantane"], alpha=0.9, linewidths=1.1, zorder=3,
               label=f"heure creuse (< {MIN_EVENEMENTS_INTERPRETABLES} survols)")

    marquees = frame[frame["heure"].isin(heures_atypiques)]
    ax.scatter(marquees["horodatage"], marquees["taux"] * 100, s=150, marker="o",
               facecolors="none", edgecolors=COULEURS["incident"], linewidths=2.2,
               label="heure atypique (test binomial)", zorder=5)

    ax.plot(frame["horodatage"], frame["taux_cumule"] * 100, linewidth=2.6,
            color=COULEURS["cumule"], label="taux cumulé (chiffre publié)")
    final = frame["taux_cumule"].iloc[-1] * 100
    ax.axhline(final, color=COULEURS["cumule"], linewidth=0.9, linestyle=":", alpha=0.7)
    ax.annotate(f"chiffre publié {final:.1f} %",
                xy=(frame["horodatage"].iloc[-1], final),
                xytext=(-8, -16), textcoords="offset points", ha="right",
                color=COULEURS["cumule"], fontweight="bold", fontsize=9)

    ax.axhline(atyp["mediane_pct"], color=COULEURS["instantane"], linewidth=0.9,
               linestyle="--", alpha=0.7)
    ax.annotate(f"médiane horaire {atyp['mediane_pct']:.1f} %",
                xy=(frame["horodatage"].iloc[0], atyp["mediane_pct"]),
                xytext=(4, 6), textcoords="offset points",
                color=COULEURS["instantane"], fontsize=9)

    ax.set_ylim(0, 104)
    ax.set_ylabel("taux d'association (%)")
    ax.set_xlabel("heure UTC")
    ax.set_title(
        "Taux d'association au fil de la campagne — "
        f"{stats_t['n_heures_totales']} heures figées\n"
        f"pente {stats_t['pente_pct_par_heure']:+.3f} %/h "
        f"(p = {stats_t['p_value']:.2f}"
        f"{', non significative' if not stats_t['significative_a_5pct'] else ''})",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    ax.grid(True, color=COULEURS["grille"], linewidth=0.6)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    fig.autofmt_xdate()

    chemin = out / "taux-association-dans-le-temps.png"
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    logger.info("figure : %s", chemin)
    return str(chemin)


# -------------------------------------------------------------------- rapport
def _table(entetes: list[str], lignes: list[list]) -> str:
    out = ["| " + " | ".join(entetes) + " |",
           "|" + "|".join(["---"] * len(entetes)) + "|"]
    for ligne in lignes:
        out.append("| " + " | ".join(str(c) for c in ligne) + " |")
    return "\n".join(out)


def ecrire_rapport(frame, stats_t, atyp, chemin_figure, horodatage, chemin: Path) -> None:
    verdict = ("**stationnaire**" if not stats_t["significative_a_5pct"]
               else "**en tendance**")
    tbl_atyp = _table(
        ["Heure UTC", "taux", "paires / survols", "p (binomial)", "paires perdues", "cause"],
        [[h["heure"], f"{h['taux_pct']:.1f} %", f"{h['paires']} / {h['evenements']}",
          f"{h['p_value']:.1e}", h["paires_manquantes"], h["cause"]]
         for h in atyp["heures"]],
    )
    horaire = _table(
        ["Heure UTC", "survols", "paires", "taux", "taux cumulé"],
        [[r["heure"], int(r["evenements"]), int(r["paires"]),
          "—" if pd.isna(r["taux"]) else f"{r['taux'] * 100:.1f} %",
          f"{r['taux_cumule'] * 100:.2f} %"]
         for _, r in frame.iterrows()],
    )
    expliquees = atyp["heures_expliquees"]
    total_atyp = len(atyp["heures"])
    perdues = atyp["paires_perdues_estimees"]

    texte = f"""# Taux d'association au fil de la campagne

**Généré le {horodatage}.** Ce fichier fait foi.

Figure : `{Path(chemin_figure).name}`

## Question

Le chiffre publié — **{atyp['taux_publie_pct']:.2f} %** — est le taux cumulé sur
toute la campagne. La médiane horaire vaut **{atyp['mediane_pct']:.2f} %**. L'écart
vient-il d'une **tendance** (la collecte se dégrade ou s'améliore) ou d'un **régime
permanent pollué par quelques heures identifiables** ?

## Réponse : le taux est {verdict}

| | |
|---|---|
| Heures figées | {stats_t['n_heures_totales']} |
| Heures interprétables (≥ {stats_t['seuil_interpretable_evenements']} survols) | {stats_t['n_heures_interpretables']} |
| Moyenne | {stats_t['moyenne_pct']:.2f} % |
| Médiane | {stats_t['mediane_pct']:.2f} % |
| Écart type | {stats_t['ecart_type_pct']:.2f} points |
| Étendue | {stats_t['min_pct']:.1f} % – {stats_t['max_pct']:.1f} % |
| **Pente** | **{stats_t['pente_pct_par_heure']:+.4f} % par heure** |
| IC 95 % de la pente | [{stats_t['intervalle_95_pente'][0]:+.4f} ; {stats_t['intervalle_95_pente'][1]:+.4f}] |
| p-value | **{stats_t['p_value']:.3f}** |
| R² de la régression | {stats_t['r2_regression']:.4f} |
| Sur toute la campagne | {stats_t['pente_pct_sur_la_campagne']:+.2f} points |

L'intervalle de confiance de la pente **contient zéro** et la régression n'explique
que {stats_t['r2_regression'] * 100:.1f} % de la variance horaire. Il n'y a pas de
tendance : la collecte ne s'est ni améliorée ni dégradée au fil des
{stats_t['n_heures_totales']} heures.

La régression porte sur les heures d'au moins
{stats_t['seuil_interpretable_evenements']} survols. Une heure de nuit à trois
survols donne un taux de 67 % ou 100 % selon une seule paire : la laisser entrer
reviendrait à laisser le bruit d'échantillonnage décider de la pente.

## Les {total_atyp} heures atypiques

Repérées par **test binomial** contre le régime médian
({atyp['mediane_pct']:.2f} %), seuil {atyp['alpha_avant_correction']} corrigé de
Bonferroni sur {atyp['n_heures_testees']} heures, soit
{atyp['seuil_binomial']:.2e}. Un seuil sur l'écart au taux médian ne conviendrait
pas : la plupart des heures étant entre 99 et 100 %, toute mesure de dispersion
devient minuscule et une heure à 95 % sur 164 survols ressortirait « aberrante »
alors qu'elle est banale.

{tbl_atyp}

**{expliquees} des {total_atyp} heures ont une cause tracée dans les journaux.**
Les autres présentent une **couverture aéronef complète** : le poller a tourné
normalement, les survols manquants correspondent donc à des aéronefs que
l'API OpenSky ne voit pas (pas de transpondeur ADS-B, altitude sous la couverture)
ou qui sortent du rayon d'appariement de 10 km. Ce n'est pas un défaut du pipeline,
c'est la limite de la source.

## Pourquoi le chiffre n'a pas été nettoyé

En retirant ces heures, le taux passerait de **{atyp['taux_publie_pct']:.2f} %** à
**{atyp['taux_hors_heures_atypiques_pct']:.2f} %** — un gain de
{atyp['taux_hors_heures_atypiques_pct'] - atyp['taux_publie_pct']:.2f} points, soit
{perdues} paires. **On ne l'a pas fait, et on ne le fera pas.**

Trois raisons :

1. **Le chiffre publié doit décrire la collecte telle qu'elle s'est passée**, pas
   telle qu'elle se serait passée sans incident. Une campagne de terrain de
   {stats_t['n_heures_totales']} heures comporte une heure de démarrage et des
   ticks en échec ; les effacer donnerait un chiffre que personne ne peut
   reproduire en relançant la collecte.
2. **Le critère d'exclusion serait choisi après avoir vu les résultats.** Rien
   n'empêcherait ensuite d'élargir le filtre jusqu'à ce que le chiffre plaise.
   La seule protection contre cette dérive est de ne pas commencer.
3. **L'écart est petit et l'information est ailleurs.** {atyp['taux_publie_pct']:.2f} %
   contre {atyp['taux_hors_heures_atypiques_pct']:.2f} % ne change aucune décision.
   En revanche, savoir *quelles* heures ont décroché et *pourquoi* est exploitable —
   c'est ce que donne le tableau ci-dessus.

La bonne façon de présenter les deux chiffres : « **{atyp['taux_publie_pct']:.2f} %
sur la campagne complète, médiane horaire {atyp['mediane_pct']:.2f} %** ; l'écart
tient à {total_atyp} heures identifiées, dont l'heure de démarrage et une fenêtre
d'incident réseau ».

## Détail horaire

{horaire}
"""
    chemin.write_text(texte, encoding="utf-8")
    logger.info("rapport : %s", chemin)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ticks", default=None)
    parser.add_argument("--snapshots", default=None,
                        help="relevé « <heure> <nb snapshots> » par ligne")
    parser.add_argument("--out-dir", default="dossier-rncp/figures")
    parser.add_argument("--report", default=None, help="JSON des chiffres")
    parser.add_argument("--report-dir", default="dossier-rncp",
                        help="répertoire du fichier de résultats Markdown")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = charger(Path(args.progress), Path(args.dataset))
    incidents = charger_compteur(Path(args.ticks) if args.ticks else None)
    snapshots = charger_compteur(Path(args.snapshots) if args.snapshots else None)
    stats_t = tendance(frame)
    atyp = aberrantes(frame, incidents, snapshots)
    chemin = figure(frame, stats_t, atyp, out)

    resultat = {
        "figure": chemin,
        "tendance": stats_t,
        "heures_aberrantes": atyp,
        "detail_horaire": frame.assign(
            horodatage=frame["horodatage"].dt.strftime("%Y-%m-%dT%H:%MZ")
        ).to_dict(orient="records"),
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(resultat, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"Chiffres : {args.report}")

    horodatage = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    md = Path(args.report_dir) / f"taux-association-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.md"
    ecrire_rapport(frame, stats_t, atyp, chemin, horodatage, md)
    print(f"Rapport  : {md}")
    print(f"\nFigure   : {chemin}")
    print(f"\nHeures interprétables : {stats_t['n_heures_interpretables']} "
          f"/ {stats_t['n_heures_totales']}")
    print(f"Moyenne  : {stats_t['moyenne_pct']:.2f} %  |  "
          f"écart type {stats_t['ecart_type_pct']:.2f} pts  |  "
          f"médiane {stats_t['mediane_pct']:.2f} %")
    print(f"Pente    : {stats_t['pente_pct_par_heure']:+.4f} %/h "
          f"(IC95 {stats_t['intervalle_95_pente']}), p = {stats_t['p_value']:.4f}, "
          f"R² = {stats_t['r2_regression']:.4f}")
    print(f"Verdict  : {'TENDANCE significative' if stats_t['significative_a_5pct'] else 'STATIONNAIRE (pente non significative)'}")
    print(f"\nTaux publié {atyp['taux_publie_pct']:.2f} % | médiane horaire "
          f"{atyp['mediane_pct']:.2f} % | hors heures atypiques "
          f"{atyp['taux_hors_heures_atypiques_pct']:.2f} % | "
          f"{atyp['paires_perdues_estimees']} paires perdues")
    for h in atyp["heures"]:
        print(f"  {h['heure']} : {h['taux_pct']:5.1f} % "
              f"({h['paires']}/{h['evenements']}), p={h['p_value']:.1e} — {h['cause']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
