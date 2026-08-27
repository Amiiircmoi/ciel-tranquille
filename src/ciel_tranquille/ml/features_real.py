"""Features de la prédiction **réelle co-localisée** : géométrie de vol → LAmax.

À ne pas confondre avec `features.py`, qui sert le pipeline synthétique agrégé
(LAeq horaire par zone). Ici une ligne = **un survol** : un événement de bruit
Bruitparif apparié à un aéronef OpenSky. La cible est le `LAmax` de ce survol.

Garde-fou anti-fuite de cible — deux familles à exclure, pour deux raisons
différentes :

1. **Les autres mesures du même événement acoustique** (`laeq`, `sel`,
   `nrj_laeq`, `duration_s`) sont des transformations du signal qui a produit
   `max_laeq`. Les donner en entrée, c'est prédire le bruit à partir du bruit :
   le R² serait excellent et le modèle sans valeur.
2. **L'identité de la station** (`station`, `station_lat`, `station_lon`) est
   exclue pour une raison distincte, propre au protocole d'évaluation : la
   station de test est **entièrement inconnue** à l'entraînement. Un modèle
   autorisé à mémoriser « à Gonesse c'est fort » ne généraliserait pas, et le
   score obtenu sur une station retenue à l'écart serait bâti sur une colonne
   dont la valeur n'a jamais été vue. On veut apprendre une **relation
   physique transportable**, pas un annuaire de stations.

`icao24` et `callsign` sont écartés pour la même raison : identifiants, pas
propriétés physiques. Le type d'appareil — la variable qui manque réellement,
et qui expliquerait la variance résiduelle à géométrie constante — n'est pas
fourni par `/states/all` ; c'est une limite assumée du jeu de données, à écrire
telle quelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "max_laeq"

# Distance de la baseline physique : la seule colonne qu'elle consomme.
DISTANCE_COL = "slant_km"

NUMERIC_FEATURES = [
    "slant_km",
    "log_slant",           # forme sous laquelle la physique est linéaire
    "horiz_km",
    "altitude_m",
    "elevation_deg",       # hauteur angulaire de l'avion au-dessus de l'horizon
    "velocity_m_s",
    "cos_aspect",          # géométrie d'émission : l'avion vient-il vers la station ?
    "hour_utc_num",
    "is_night",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["airport"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Interdits en entrée : fuite de cible (bloc 1) ou identifiants (bloc 2).
_INTERDITS = [
    "max_laeq", "laeq", "sel", "nrj_laeq", "duration_s",
    "event_id", "station", "station_lat", "station_lon", "icao24", "callsign",
]
# Le gel conserve le contexte de l'événement à côté de la géométrie ; les
# colonnes présentes des deux côtés y portent le suffixe `_evt`. Une cible
# recopiée sous un autre nom reste une cible : la liste couvre les deux formes.
LEAKAGE_COLS = [*_INTERDITS, *(f"{c}_evt" for c in _INTERDITS)]

# Colonne de découpe : jamais une feature, toujours la clé du split.
GROUP_COL = "station"


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Relèvement station→avion, vectorisé (degrés depuis le Nord)."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Dérive les features de la table de paires figée. Aucun accès disque.

    Purement fonctionnel : mêmes paires en entrée ⇒ mêmes features en sortie.
    C'est la contrepartie du gel de l'instantané — figer les données ne sert à
    rien si la préparation, elle, dépend de l'horloge ou de la configuration.
    """
    df = pairs.copy()

    df["log_slant"] = np.log10(np.maximum(df["slant_km"].astype(float), 0.05))

    # Élévation : angle entre l'horizon de la station et l'aéronef. Deux survols
    # à même distance oblique mais l'un à la verticale et l'autre rasant ne se
    # propagent pas pareil (effet de sol, absorption sur trajet long).
    slant_m = np.maximum(df["slant_km"].astype(float) * 1000.0, 1.0)
    df["elevation_deg"] = np.degrees(
        np.arcsin(np.clip(df["altitude_m"].astype(float) / slant_m, -1.0, 1.0))
    )

    # Aspect : angle entre le cap de l'avion et la direction station→avion.
    # cos = +1 : l'avion s'éloigne de la station ; -1 : il vient vers elle.
    releve = _bearing_deg(
        df["station_lat"].astype(float), df["station_lon"].astype(float),
        df["aircraft_lat"].astype(float), df["aircraft_lon"].astype(float),
    )
    df["cos_aspect"] = np.cos(np.radians(df["heading_deg"].astype(float) - releve))

    horodatage = pd.to_datetime(df["max_ts_unix"].astype(float), unit="s", utc=True)
    df["hour_utc_num"] = horodatage.dt.hour + horodatage.dt.minute / 60.0
    df["is_night"] = ((horodatage.dt.hour < 6) | (horodatage.dt.hour >= 22)).astype(int)
    df["is_weekend"] = (horodatage.dt.dayofweek >= 5).astype(int)

    df["airport"] = df["airport"].astype(str)
    return df


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Écarte les lignes inexploitables. Le motif du retrait est journalisable."""
    besoins = [TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, GROUP_COL]
    presents = [c for c in besoins if c in df.columns]
    return df.dropna(subset=presents).reset_index(drop=True)


def check_no_leakage(columns) -> None:
    """Échec bruyant si une colonne interdite atteint la matrice de features.

    Ce contrôle vaut mieux qu'un commentaire : la liste des features est amenée
    à bouger, et une fuite de cible ne se voit pas dans les métriques — elle les
    rend seulement trop belles.
    """
    fautes = sorted(set(columns) & set(LEAKAGE_COLS))
    if fautes:
        raise ValueError(
            "Colonnes interdites dans les features (fuite de cible ou "
            f"identifiant) : {', '.join(fautes)}"
        )
