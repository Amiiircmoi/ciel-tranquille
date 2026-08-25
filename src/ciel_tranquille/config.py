"""Configuration centralisée, chargée depuis l'environnement / `.env`.

Les secrets OpenSky ne sont JAMAIS codés en dur : ils proviennent de variables
d'environnement (fichier `.env` gitignoré, injecté par `env_file` en conteneur).
Voir `.env.example`.

**Contrat d'hébergement (déploiement conteneurisé).** Toute écriture se fait sous
`CIEL_DATA_DIR` (le volume monté, `/data` dans l'image) :

    $CIEL_DATA_DIR/
      landing/date=YYYY-MM-DD/states_<ts>.parquet  snapshots d'états (idempotents)
      raw/noise_events/station=…/date=…/           événements de survol Bruitparif
      curated/                                      compaction horaire, DuckDB, métriques
      status/                                       heartbeat + status.json (supervision)

En développement local, `CIEL_DATA_DIR` n'est pas défini : on retombe sur
`CT_DATA_DIR` (défaut `data/` dans le dépôt), ce qui laisse la boucle de dev et
les tests inchangés.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Racine du dépôt (src/ciel_tranquille/config.py -> remonte de 3 niveaux)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Liste de stations livrée avec le dépôt (surchargeable, cf. `stations.py`).
REPO_STATIONS_FILE = REPO_ROOT / "config" / "stations.json"

# Aire maximale d'une bbox `/states/all` restant dans la tranche à 1 crédit.
# https://openskynetwork.github.io/opensky-api/ — ≤ 25 deg² = 1 crédit/appel.
MAX_BBOX_AREA_DEG2 = 25.0


class StationConfigError(RuntimeError):
    """Configuration de stations absente, illisible ou sans liste active.

    On échoue **bruyamment** plutôt que de retomber sur une liste par défaut :
    une collecte qui tourne six jours sur les mauvaises stations — ou sur aucune —
    est un échec silencieux, le pire mode de défaillance de ce système.
    """


@dataclass(frozen=True)
class Station:
    """Station de mesure Bruitparif retenue (sous un couloir aérien clair)."""

    measurement_id: str
    latitude: float
    longitude: float
    airport: str  # couloir dominant (CDG / ORY / LBG)
    label: str = ""  # libellé lisible, facultatif (affichage/doc)

    def to_dict(self) -> dict:
        return {
            "measurement_id": self.measurement_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "airport": self.airport,
            "label": self.label,
        }


# Stations socle : identifiants confirmés via `/sites` et **validés par la
# collecte réelle** (GATE 1b). Elles ne servent PAS de repli implicite à la
# collecte — `Settings.stations` exige un fichier de configuration avec une liste
# active explicite. Elles servent de socle au sondage
# (`python -m ciel_tranquille.ingest.stations probe`) et de décor au générateur
# synthétique, qui a besoin de stations fixes pour être reproductible.
STATIONS: tuple[Station, ...] = (
    Station("95500-GONESSE-MEDIATHEQUE-M", 48.985000, 2.447118, "CDG", "Gonesse — Médiathèque"),
    Station("94290-VILLENEUVE-LE-ROI-NOBLECOURT", 48.730312, 2.427967, "ORY", "Villeneuve-le-Roi — Noblecourt"),
    Station("95390-ST-PRIX-MAIRIE-M", 49.006393, 2.263266, "CDG", "Saint-Prix — Mairie"),
)


def _station_from_entry(entry: dict) -> Station:
    return Station(
        measurement_id=str(entry["measurement_id"]),
        latitude=float(entry["latitude"]),
        longitude=float(entry["longitude"]),
        airport=str(entry.get("airport", "")),
        label=str(entry.get("label", "")),
    )


def load_stations_file(path: Path) -> tuple[Station, ...]:
    """Charge les stations **actives** depuis le JSON de configuration.

    Structure attendue :

        {"active": ["ID-A", "ID-B"], "stations": [{"measurement_id": "ID-A", ...}]}

    `active` est **obligatoire et explicite** : c'est lui qui décide de ce qui est
    collecté, renseigné à partir du classement par rendement du sondage. Sans ce
    champ, on refuse de deviner. Toute anomalie lève `StationConfigError` — un
    démarrage qui échoue est infiniment préférable à six jours de collecte vide.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StationConfigError(f"Fichier de stations illisible ({path}) : {exc}") from exc
    if not isinstance(payload, dict):
        raise StationConfigError(f"Fichier de stations mal formé ({path}) : objet JSON attendu.")

    entries = payload.get("stations")
    if not entries:
        raise StationConfigError(f"Aucune station décrite dans {path}.")
    try:
        catalogue = {str(e["measurement_id"]): _station_from_entry(e) for e in entries}
    except (KeyError, TypeError, ValueError) as exc:
        raise StationConfigError(f"Station mal décrite dans {path} : {exc}") from exc

    active = payload.get("active")
    if active is None:
        raise StationConfigError(
            f"Champ 'active' absent de {path}. La liste des stations collectées doit "
            "être explicite : lancez le sondage "
            "`python -m ciel_tranquille.ingest.stations probe --write` puis reportez "
            "le classement par rendement dans ce champ."
        )
    if not isinstance(active, list) or not active:
        raise StationConfigError(f"Champ 'active' vide ou mal formé dans {path}.")

    missing = [a for a in active if a not in catalogue]
    if missing:
        raise StationConfigError(
            f"Stations actives absentes du catalogue de {path} : {', '.join(missing)}."
        )
    return tuple(catalogue[a] for a in active)


def load_stations_catalogue(path: Path) -> tuple[Station, ...]:
    """Toutes les stations décrites par le fichier, actives ou non (outillage)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_station_from_entry(e) for e in payload.get("stations", []))


class BoundingBox(BaseSettings):
    """Bounding box WGS84 de captation (filtre OpenSky `/states/all`)."""

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    def as_params(self) -> dict[str, float]:
        return {
            "lamin": self.lat_min,
            "lomin": self.lon_min,
            "lamax": self.lat_max,
            "lomax": self.lon_max,
        }

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    @property
    def area_deg2(self) -> float:
        """Aire en degrés carrés — l'unité de tarification OpenSky."""
        return abs(self.lat_max - self.lat_min) * abs(self.lon_max - self.lon_min)


class Settings(BaseSettings):
    """Paramètres applicatifs.

    Trois préfixes d'environnement, par ordre d'apparition historique :
    `OPENSKY_` (secrets), `CT_` (pipeline), `CIEL_` (contrat d'hébergement et
    garde-fous de la collecte longue durée).
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets OpenSky (OAuth2 client credentials) ---
    # `repr=False` : sans cela, le `repr()` de Settings (traces pytest, logs
    # d'exception, rapports CI) recracherait le secret en clair.
    opensky_client_id: str = Field(default="", alias="OPENSKY_CLIENT_ID", repr=False)
    opensky_client_secret: str = Field(default="", alias="OPENSKY_CLIENT_SECRET", repr=False)

    # --- Bounding box COMBINÉE (une seule requête pour toutes les stations) ---
    # Une bbox de 25 deg² coûte le même crédit unique qu'une bbox de 1 deg² :
    # on couvre donc largement les couloirs CDG / Orly / Le Bourget, ce qui permet
    # d'héberger 8-10 stations dans une seule requête. Défaut ≈ 3,8 deg².
    bbox_lat_min: float = Field(default=48.10, alias="CT_BBOX_LAT_MIN")
    bbox_lon_min: float = Field(default=1.30, alias="CT_BBOX_LON_MIN")
    bbox_lat_max: float = Field(default=49.50, alias="CT_BBOX_LAT_MAX")
    bbox_lon_max: float = Field(default=4.00, alias="CT_BBOX_LON_MAX")

    # --- Cadence micro-batch (s) : 30 s ≈ 2 880 appels/j < plafond 4000 crédits ---
    poll_interval_s: int = Field(default=30, alias="CT_POLL_INTERVAL_S")

    # --- Budget crédits OpenSky (allocation QUOTIDIENNE, header x-rate-limit-remaining) ---
    daily_credit_budget: int = Field(default=4000, alias="CT_DAILY_CREDIT_BUDGET")
    # Plancher de sécurité : le poller s'arrête si le restant passe sous ce seuil.
    credit_floor: int = Field(default=200, alias="CT_CREDIT_FLOOR")
    # Cadence plafond (s) quand le garde-fou budget ralentit la collecte.
    max_poll_interval_s: int = Field(default=300, alias="CIEL_MAX_POLL_INTERVAL_S")
    # Au plancher, délai avant de relire le solde. Le quota OpenSky se
    # réapprovisionne en cours de journée : attendre minuit ferait perdre des
    # heures de collecte pour rien.
    credit_recheck_s: int = Field(default=600, alias="CIEL_CREDIT_RECHECK_S")

    # --- OAuth2 : durée de vie du jeton (30 min côté OpenSky) et marge de refresh ---
    token_ttl_s: int = Field(default=1800, alias="CIEL_TOKEN_TTL_S")
    token_refresh_margin_s: int = Field(default=30, alias="CIEL_TOKEN_REFRESH_MARGIN_S")

    # --- Identité sortante (obligatoire : IP partagée avec une production tierce) ---
    user_agent: str = Field(default="", alias="CIEL_USER_AGENT")
    contact: str = Field(default="", alias="CIEL_CONTACT")

    # --- Bruitparif : espacement et back-off entre fenêtres horaires ---
    # Plancher dur d'1 s entre appels (cf. `polite_pause_s`) : non négociable,
    # l'IP sortante est partagée avec une production tierce.
    bruitparif_pause_s: float = Field(default=1.5, alias="CIEL_BRUITPARIF_PAUSE_S")
    bruitparif_max_retries: int = Field(default=3, alias="CIEL_BRUITPARIF_MAX_RETRIES")
    bruitparif_backoff_base_s: float = Field(default=5.0, alias="CIEL_BRUITPARIF_BACKOFF_BASE_S")

    # --- Anti-troncature : `/events` plafonne à ~50 réponses, biaisées vers le
    # début de l'intervalle. Au-delà de `cap × ratio`, la fenêtre est redécoupée. ---
    events_page_cap: int = Field(default=50, alias="CIEL_EVENTS_PAGE_CAP")
    events_saturation_ratio: float = Field(default=0.9, alias="CIEL_EVENTS_SATURATION_RATIO")
    events_min_window_minutes: int = Field(default=15, alias="CIEL_EVENTS_MIN_WINDOW_MIN")

    # --- Sondage des stations (classement par rendement) ---
    probe_window_hours: int = Field(default=6, alias="CIEL_PROBE_WINDOW_H")
    probe_window_start_utc: int = Field(default=6, alias="CIEL_PROBE_WINDOW_START_UTC")
    probe_max_candidates: int = Field(default=40, alias="CIEL_PROBE_MAX_CANDIDATES")
    max_active_stations: int = Field(default=10, alias="CIEL_MAX_ACTIVE_STATIONS")

    # --- Santé de la source bruit (contrôle distinct du poller avion) ---
    noise_max_silence_s: int = Field(default=7200, alias="CIEL_NOISE_MAX_SILENCE_S")
    noise_day_start_utc: int = Field(default=5, alias="CIEL_NOISE_DAY_START_UTC")
    noise_day_end_utc: int = Field(default=21, alias="CIEL_NOISE_DAY_END_UTC")

    # --- Compaction horaire : on ne compacte pas les heures encore alimentées ---
    compact_lag_h: int = Field(default=2, alias="CIEL_COMPACT_LAG_H")
    # --- Comptage des paires : une heure n'est figée qu'une fois le bruit publié
    # ET collecté (latence Bruitparif ~1 h + intervalle du collecteur). ---
    pairs_lag_h: int = Field(default=6, alias="CIEL_PAIRS_LAG_H")
    # Distance oblique max (km) pour qu'une association compte comme paire propre.
    pairs_max_slant_km: float = Field(default=10.0, alias="CIEL_PAIRS_MAX_SLANT_KM")

    # --- Verdict de supervision : âge max du dernier snapshot avant KO ---
    status_max_snapshot_age_s: int = Field(default=300, alias="CIEL_STATUS_MAX_SNAPSHOT_AGE_S")

    # --- Mode d'ingestion : "live" (API réelle) | "replay" (snapshots) ---
    ingest_mode: str = Field(default="replay", alias="CT_INGEST_MODE")

    # --- Chemins ---
    # `CIEL_DATA_DIR` (volume monté, prioritaire) puis `CT_DATA_DIR` (dev local).
    ciel_data_dir: str = Field(default="", alias="CIEL_DATA_DIR")
    data_dir: str = Field(default="data", alias="CT_DATA_DIR")
    # Sorties annexes (captures, exports d'analyse) — routables hors dépôt via env.
    output_dir: str = Field(default="outputs", alias="CT_OUTPUT_DIR")
    # Liste de stations Bruitparif (JSON). Vide -> résolution automatique.
    stations_file: str = Field(default="", alias="CIEL_STATIONS_FILE")

    # --- Endpoints OpenSky ---
    opensky_states_url: str = "https://opensky-network.org/api/states/all"
    opensky_token_url: str = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/"
        "protocol/openid-connect/token"
    )

    # --- Bruitparif Survol (token public scrapé depuis la page de l'app) ---
    bruitparif_app_url: str = "https://survol.bruitparif.fr/"
    bruitparif_api_url: str = "https://rumeurengine.bruitparif.fr"

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(
            lat_min=self.bbox_lat_min,
            lon_min=self.bbox_lon_min,
            lat_max=self.bbox_lat_max,
            lon_max=self.bbox_lon_max,
        )

    @property
    def stations_path(self) -> Path | None:
        """Fichier de stations effectif, par ordre de priorité.

        1. `CIEL_STATIONS_FILE` (explicite) ;
        2. `$CIEL_DATA_DIR/stations.json` (produit par le sondage au déploiement,
           seul emplacement inscriptible en conteneur) ;
        3. `config/stations.json` du dépôt, cherché à deux endroits :
           l'arborescence source (`REPO_ROOT`) **et** le répertoire de travail.

        Le double emplacement n'est pas une ceinture et bretelles : une fois le
        paquet installé (image Docker), `REPO_ROOT` pointe dans `site-packages`,
        pas sur le `/app` où la configuration a été copiée. Sans le repli sur le
        répertoire de travail, le conteneur ne trouve aucune station et refuse de
        démarrer.
        """
        if self.stations_file:
            return Path(self.stations_file)
        for candidate in (
            self.data_path / "stations.json",
            REPO_STATIONS_FILE,
            Path.cwd() / "config" / "stations.json",
        ):
            if candidate.exists():
                return candidate
        return None

    @property
    def stations(self) -> tuple[Station, ...]:
        """Stations actives, lues dans le fichier de configuration.

        Aucun repli implicite : si la configuration manque ou n'a pas de liste
        active, on lève `StationConfigError`. Voir `load_stations_file`.
        """
        path = self.stations_path
        if path is None:
            raise StationConfigError(
                "Aucun fichier de stations trouvé. Attendu : CIEL_STATIONS_FILE, "
                "$CIEL_DATA_DIR/stations.json, ou config/stations.json. Lancez "
                "`python -m ciel_tranquille.ingest.stations probe --write`."
            )
        return load_stations_file(path)

    @property
    def http_user_agent(self) -> str:
        """User-Agent identifiable envoyé à OpenSky et Bruitparif.

        L'IP sortante est partagée avec une production tierce : un opérateur doit
        pouvoir nous identifier et nous joindre avant d'envisager un blocage.
        """
        if self.user_agent:
            return self.user_agent
        contact = self.contact or "contact-non-renseigne"
        return f"ciel-tranquille/0.1 (analyse du bruit aerien; +{contact})"

    @property
    def has_contact(self) -> bool:
        return bool(self.user_agent or self.contact)

    @property
    def polite_pause_s(self) -> float:
        """Espacement effectif entre deux appels Bruitparif, plancher 1 s.

        Le plancher est **dur** : une configuration qui descendrait sous la
        seconde serait ignorée. Le coût d'une collecte lente est quelques minutes ;
        le coût d'un blocage d'IP est la production tierce qui partage cette IP.
        """
        return max(1.0, float(self.bruitparif_pause_s))

    @property
    def events_saturation_threshold(self) -> int:
        """Nombre d'événements à partir duquel une fenêtre est jugée tronquée."""
        return max(1, int(self.events_page_cap * self.events_saturation_ratio))

    @property
    def noise_health_path(self) -> Path:
        """Santé de la source bruit (token, dernier passage) — écrite par le collecteur."""
        return self.status_dir / "noise_health.json"

    @property
    def has_credentials(self) -> bool:
        return bool(self.opensky_client_id and self.opensky_client_secret)

    @property
    def data_path(self) -> Path:
        raw = self.ciel_data_dir or self.data_dir
        p = Path(raw)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else REPO_ROOT / self.output_dir

    @property
    def raw_dir(self) -> Path:
        """Racine historique (collectes antérieures) — toujours lue, plus écrite."""
        return self.data_path / "raw"

    @property
    def landing_dir(self) -> Path:
        """Landing zone des snapshots d'états : `landing/date=YYYY-MM-DD/`."""
        return self.data_path / "landing"

    @property
    def curated_dir(self) -> Path:
        return self.data_path / "curated"

    @property
    def logs_dir(self) -> Path:
        """Journaux applicatifs, dans le volume : lisibles à côté des données."""
        return self.data_path / "logs"

    @property
    def status_dir(self) -> Path:
        """Supervision : heartbeat, status.json, avancement du comptage de paires."""
        return self.data_path / "status"

    @property
    def compacted_states_dir(self) -> Path:
        """Sortie de la compaction horaire (`ciel_tranquille.compact`)."""
        return self.curated_dir / "states_hourly"

    @property
    def heartbeat_path(self) -> Path:
        return self.status_dir / "heartbeat.json"

    @property
    def status_path(self) -> Path:
        return self.status_dir / "status.json"

    @property
    def pairs_progress_path(self) -> Path:
        return self.status_dir / "pairs_progress.json"

    @property
    def samples_dir(self) -> Path:
        return self.data_path / "samples"

    @property
    def real_survol_dir(self) -> Path:
        """JSON bruts d'événements Bruitparif (gitignoré : donnée tierce brute)."""
        return self.data_path / "real_survol"

    @property
    def noise_events_dir(self) -> Path:
        """Landing zone Parquet des événements de survol (partition station/date)."""
        return self.raw_dir / "noise_events"

    @property
    def duckdb_path(self) -> Path:
        return self.curated_dir / "ciel_tranquille.duckdb"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.landing_dir,
            self.curated_dir,
            self.status_dir,
            self.logs_dir,
            self.samples_dir,
            self.real_survol_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Settings en singleton (cache process)."""
    return Settings()
