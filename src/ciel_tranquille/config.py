"""Configuration centralisée, chargée depuis l'environnement / `.env`.

Les secrets OpenSky ne sont JAMAIS codés en dur : ils proviennent de variables
d'environnement (fichier `.env` gitignoré). Voir `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du dépôt (src/ciel_tranquille/config.py -> remonte de 3 niveaux)
REPO_ROOT = Path(__file__).resolve().parents[2]


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


class Settings(BaseSettings):
    """Paramètres applicatifs (préfixe d'env `CT_`, secrets OpenSky `OPENSKY_`)."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets OpenSky (OAuth2 client credentials) ---
    opensky_client_id: str = Field(default="", alias="OPENSKY_CLIENT_ID")
    opensky_client_secret: str = Field(default="", alias="OPENSKY_CLIENT_SECRET")

    # --- Bounding box (Île-de-France par défaut) ---
    bbox_lat_min: float = Field(default=48.50, alias="CT_BBOX_LAT_MIN")
    bbox_lon_min: float = Field(default=1.90, alias="CT_BBOX_LON_MIN")
    bbox_lat_max: float = Field(default=49.10, alias="CT_BBOX_LAT_MAX")
    bbox_lon_max: float = Field(default=2.80, alias="CT_BBOX_LON_MAX")

    # --- Cadence micro-batch ---
    poll_interval_s: int = Field(default=12, alias="CT_POLL_INTERVAL_S")

    # --- Mode d'ingestion : "live" (API réelle) | "replay" (snapshots) ---
    ingest_mode: str = Field(default="replay", alias="CT_INGEST_MODE")

    # --- Chemins ---
    data_dir: str = Field(default="data", alias="CT_DATA_DIR")

    # --- Endpoints OpenSky ---
    opensky_states_url: str = "https://opensky-network.org/api/states/all"
    opensky_token_url: str = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/"
        "protocol/openid-connect/token"
    )

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(
            lat_min=self.bbox_lat_min,
            lon_min=self.bbox_lon_min,
            lat_max=self.bbox_lat_max,
            lon_max=self.bbox_lon_max,
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.opensky_client_id and self.opensky_client_secret)

    @property
    def data_path(self) -> Path:
        p = REPO_ROOT / self.data_dir
        return p if p.is_absolute() else REPO_ROOT / self.data_dir

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_path / "curated"

    @property
    def samples_dir(self) -> Path:
        return self.data_path / "samples"

    @property
    def duckdb_path(self) -> Path:
        return self.curated_dir / "ciel_tranquille.duckdb"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.curated_dir, self.samples_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Settings en singleton (cache process)."""
    return Settings()
