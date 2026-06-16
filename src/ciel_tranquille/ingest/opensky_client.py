"""Client OpenSky Network — OAuth2 *client credentials* + endpoint `/states/all`.

Pourquoi ce design :
- **OAuth2** (Basic Auth est déprécié côté OpenSky) : on échange `client_id` /
  `client_secret` contre un *access token* à durée de vie limitée, mis en cache
  et rafraîchi automatiquement.
- **httpx** : client HTTP moderne (timeouts explicites, HTTP/2).
- **tenacity** : retry exponentiel sur erreurs réseau / 429 (rate-limit), pour
  un poller robuste face aux aléas de l'API publique gratuite.

Garde-fou d'honnêteté : OpenSky en gratuit n'est PAS un flux *push*. On fait du
**polling micro-batch** (≈1 req / 10 s). Ce module ne prétend rien d'autre.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ciel_tranquille.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Ordre des 17 colonnes du vecteur d'état OpenSky (`/states/all`).
# https://openskynetwork.github.io/opensky-api/rest.html#response
STATE_FIELDS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude_m",
    "on_ground",
    "velocity_m_s",
    "true_track_deg",
    "vertical_rate_m_s",
    "sensors",
    "geo_altitude_m",
    "squawk",
    "spi",
    "position_source",
]


class OpenSkyError(RuntimeError):
    """Erreur non récupérable côté OpenSky (auth, configuration)."""


@dataclass
class _Token:
    value: str
    expires_at: float  # epoch seconds

    @property
    def is_valid(self) -> bool:
        # marge de 30 s pour éviter d'utiliser un token au bord de l'expiration
        return bool(self.value) and (time.time() < self.expires_at - 30)


class OpenSkyClient:
    """Accès authentifié à l'API REST OpenSky.

    Le `clock` injectable (par défaut `time.time`) rend l'expiration de token
    testable sans dépendre de l'horloge réelle.
    """

    def __init__(self, settings: Settings | None = None, clock=time.time):
        self.settings = settings or get_settings()
        self._clock = clock
        self._token: _Token | None = None
        self._client = httpx.Client(timeout=httpx.Timeout(15.0))

    # ------------------------------------------------------------------ auth
    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _fetch_token(self) -> _Token:
        if not self.settings.has_credentials:
            raise OpenSkyError(
                "Identifiants OpenSky absents : renseignez OPENSKY_CLIENT_ID / "
                "OPENSKY_CLIENT_SECRET dans .env (mode live)."
            )
        resp = self._client.post(
            self.settings.opensky_token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.settings.opensky_client_id,
                "client_secret": self.settings.opensky_client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code in (400, 401, 403):
            raise OpenSkyError(f"Échec d'authentification OpenSky ({resp.status_code}).")
        resp.raise_for_status()
        payload = resp.json()
        ttl = float(payload.get("expires_in", 1800))
        return _Token(value=payload["access_token"], expires_at=self._clock() + ttl)

    def _auth_header(self) -> dict[str, str]:
        if self._token is None or not self._token.is_valid:
            self._token = self._fetch_token()
        return {"Authorization": f"Bearer {self._token.value}"}

    # ----------------------------------------------------------------- states
    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def fetch_states(self) -> dict:
        """Récupère un *snapshot* d'états dans la bounding box configurée.

        Retourne le JSON brut OpenSky : ``{"time": int, "states": [[...], ...]}``.
        """
        header = self._auth_header()
        resp = self._client.get(
            self.settings.opensky_states_url,
            params=self.settings.bbox.as_params(),
            headers=header,
        )
        if resp.status_code == 429:
            logger.warning("OpenSky rate-limit (429) — backoff via tenacity.")
            resp.raise_for_status()
        if resp.status_code == 401:
            # token périmé entre-temps : on l'invalide et on laisse tenacity rejouer
            self._token = None
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenSkyClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def states_to_records(payload: dict, snapshot_ts: int | None = None) -> list[dict]:
    """Transforme le JSON OpenSky en liste de dict (1 dict = 1 aéronef).

    Normalise vers le schéma interne aligné sur ``opensky_snapshot.csv``
    (``time_position_unix``), pour que live et replay soient interchangeables.
    """
    ts = snapshot_ts if snapshot_ts is not None else payload.get("time")
    records: list[dict] = []
    for state in payload.get("states") or []:
        row = dict(zip(STATE_FIELDS, state, strict=False))
        callsign = row.get("callsign")
        records.append(
            {
                "icao24": row.get("icao24"),
                "callsign": (callsign or "").strip() or None,
                "origin_country": row.get("origin_country"),
                "time_position_unix": row.get("time_position") or ts,
                "longitude": row.get("longitude"),
                "latitude": row.get("latitude"),
                "baro_altitude_m": row.get("baro_altitude_m"),
                "velocity_m_s": row.get("velocity_m_s"),
                "heading_deg": row.get("true_track_deg"),
                "squawk": row.get("squawk"),
                "last_contact_unix": row.get("last_contact"),
                "on_ground": row.get("on_ground"),
                "snapshot_ts": ts,
            }
        )
    return records
