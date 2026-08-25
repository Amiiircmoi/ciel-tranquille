"""Client OpenSky Network — OAuth2 *client credentials* + endpoint `/states/all`.

Pourquoi ce design :
- **OAuth2** (Basic Auth est déprécié côté OpenSky) : le jeton est détenu par un
  `TokenManager` **unique et partagé** (cf. `token_manager.py`), rafraîchi
  proactivement 30 s avant son échéance de 30 min.
- **httpx** : client HTTP moderne (timeouts explicites, HTTP/2).
- **tenacity** : retry exponentiel sur erreurs réseau / 429 (rate-limit), pour
  un poller robuste face aux aléas de l'API publique gratuite.

**Civisme réseau (contrainte d'exploitation).** L'IP sortante est partagée avec
une production tierce : un blocage côté OpenSky ferait tomber cette production.
Trois garde-fous, non négociables :
1. **User-Agent identifiable** avec une adresse de contact (`CIEL_USER_AGENT` /
   `CIEL_CONTACT`) — un opérateur doit pouvoir nous joindre avant de bloquer ;
2. **respect strict du 429** : on lit `X-Rate-Limit-Retry-After-Seconds` (en-tête
   propre à OpenSky) avant `Retry-After`, et on attend réellement ce délai ;
3. **journalisation de `X-Rate-Limit-Remaining` à chaque appel** — c'est ce solde
   qui pilote le garde-fou de budget du poller.

Garde-fou d'honnêteté : OpenSky en gratuit n'est PAS un flux *push*. On fait du
**polling micro-batch** (cadence 30 s). Ce module ne prétend rien d'autre.
"""

from __future__ import annotations

import logging
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.ingest.token_manager import (
    OpenSkyAuthError,
    OpenSkyError,
    TokenManager,
    get_token_manager,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STATE_FIELDS",
    "OpenSkyAuthError",
    "OpenSkyClient",
    "OpenSkyError",
    "states_to_records",
]

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

# En-tête de back-off propre à OpenSky, prioritaire sur `Retry-After` standard.
_RETRY_AFTER_HEADERS = ("x-rate-limit-retry-after-seconds", "retry-after")
# Borne haute d'attente sur 429 : au-delà, tenacity reprend la main.
_MAX_RETRY_AFTER_S = 120.0


class OpenSkyClient:
    """Accès authentifié à l'API REST OpenSky.

    Le jeton n'appartient pas au client : il vient du `TokenManager` partagé
    (`token_manager` injectable pour les tests). Le `clock` injectable (par défaut
    `time.time`) et `sleep` gardent le back-off testable sans horloge réelle.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        clock=time.time,
        sleep=time.sleep,
        token_manager: TokenManager | None = None,
    ):
        self.settings = settings or get_settings()
        self._clock = clock
        self._sleep = sleep
        self._tokens = token_manager or get_token_manager(self.settings)
        self._client = httpx.Client(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": self.settings.http_user_agent},
        )
        # Crédits OpenSky restants (header `x-rate-limit-remaining`) — allocation
        # QUOTIDIENNE. Mis à jour à chaque appel ; lu par le poller (garde-fou budget).
        self.last_rate_limit_remaining: int | None = None
        # Dernier délai de back-off imposé par l'API (429), en secondes.
        self.last_retry_after_s: float | None = None

    # --------------------------------------------------------- rate-limiting
    def _capture_rate_limit(self, resp: httpx.Response) -> None:
        """Mémorise le solde de crédits quotidien (header `x-rate-limit-remaining`)."""
        raw = resp.headers.get("x-rate-limit-remaining")
        if raw is not None:
            try:
                self.last_rate_limit_remaining = int(raw)
            except ValueError:
                logger.debug("x-rate-limit-remaining illisible: %r", raw)

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response, default: float = 10.0) -> float:
        """Délai d'attente sur 429, borné à 120 s.

        Priorité à `X-Rate-Limit-Retry-After-Seconds` (en-tête OpenSky), puis au
        `Retry-After` standard, puis au défaut. Un délai négatif ou illisible
        retombe sur le défaut plutôt que de repartir immédiatement.
        """
        for name in _RETRY_AFTER_HEADERS:
            raw = resp.headers.get(name)
            if not raw:
                continue
            try:
                return max(1.0, min(float(raw), _MAX_RETRY_AFTER_S))
            except ValueError:
                logger.debug("%s illisible: %r", name, raw)
        return default

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
        Met à jour `last_rate_limit_remaining` (budget crédits) à chaque appel et
        respecte le délai imposé sur un 429 avant de laisser tenacity rejouer.
        """
        resp = self._client.get(
            self.settings.opensky_states_url,
            params=self.settings.bbox.as_params(),
            headers=self._tokens.auth_header(),
        )
        self._capture_rate_limit(resp)
        logger.debug(
            "GET /states/all -> %s ; crédits restants=%s",
            resp.status_code,
            self.last_rate_limit_remaining,
        )
        if resp.status_code == 429:
            wait_s = self._retry_after_seconds(resp)
            self.last_retry_after_s = wait_s
            logger.warning(
                "OpenSky rate-limit (429) — attente imposée %.0fs, crédits restants=%s ; "
                "back-off puis retry exponentiel.",
                wait_s,
                self.last_rate_limit_remaining,
            )
            self._sleep(wait_s)
            resp.raise_for_status()
        if resp.status_code == 401:
            # jeton périmé entre-temps : on l'invalide et on laisse tenacity rejouer
            self._tokens.invalidate()
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
