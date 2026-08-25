"""Gestion du jeton OAuth2 OpenSky — **une seule instance partagée**.

Pourquoi extraire ce composant du client HTTP :

- Les jetons OpenSky (`client_credentials`) expirent au bout de **30 minutes**.
  Sur une collecte de six jours, cela fait ~290 rafraîchissements : ils doivent
  se faire **proactivement** (marge de 30 s avant expiration) plutôt qu'en
  réaction à un 401, sinon chaque expiration coûte un tick de collecte perdu.
- Plusieurs composants peuvent parler à OpenSky dans le même process (poller,
  outils de diagnostic). S'ils détenaient chacun leur jeton, chacun ferait ses
  propres appels au serveur d'auth — trafic inutile depuis une **IP sortante
  partagée avec une production tierce**. `get_token_manager()` renvoie donc une
  **instance unique par process**, protégée par un verrou : deux threads qui
  constatent l'expiration en même temps ne déclenchent qu'un seul refresh.

Le jeton n'est jamais journalisé ni écrit sur disque : seule sa durée de vie
résiduelle apparaît dans les logs.
"""

from __future__ import annotations

import logging
import threading
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


class OpenSkyError(RuntimeError):
    """Erreur non récupérable côté OpenSky (auth, configuration)."""


class OpenSkyAuthError(OpenSkyError):
    """Erreur d'authentification non récupérable (identifiants, configuration)."""


@dataclass(frozen=True)
class Token:
    """Jeton d'accès et son échéance (epoch secondes)."""

    value: str
    expires_at: float

    def is_fresh(self, now: float, margin_s: float) -> bool:
        return bool(self.value) and now < (self.expires_at - margin_s)

    def ttl_s(self, now: float) -> float:
        return max(0.0, self.expires_at - now)


class TokenManager:
    """Détenteur unique du jeton OAuth2, avec rafraîchissement proactif.

    `clock` est injectable pour tester l'expiration sans dépendre de l'horloge
    réelle ; `http_client` l'est pour tester sans réseau.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        clock=time.time,
        http_client: httpx.Client | None = None,
    ):
        self.settings = settings or get_settings()
        self._clock = clock
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(15.0),
            headers={"User-Agent": self.settings.http_user_agent},
        )
        self._lock = threading.Lock()
        self._token: Token | None = None
        self.refresh_count = 0

    # ------------------------------------------------------------------ public
    def access_token(self) -> str:
        """Jeton valide, rafraîchi d'avance si l'échéance approche."""
        margin = float(self.settings.token_refresh_margin_s)
        with self._lock:
            now = self._clock()
            if self._token is None or not self._token.is_fresh(now, margin):
                self._token = self._request_token()
                self.refresh_count += 1
                logger.info(
                    "Jeton OpenSky rafraîchi (#%d) — valide encore %.0f s.",
                    self.refresh_count,
                    self._token.ttl_s(self._clock()),
                )
            return self._token.value

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token()}"}

    def invalidate(self) -> None:
        """Force un refresh au prochain appel (ex. 401 inattendu côté API)."""
        with self._lock:
            self._token = None

    @property
    def expires_in_s(self) -> float | None:
        """TTL résiduel du jeton courant (diagnostic ; None si aucun jeton)."""
        return None if self._token is None else self._token.ttl_s(self._clock())

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------ privé
    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request_token(self) -> Token:
        if not self.settings.has_credentials:
            raise OpenSkyAuthError(
                "Identifiants OpenSky absents : renseignez OPENSKY_CLIENT_ID / "
                "OPENSKY_CLIENT_SECRET dans le fichier d'environnement (mode live)."
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
            raise OpenSkyAuthError(f"Échec d'authentification OpenSky ({resp.status_code}).")
        resp.raise_for_status()
        payload = resp.json()
        ttl = float(payload.get("expires_in") or self.settings.token_ttl_s)
        return Token(value=payload["access_token"], expires_at=self._clock() + ttl)


_MANAGER: TokenManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_token_manager(settings: Settings | None = None) -> TokenManager:
    """Instance unique de `TokenManager` pour le process."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = TokenManager(settings)
        return _MANAGER


def reset_token_manager() -> None:
    """Réinitialise l'instance partagée (tests, changement de configuration)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.close()
        _MANAGER = None
