"""Client Bruitparif *Survol* — événements de survol (source bruit).

Pourquoi ce design :
- L'app `survol.bruitparif.fr` est une SPA qui embarque un **token public** dans
  son HTML (`token: '...'`). On le **scrape** à chaud plutôt que de le coder en
  dur : ces tokens tournent à chaque déploiement de l'app. Aucun secret côté
  Bruitparif (donnée ouverte, Licence Ouverte Etalab) → pas de `.env`.
- **httpx** + **tenacity** : mêmes briques que le client OpenSky. L'API renvoie
  parfois des coupures de connexion sur des rafales de requêtes (observé) → retry
  exponentiel sur erreurs réseau, et espacement entre appels côté collecteur.

Endpoint clé : ``GET /events/{station}/{from}/{to}?token=…`` où l'intervalle de
dates est **semi-ouvert** ``[from, to)`` (un jour = ``[J, J+1)``). La réponse est
plafonnée à ~50 événements et biaisée vers le début de l'intervalle → on
interroge **jour par jour** pour une couverture complète (cf. phase0).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ciel_tranquille.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Token public embarqué dans la page de l'app : `token: 'XXXXXXXX...'`.
_TOKEN_RE = re.compile(r"token:\s*'([^']+)'")


class BruitparifError(RuntimeError):
    """Erreur non récupérable côté Bruitparif (token, configuration)."""


class BruitparifClient:
    """Accès à l'API REST publique `rumeurengine.bruitparif.fr`."""

    def __init__(self, settings: Settings | None = None, token: str | None = None):
        self.settings = settings or get_settings()
        self._token: str | None = token
        self._client = httpx.Client(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "ciel-tranquille/1.0 (urban-noise-analysis; +github.com/Amiiircmoi/ciel-tranquille)"},
        )

    # ------------------------------------------------------------------ token
    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def fetch_token(self) -> str:
        """Scrape un token public courant depuis la page de l'app Survol."""
        resp = self._client.get(self.settings.bruitparif_app_url)
        resp.raise_for_status()
        match = _TOKEN_RE.search(resp.text)
        if not match:
            raise BruitparifError(
                "Token public introuvable dans la page Bruitparif "
                "(structure de l'app modifiée ?)."
            )
        return match.group(1)

    def token(self) -> str:
        if not self._token:
            self._token = self.fetch_token()
            logger.info("Token Bruitparif obtenu (len=%d).", len(self._token))
        return self._token

    # ----------------------------------------------------------------- events
    @staticmethod
    def _fmt(bound: date | datetime | str) -> str:
        """Formate une borne pour l'URL : date -> 'YYYY-MM-DD', datetime ISO 'T'.

        L'API interprète l'heure en **UTC** et honore la granularité **infra-jour**
        (vérifié) — indispensable car un appel est plafonné à ~50 événements.
        """
        if isinstance(bound, datetime):
            return bound.strftime("%Y-%m-%dT%H:%M:%S")
        return str(bound)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def fetch_events(
        self, station: str, frm: date | datetime, to: date | datetime
    ) -> list[dict]:
        """Événements bruts d'une station sur l'intervalle semi-ouvert [frm, to).

        `frm`/`to` peuvent être des `date` (granularité jour) ou des `datetime`
        (granularité infra-jour, UTC). Retourne la liste JSON brute (toutes
        catégories). Plafond serveur ~50 événements/appel.
        """
        url = f"{self.settings.bruitparif_api_url}/events/{station}/{self._fmt(frm)}/{self._fmt(to)}"
        resp = self._client.get(url, params={"token": self.token()})
        if resp.status_code in (401, 403):
            # token périmé : on le réinitialise et on laisse tenacity rejouer
            self._token = None
            raise BruitparifError(f"Accès refusé Bruitparif ({resp.status_code}).")
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def fetch_day_air_events(self, station: str, day: date) -> list[dict]:
        """Survols ('air') d'une station pour un jour J (granularité jour — tronqué
        si >50/j ; préférer `fetch_window_air_events` sur ces couloirs chargés)."""
        events = self.fetch_events(station, day, day + timedelta(days=1))
        return [e for e in events if e.get("category") == "air"]

    def fetch_window_air_events(
        self, station: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Survols ('air') d'une station sur une fenêtre infra-journalière [start, end)."""
        events = self.fetch_events(station, start, end)
        return [e for e in events if e.get("category") == "air"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BruitparifClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
