"""Tests du TokenManager : rafraîchissement proactif, partage, back-off 429."""

from __future__ import annotations

import httpx
import pytest

from ciel_tranquille.ingest.opensky_client import OpenSkyClient
from ciel_tranquille.ingest.token_manager import (
    OpenSkyAuthError,
    TokenManager,
    get_token_manager,
    reset_token_manager,
)


class _Clock:
    """Horloge pilotée : permet de franchir l'expiration sans attendre 30 min."""

    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _token_transport(counter: dict, expires_in: int = 1800) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] = counter.get("n", 0) + 1
        return httpx.Response(
            200,
            json={"access_token": f"tok-{counter['n']}", "expires_in": expires_in},
        )

    return httpx.MockTransport(handler)


def _manager(settings, clock, counter, expires_in: int = 1800) -> TokenManager:
    settings.opensky_client_id = "id-de-test"
    settings.opensky_client_secret = "secret-de-test"
    return TokenManager(
        settings=settings,
        clock=clock,
        http_client=httpx.Client(transport=_token_transport(counter, expires_in)),
    )


def test_token_is_cached_between_calls(settings):
    clock, counter = _Clock(), {}
    mgr = _manager(settings, clock, counter)
    assert mgr.access_token() == "tok-1"
    assert mgr.access_token() == "tok-1"
    assert counter["n"] == 1  # un seul aller-retour vers le serveur d'auth


def test_token_refreshes_proactively_before_expiry(settings):
    """Le jeton est renouvelé AVANT l'échéance (marge de 30 s), pas après."""
    clock, counter = _Clock(), {}
    settings.token_refresh_margin_s = 30
    mgr = _manager(settings, clock, counter, expires_in=1800)
    assert mgr.access_token() == "tok-1"

    clock.advance(1800 - 31)  # encore 31 s de validité : au-delà de la marge
    assert mgr.access_token() == "tok-1"

    clock.advance(2)  # il reste 29 s : on entre dans la marge -> refresh anticipé
    assert mgr.access_token() == "tok-2"
    assert mgr.refresh_count == 2


def test_invalidate_forces_refresh(settings):
    clock, counter = _Clock(), {}
    mgr = _manager(settings, clock, counter)
    assert mgr.access_token() == "tok-1"
    mgr.invalidate()
    assert mgr.access_token() == "tok-2"


def test_auth_header_shape(settings):
    clock, counter = _Clock(), {}
    mgr = _manager(settings, clock, counter)
    assert mgr.auth_header() == {"Authorization": "Bearer tok-1"}


def test_missing_credentials_raises(settings):
    settings.opensky_client_id = ""
    settings.opensky_client_secret = ""
    mgr = TokenManager(settings=settings, clock=_Clock(), http_client=httpx.Client())
    with pytest.raises(OpenSkyAuthError):
        mgr.access_token()


def test_rejected_credentials_raise_without_retry_storm(settings):
    settings.opensky_client_id = "mauvais"
    settings.opensky_client_secret = "mauvais"
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid_client"})

    mgr = TokenManager(
        settings=settings,
        clock=_Clock(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OpenSkyAuthError):
        mgr.access_token()
    # 401 = refus définitif : on n'insiste pas (IP sortante partagée).
    assert calls["n"] == 1


def test_get_token_manager_returns_single_instance(settings):
    reset_token_manager()
    try:
        first = get_token_manager(settings)
        second = get_token_manager(settings)
        assert first is second
        # Le client OpenSky réutilise l'instance partagée plutôt que la sienne.
        client = OpenSkyClient(settings)
        try:
            assert client._tokens is first
        finally:
            client.close()
    finally:
        reset_token_manager()


def test_retry_after_prefers_opensky_header():
    """`X-Rate-Limit-Retry-After-Seconds` prime sur `Retry-After` standard."""
    resp = httpx.Response(
        429,
        headers={"x-rate-limit-retry-after-seconds": "42", "retry-after": "5"},
    )
    assert OpenSkyClient._retry_after_seconds(resp) == 42.0


def test_retry_after_falls_back_and_bounds():
    assert OpenSkyClient._retry_after_seconds(httpx.Response(429, headers={"retry-after": "7"})) == 7.0
    assert (
        OpenSkyClient._retry_after_seconds(
            httpx.Response(429, headers={"x-rate-limit-retry-after-seconds": "99999"})
        )
        == 120.0
    )
    assert OpenSkyClient._retry_after_seconds(httpx.Response(429)) == 10.0
    # En-tête illisible : on retombe sur le défaut plutôt que de repartir aussitôt.
    assert (
        OpenSkyClient._retry_after_seconds(
            httpx.Response(429, headers={"x-rate-limit-retry-after-seconds": "bientot"})
        )
        == 10.0
    )


def test_user_agent_carries_contact(settings):
    settings.user_agent = ""
    settings.contact = "ops@exemple.test"
    assert "ops@exemple.test" in settings.http_user_agent
    assert settings.has_contact
