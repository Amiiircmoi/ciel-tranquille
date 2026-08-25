"""Tests des notifications ntfy : transitions, anti-spam, confidentialité."""

from __future__ import annotations

import json

from ciel_tranquille.monitoring import notify

NOON_UTC = 1_787_659_200  # 2026-08-25T12:00:00Z


def _payload(verdict="OK", echecs=(), checks=None):
    return {
        "verdict": verdict,
        "failed_checks": list(echecs),
        "checks": checks
        or [{"name": "snapshot_recent", "ok": verdict == "OK", "detail": "il y a 12 s"}],
        "collecte": {"snapshots_last_hour": 118, "last_snapshot_age_s": 12.0},
        "credits": {"remaining": 3712, "daily_budget": 4000},
        "bruit": {"events_today": 2026, "stations_reporting_today": 10, "stations_configured": 10},
        "paires": {"clean_pairs_total": 512},
    }


class _Canal:
    """Faux transport : capture ce qui serait publié, sans réseau."""

    def __init__(self):
        self.envois: list[tuple[str, dict]] = []

    def __call__(self, url, payload):
        self.envois.append((url, payload))


def _actif(settings, topic="sujet-de-test"):
    settings.ntfy_topic = topic
    settings.ntfy_url = "https://exemple.test"
    settings.ntfy_digest_hour_utc = 7
    return settings


# ----------------------------------------------------------------- activation
def test_disabled_without_topic(settings):
    settings.ntfy_topic = ""
    canal = _Canal()
    assert notify.notify_from_status(settings, _payload(), NOON_UTC, canal) is None
    assert canal.envois == []


# ------------------------------------------------------------------ decisions
def test_first_run_announces_itself():
    """Le tout premier passage confirme que le canal fonctionne."""
    assert notify.decide_event({}, _payload(), NOON_UTC, 7) == notify.DEMARRAGE


def test_steady_state_says_nothing():
    """Verdict inchangé, résumé du jour déjà envoyé : silence.

    C'est la règle qui rend le dispositif utilisable : 96 calculs de verdict par
    jour ne doivent pas produire 96 notifications.
    """
    etat = {"last_verdict": "OK", "last_digest_date": "2026-08-25"}
    assert notify.decide_event(etat, _payload(), NOON_UTC, 7) is None


def test_failure_transition_is_notified():
    etat = {"last_verdict": "OK", "last_digest_date": "2026-08-25"}
    assert notify.decide_event(etat, _payload("KO", ["bruit_recent"]), NOON_UTC, 7) == notify.PANNE


def test_recovery_transition_is_notified():
    etat = {"last_verdict": "KO", "last_digest_date": "2026-08-25"}
    assert notify.decide_event(etat, _payload("OK"), NOON_UTC, 7) == notify.RETABLI


def test_daily_digest_fires_once_per_day():
    etat = {"last_verdict": "OK", "last_digest_date": "2026-08-24"}
    assert notify.decide_event(etat, _payload(), NOON_UTC, 7) == notify.RESUME
    # Une fois envoyé, plus rien avant le lendemain.
    etat["last_digest_date"] = "2026-08-25"
    assert notify.decide_event(etat, _payload(), NOON_UTC, 7) is None


def test_digest_waits_for_its_hour():
    etat = {"last_verdict": "OK", "last_digest_date": "2026-08-24"}
    minuit_trente = NOON_UTC - 11 * 3600 - 1800  # 00:30 UTC, avant l'heure de 7 h
    assert notify.decide_event(etat, _payload(), minuit_trente, 7) is None


def test_failure_takes_precedence_over_digest():
    """Une panne ne doit pas être maquillée en point quotidien."""
    etat = {"last_verdict": "OK", "last_digest_date": "2026-08-24"}
    assert notify.decide_event(etat, _payload("KO", ["x"]), NOON_UTC, 7) == notify.PANNE


# -------------------------------------------------------------------- contenu
def test_failure_message_lists_the_failing_checks():
    checks = [
        {"name": "snapshot_recent", "ok": False, "detail": "dernier snapshot il y a 900 s"},
        {"name": "budget_credits", "ok": True, "detail": "crédits restants=3712"},
    ]
    notif = notify.build_notification(
        notify.PANNE, _payload("KO", ["snapshot_recent"], checks)
    )
    assert "snapshot_recent" in notif.message
    assert "900 s" in notif.message
    assert "budget_credits" not in notif.message  # on ne liste que ce qui échoue
    assert notif.to_payload("t")["priority"] == 5  # urgente


def test_digest_says_that_its_absence_matters():
    notif = notify.build_notification(notify.RESUME, _payload())
    assert "absence" in notif.message.lower()
    assert notif.to_payload("t")["priority"] == 2  # discrète


def test_messages_carry_no_secret_and_no_raw_data():
    for event in (notify.DEMARRAGE, notify.PANNE, notify.RETABLI, notify.RESUME):
        notif = notify.build_notification(event, _payload("KO", ["bruit_recent"]))
        texte = (notif.title + notif.message).lower()
        for interdit in ("secret", "client_id", "bearer", "token", "icao", "callsign"):
            assert interdit not in texte, (event, interdit)


def test_topic_travels_in_the_payload_not_the_url(settings):
    _actif(settings, "sujet-prive-xyz")
    canal = _Canal()
    notify.notify_from_status(settings, _payload(), NOON_UTC, canal)
    url, payload = canal.envois[0]
    assert url == "https://exemple.test"       # pas de sujet dans l'URL journalisée
    assert payload["topic"] == "sujet-prive-xyz"


# ---------------------------------------------------------------- persistance
def test_state_is_written_after_a_successful_send(settings):
    _actif(settings)
    canal = _Canal()
    assert notify.notify_from_status(settings, _payload(), NOON_UTC, canal) == notify.DEMARRAGE
    etat = json.loads(settings.notify_state_path.read_text(encoding="utf-8"))
    assert etat["last_verdict"] == "OK"
    assert etat["last_digest_date"] == "2026-08-25"
    # Deuxième passage identique : plus rien à dire.
    assert notify.notify_from_status(settings, _payload(), NOON_UTC, canal) is None
    assert len(canal.envois) == 1


def test_failed_send_does_not_consume_the_alert(settings):
    """ntfy injoignable : la panne doit être renotifiée au cycle suivant.

    Marquer l'état comme notifié alors que rien n'est parti perdrait l'alerte
    pour de bon — précisément le cas où l'on a besoin d'elle.
    """
    _actif(settings)

    def _casse(url, payload):
        raise RuntimeError("ntfy injoignable")

    assert notify.notify_from_status(settings, _payload("KO", ["x"]), NOON_UTC, _casse) is None
    assert not settings.notify_state_path.exists()

    canal = _Canal()
    assert notify.notify_from_status(settings, _payload("KO", ["x"]), NOON_UTC, canal) is not None
    assert len(canal.envois) == 1


def test_notification_failure_never_breaks_the_status_report(settings, monkeypatch):
    """Le rapport de supervision prime : il s'écrit même si l'alerte échoue."""
    from ciel_tranquille import status

    _actif(settings)
    monkeypatch.setattr(
        "ciel_tranquille.status.notify_from_status",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boum")),
    )
    payload = status.write_status(settings, now=NOON_UTC, with_pairs=False)
    assert settings.status_path.exists()
    assert payload["verdict"] in ("OK", "KO")
