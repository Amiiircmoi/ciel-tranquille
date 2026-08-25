"""Notifications mobiles via ntfy — alerter sans noyer.

Une collecte de six jours sans supervision n'a d'intérêt que si l'on apprend
qu'elle s'est arrêtée **au moment où elle s'arrête**, pas en ouvrant les données
le septième jour.

**La règle qui gouverne tout ce module : on ne notifie pas un état, on notifie un
changement d'état.** Le verdict est recalculé tous les quarts d'heure ; envoyer
chaque calcul ferait 96 notifications par jour. Au bout de deux jours le canal
est coupé ou ignoré, et le dispositif ne sert plus à rien — il serait alors pire
qu'absent, puisqu'on croirait être couvert. Quatre événements seulement :

- `demarrage` : première exécution — confirme que le canal fonctionne ;
- `panne`     : le verdict passe OK → KO, priorité haute ;
- `retabli`   : le verdict repasse KO → OK ;
- `resume`    : un point quotidien à heure fixe, même quand tout va bien.

Le résumé quotidien n'est pas décoratif : il est la seule protection contre le
mode de défaillance le plus perfide, celui où la supervision elle-même est morte
et où le silence passe pour une bonne nouvelle. **Son absence est un signal.**
Aucune notification push ne peut prouver qu'un système est vivant ; seul un
message attendu et non reçu le peut.

Confidentialité : le sujet ntfy vaut mot de passe (qui le connaît lit les
notifications). Il vient de `CIEL_NTFY_TOPIC`, n'est jamais journalisé, et
`CIEL_NTFY_URL` permet de viser une instance auto-hébergée. Le corps des messages
ne contient que des compteurs — ni jeton, ni identifiant d'aéronef, ni position.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from ciel_tranquille.config import Settings, get_settings
from ciel_tranquille.monitoring.heartbeat import write_json_atomic

logger = logging.getLogger(__name__)

DEMARRAGE = "demarrage"
PANNE = "panne"
RETABLI = "retabli"
RESUME = "resume"

# Priorités ntfy : 1 min, 3 défaut, 4 haute, 5 urgente.
_PRIORITE = {DEMARRAGE: 3, PANNE: 5, RETABLI: 4, RESUME: 2}
_TAGS = {
    DEMARRAGE: ["satellite"],
    PANNE: ["rotating_light"],
    RETABLI: ["white_check_mark"],
    RESUME: ["bar_chart"],
}


@dataclass
class Notification:
    """Message prêt à publier (aucun secret, aucune donnée brute)."""

    event: str
    title: str
    message: str

    def to_payload(self, topic: str) -> dict:
        return {
            "topic": topic,
            "title": self.title,
            "message": self.message,
            "priority": _PRIORITE.get(self.event, 3),
            "tags": _TAGS.get(self.event, []),
        }


def _resume_chiffres(payload: dict) -> str:
    collecte = payload.get("collecte", {})
    credits = payload.get("credits", {})
    bruit = payload.get("bruit", {})
    paires = payload.get("paires", {})
    lignes = [
        f"Snapshots/h : {collecte.get('snapshots_last_hour')}",
        f"Dernier snapshot : {collecte.get('last_snapshot_age_s')} s",
        f"Crédits OpenSky : {credits.get('remaining')} / {credits.get('daily_budget')}",
        f"Bruit du jour : {bruit.get('events_today')} évts "
        f"({bruit.get('stations_reporting_today')}/{bruit.get('stations_configured')} stations)",
        f"Paires propres : {paires.get('clean_pairs_total')}",
    ]
    return "\n".join(lignes)


def build_notification(event: str, payload: dict) -> Notification:
    """Compose le message d'un événement à partir du rapport de supervision."""
    verdict = payload.get("verdict", "?")
    echecs = payload.get("failed_checks") or []
    chiffres = _resume_chiffres(payload)

    if event == PANNE:
        details = {c["name"]: c["detail"] for c in payload.get("checks", []) if not c["ok"]}
        corps = "\n".join(f"• {nom} : {detail}" for nom, detail in details.items())
        return Notification(
            event,
            "Ciel Tranquille — collecte en défaut",
            f"{corps}\n\n{chiffres}",
        )
    if event == RETABLI:
        return Notification(
            event,
            "Ciel Tranquille — collecte rétablie",
            f"Tous les contrôles repassent au vert.\n\n{chiffres}",
        )
    if event == DEMARRAGE:
        return Notification(
            event,
            "Ciel Tranquille — supervision active",
            f"Notifications branchées. Verdict actuel : {verdict}.\n\n{chiffres}",
        )
    return Notification(
        event,
        f"Ciel Tranquille — point quotidien ({verdict})",
        (f"Contrôles en défaut : {', '.join(echecs)}\n\n" if echecs else "")
        + chiffres
        + "\n\nCe message arrive une fois par jour. Son absence est un signal.",
    )


def _lire_etat(settings: Settings) -> dict:
    path = settings.notify_state_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("État des notifications illisible — repart de zéro.")
        return {}


def decide_event(etat: dict, payload: dict, now: float, heure_resume: int) -> str | None:
    """Détermine l'événement à notifier, ou None s'il n'y a rien à dire.

    Priorité au changement d'état : une panne ou un rétablissement passe avant le
    résumé quotidien, qui peut attendre le cycle suivant.
    """
    verdict = payload.get("verdict")
    precedent = etat.get("last_verdict")

    if precedent is None:
        return DEMARRAGE
    if verdict != precedent:
        return PANNE if verdict != "OK" else RETABLI

    jour = time.strftime("%Y-%m-%d", time.gmtime(now))
    heure = time.gmtime(now).tm_hour
    if heure >= heure_resume and etat.get("last_digest_date") != jour:
        return RESUME
    return None


def _publier(settings: Settings, notif: Notification, post: Callable | None = None) -> bool:
    """Publie sur ntfy. Retourne False sur échec, sans jamais lever."""
    payload = notif.to_payload(settings.ntfy_topic)
    try:
        if post is not None:
            post(settings.ntfy_url, payload)
        else:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                reponse = client.post(settings.ntfy_url, json=payload)
                reponse.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001 — une notification ratée n'arrête rien
        # Le sujet n'apparaît jamais dans les logs : il vaut mot de passe.
        logger.warning("Publication ntfy en échec (%s) : %s", notif.event, exc)
        return False


def notify_from_status(
    settings: Settings | None = None,
    payload: dict | None = None,
    now: float | None = None,
    post: Callable | None = None,
) -> str | None:
    """Notifie si l'état a changé. Retourne l'événement publié, ou None.

    L'état n'est mis à jour **qu'après un envoi réussi** : si ntfy est
    injoignable, la panne sera renotifiée au cycle suivant plutôt que perdue.
    """
    settings = settings or get_settings()
    if not settings.ntfy_enabled or payload is None:
        return None
    now = time.time() if now is None else now

    etat = _lire_etat(settings)
    event = decide_event(etat, payload, now, settings.ntfy_digest_hour_utc)
    if event is None:
        return None

    if not _publier(settings, build_notification(event, payload), post):
        return None

    nouvel_etat = dict(etat)
    nouvel_etat["last_verdict"] = payload.get("verdict")
    nouvel_etat["last_event"] = event
    nouvel_etat["last_sent_unix"] = round(now, 1)
    nouvel_etat["last_sent_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    if event in (RESUME, DEMARRAGE):
        nouvel_etat["last_digest_date"] = time.strftime("%Y-%m-%d", time.gmtime(now))
    write_json_atomic(settings.notify_state_path, nouvel_etat)
    logger.info("Notification publiée : %s (verdict %s).", event, payload.get("verdict"))
    return event
