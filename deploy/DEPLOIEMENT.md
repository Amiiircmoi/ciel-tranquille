# Déploiement de la collecte (VPS Debian 13 · Docker Compose)

Procédure pour une **collecte continue de six jours sans supervision humaine**.

Aucune valeur propre à une machine ne figure dans ce dépôt : chemins, contacts et
secrets viennent tous du fichier d'environnement. Remplacez `<…>` par vos valeurs.

> **La contrainte qui commande tout le reste.** L'IP sortante de cette machine est
> partagée avec une application de production tierce. Un blocage chez OpenSky ou
> Bruitparif ferait tomber cette production. D'où : User-Agent identifiable avec
> contact, plancher d'une seconde entre appels Bruitparif, back-off exponentiel,
> arrêt net sur 429, et plafonds de ressources sur chaque conteneur.

---

## 1. Prérequis

```bash
docker --version && docker compose version   # plugin Compose v2 requis
```

Un compte OpenSky avec un **API client** (OAuth2 `client_id` / `client_secret`) :
https://opensky-network.org/ → profil → API client. Bruitparif ne demande aucun
compte (donnée ouverte, Licence Ouverte Etalab).

## 2. Récupérer le dépôt

```bash
git clone <url-du-depot> ciel-tranquille
cd ciel-tranquille
```

## 3. Créer le fichier d'environnement

```bash
cp .env.example .env
chmod 600 .env            # secrets : lisible par le seul propriétaire
$EDITOR .env
```

À renseigner **impérativement** :

| Variable | Valeur |
|---|---|
| `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET` | vos identifiants OAuth2 |
| `CIEL_CONTACT` | une adresse de contact joignable (figure dans le User-Agent) |
| `CIEL_HOST_DATA_DIR` | répertoire de l'hôte qui recevra les données |
| `CT_INGEST_MODE` | `live` |

Vérifiez qu'aucun secret n'a fui dans l'historique avant de pousser :

```bash
git check-ignore -v .env          # doit répondre : .gitignore:2:.env
git log -p --all -- .env | head   # doit être vide
```

## 4. Préparer le volume de données

Le conteneur tourne en **non-root** (uid 10001) et n'écrit **que** dans `/data`.
Le répertoire de l'hôte doit donc appartenir à cet uid :

```bash
mkdir -p "$CIEL_HOST_DATA_DIR"
sudo chown -R 10001:10001 "$CIEL_HOST_DATA_DIR"
```

Si vous préférez un autre propriétaire, alignez `CIEL_UID` / `CIEL_GID` dans
`.env` sur celui du répertoire.

## 5. Construire l'image de collecte

```bash
docker compose -f compose.prod.yaml build
```

L'image n'embarque ni scikit-learn ni Streamlit : uniquement le socle de collecte.

## 6. Choisir les stations — par **sondage**, jamais de mémoire

Le réseau Survol évolue (ouvertures, pannes, fermetures). On ne recopie aucune
liste : on interroge le réseau et on laisse le **rendement observé** décider.

```bash
docker compose -f compose.prod.yaml run --rm stations
```

Ce que fait la commande :

1. `GET /sites` fournit les **candidats** — seul endpoint exposant les
   identifiants **et leurs coordonnées** (`/events` n'en renvoie aucune, et sans
   coordonnées ni jointure ni contrôle de bbox ne sont possibles) ;
2. bornage à `CIEL_PROBE_MAX_CANDIDATES` en servant les trois couloirs à tour de
   rôle (CDG / Orly / Le Bourget) ;
3. **sondage** d'un appel par station sur une fenêtre diurne de 6 h du dernier
   jour complet, avec ≥ 1 s entre appels et arrêt net sur 429 ;
4. classement par événements exploitables, socle validé en tête, plafond
   `CIEL_MAX_ACTIVE_STATIONS` ;
5. écriture de `$CIEL_HOST_DATA_DIR/stations.json`, qui prime sur
   `config/stations.json`.

**Relisez le champ `active` avant de continuer** — c'est lui, et lui seul, qui
décide de ce qui sera collecté pendant six jours :

```bash
python3 -c "import json;d=json.load(open('$CIEL_HOST_DATA_DIR/stations.json'));print('\n'.join(d['active']))"
```

Puis contrôlez que chaque station active répond et tient dans la bbox :

```bash
docker compose -f compose.prod.yaml run --rm stations \
  python -m ciel_tranquille.ingest.stations validate
```

## 7. Démarrer la collecte

```bash
docker compose -f compose.prod.yaml up -d poller noise
```

Le poller **refuse de démarrer** si une station active est hors bbox ou muette
(code de sortie 2) : un silence de collecte ne doit jamais être une panne muette.

## 8. Vérifier le premier snapshot

Attendez une minute (deux ticks à 30 s), puis :

```bash
# 1. Le poller a démarré et validé ses stations
docker compose -f compose.prod.yaml logs --tail=30 poller

# 2. Le heartbeat existe et est frais
cat "$CIEL_HOST_DATA_DIR/status/heartbeat.json"

# 3. Un Parquet est bien tombé dans la landing du jour
ls -lh "$CIEL_HOST_DATA_DIR/landing/date=$(date -u +%F)/" | head

# 4. Le solde de crédits diminue d'exactement 1 par snapshot
tail -n 2 "$CIEL_HOST_DATA_DIR/curated/pipeline_metrics.jsonl"

# 5. Verdict global
docker compose -f compose.prod.yaml run --rm status
```

Attendu : `OK`, un âge de snapshot inférieur à 60 s, et un `credits_remaining`
qui décroît d'une unité par tick.

## 9. Ordonnancement : compaction horaire et statut au quart d'heure

Deux tâches à planifier. Choisissez selon ce dont dispose l'hôte.

### a. Hôte avec cron — deux lignes

`crontab -e` :

```cron
# Compaction horaire des snapshots (évite des dizaines de milliers de fichiers)
7 * * * * cd <chemin-du-depot> && docker compose -f compose.prod.yaml run --rm compact >> <chemin-des-donnees>/logs/compact.log 2>&1

# Rapport de supervision, toutes les 15 minutes
*/15 * * * * cd <chemin-du-depot> && docker compose -f compose.prod.yaml run --rm status >> <chemin-des-donnees>/logs/status.log 2>&1
```

Décalage volontaire à la minute 7 : la compaction passe avant le statut du
quart d'heure suivant, qui lit alors un état stabilisé. Le service `status` sort
en **code 1 quand le verdict est KO** — de quoi déclencher un mail cron sans
outillage supplémentaire.

### b. Hôte sans cron — ordonnanceurs conteneurisés

Debian n'installe plus cron par défaut, et créer une tâche système demande des
droits qu'un compte applicatif n'a pas forcément. Deux services du compose font
le même travail **sans aucun privilège** :

```bash
docker compose -f compose.prod.yaml up -d compact-scheduler status-scheduler
```

Ils bouclent en interne (`--every`), survivent à la déconnexion comme au
redémarrage de la machine (`restart: unless-stopped`), et journalisent dans
`<chemin-des-donnees>/logs/`. Cadences réglables par `CIEL_COMPACT_INTERVAL_S`
et `CIEL_STATUS_INTERVAL_S`.

N'activez qu'**une** des deux méthodes : cumuler cron et ordonnanceurs ferait
tourner deux compactions concurrentes sur les mêmes fichiers.

---

## Diagnostic

| Question | Commande |
|---|---|
| Est-ce que ça collecte ? | `docker compose -f compose.prod.yaml run --rm status` |
| Depuis quand ? | `cat $CIEL_HOST_DATA_DIR/status/heartbeat.json` |
| Rapport complet | `python3 -m json.tool $CIEL_HOST_DATA_DIR/status/status.json` |
| Combien de paires ? | `python3 -c "import json;print(json.load(open('$CIEL_HOST_DATA_DIR/status/status.json'))['paires'])"` |
| Santé du token Bruitparif | `python3 -c "import json;print(json.load(open('$CIEL_HOST_DATA_DIR/status/noise_health.json'))['token'])"` |
| Volumétrie | `du -sh $CIEL_HOST_DATA_DIR/landing $CIEL_HOST_DATA_DIR/curated` |
| Nombre de fichiers en landing | `find $CIEL_HOST_DATA_DIR/landing -name '*.parquet' \| wc -l` |
| Logs du poller | `docker compose -f compose.prod.yaml logs --tail=100 poller` |
| Journaux dans le volume | `tail -f $CIEL_HOST_DATA_DIR/logs/{poller,noise,compact,status}.log` |
| Consommation réelle | `docker stats --no-stream ciel-poller ciel-noise` |

### Lecture des verdicts KO

| Contrôle en échec | Cause probable | Geste |
|---|---|---|
| `snapshot_recent` / `debit_horaire` | poller arrêté, OpenSky injoignable | `logs poller` ; le service redémarre seul (`unless-stopped`) |
| `budget_credits` | budget quotidien épuisé | attendre minuit UTC — la collecte reprend seule ; vérifier `CT_POLL_INTERVAL_S` |
| `token_bruitparif` | motif de scraping cassé (front redéployé) | ajuster `_TOKEN_RE` dans `bruitparif_client.py` |
| `bruit_recent` | source bruit muette depuis > 2 h en journée | `logs noise` puis `stations validate` |
| `config_stations` | `stations.json` absent ou sans `active` | relancer le sondage (§6) |

### Arrêt et reprise

```bash
docker compose -f compose.prod.yaml stop poller noise    # arrêt propre (SIGTERM)
docker compose -f compose.prod.yaml up -d poller noise
```

Le poller intercepte SIGTERM et termine son tick en cours. Les écritures Parquet
passent par un temporaire suivi d'un `os.replace` : un arrêt brutal ne laisse
jamais de fichier tronqué dans la landing.

### En fin de collecte

```bash
# Poller arrêté : on peut compacter jusqu'à l'heure en cours incluse.
docker compose -f compose.prod.yaml run --rm compact \
  python -m ciel_tranquille.compact --lag-hours 0 --include-open-hours
docker compose -f compose.prod.yaml down
```

Les données restent dans `$CIEL_HOST_DATA_DIR` : `landing/` et
`curated/states_hourly/` (trafic), `raw/noise_events/` (bruit), `status/`
(supervision). Toutes sont relues par la validation de jointure sans réglage.
