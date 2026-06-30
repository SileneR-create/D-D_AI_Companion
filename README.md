# 🐉 D&D AI Companion

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Vite-61dafb?logo=react)](https://react.dev/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Compagnon de jeu pour Donjons & Dragons (5e / 2024)** : un oracle des règles (RAG sur le manuel
de base + recherche d'outils), une table de Maître du Jeu pilotée par un LLM **local** (Ollama),
la création de personnages au format **feuille 2024**, une bibliothèque transversale de PNJ / lieux /
objets / trésors, une forge d'objets, et la gestion complète des campagnes.

> Tout fonctionne **en local** : aucune donnée n'est envoyée à un service tiers, le modèle tourne sur votre machine.

<!-- DÉMO : remplacez par un GIF de prise en main globale (voir docs/demos/README.md) -->
![Aperçu général](docs/demos/01-apercu.gif)

---

## ✨ Fonctionnalités

- **Le Grimoire** — consultation des règles 5e : **recherche instantanée** (sorts, monstres,
  équipement, objets magiques, conditions issus du SRD) qui répond en < 1 s, **plus** un chat **RAG**
  (manuel FR indexé + outils MCP) pour les questions ouvertes. Vous pouvez **ajouter vos documents**
  (PDF/txt/md) ou **aspirer un site** dans l'index.
- **La Table du Maître** — chat de campagne avec **historique persistant**, création de PNJ / quêtes /
  lieux, **ajout de PJ** à l'aventure, lancer de dés, **détection d'intention** pour accélérer le LLM,
  et une **référence rapide hors-ligne** (conditions, actions, couvert, DD, repos) en un clic.
- **Création de personnage** — point-buy guidé, sorts par classe avec descriptions, **sous-classe**
  et **historique 2024** (compétences, don d'origine et bonus de caractéristiques ajoutés
  automatiquement), bourse de départ.
- **Feuille de personnage 2024 complète** — CA, initiative, bonus de maîtrise, sauvegardes,
  18 compétences, DD/attaque de sorts **calculés automatiquement** ; PV, dés de vie, jets de mort,
  langues, dons, équipement et bourse (po/pa/pe/po/pp) éditables.
- **PJ multi-campagnes** — un personnage peut **rejoindre/quitter plusieurs aventures** (depuis Mon
  antre ou La Table).
- **Les Archives** — vos **PNJ et lieux**, activables/désactivables par campagne. Les PNJ peuvent être
  des **adversaires** (CA, PV, FP, ou stats d'un **monstre SRD** pré-rempli).
- **L'Arsenal** — **catalogue d'objets D&D** (armes, armures, équipement, objets magiques) avec fiche
  détaillée, **plus** vos objets forgés / trésors ; tout est **attribuable** à un PJ, un PNJ ou une quête.
- **La Forge** — créez des objets/armes/artefacts/trésors (valeur en po/pa/pc).
- **Gestion des campagnes** — activer, **terminer** (archive des quêtes réussies/échouées) ou
  **supprimer** ; nom **unique par utilisateur** (deux comptes peuvent réutiliser un nom).
- **Comptes & rôles** — authentification JWT, distinction **Maître du Jeu / Joueur** (outils filtrés par rôle).

---

## 🖼️ Captures & démos

> Les emplacements ci-dessous sont prévus pour vos propres captures/GIF.
> Voir [`docs/demos/README.md`](docs/demos/README.md) pour la liste recommandée et comment les enregistrer.

| Écran | Aperçu |
|------|--------|
| Accueil (le Seuil) | ![Accueil](docs/demos/02-accueil.png) |
| Création de personnage (sous-classe + historique) | ![Création](docs/demos/03-creation-personnage.gif) |
| Feuille de personnage 2024 | ![Fiche](docs/demos/04-fiche.png) |
| La Table du Maître (+ référence rapide) | ![Table](docs/demos/05-table.gif) |
| Archives & Arsenal | ![Archives](docs/demos/06-archives.png) |

---

## 🧱 Architecture

```
┌──────────────┐   HTTP / SSE    ┌────────────────────┐     ┌──────────────┐
│  Frontend    │  ───────────▶   │     Backend        │ ──▶ │  Ollama      │
│  React/Vite  │  ◀───────────   │     FastAPI        │     │  (HÔTE, GPU) │
│  (nginx)     │   /api proxy    │  auth · chat · RAG │     └──────────────┘
└──────────────┘                 │  campagnes · fiches│
                                 │      MCP servers   │ ──▶  Outils règles / gamemaster
                                 └─────────┬──────────┘
                                           │ SQLModel
                                     ┌─────▼─────┐
                                     │ PostgreSQL│
                                     └───────────┘
```

- **Backend** (`backend/`, FastAPI) : toute la logique — auth JWT, chat LLM en streaming, RAG (FAISS),
  serveurs MCP (règles + gamemaster), campagnes, personnages, bibliothèque, forge.
- **Frontend** (`frontend/`, React + Vite) : l'interface « Grimoire ». En prod, nginx sert le build et
  proxifie `/api` vers le backend.
- **Base de données** : PostgreSQL (SQLite en repli pour le dev local).
- **LLM** : **Ollama tourne sur l'hôte** (pour profiter du GPU) ; le backend l'atteint via
  `host.docker.internal`.

Voir [`ARCHITECTURE.md`](ARCHITECTURE.md) pour le contrat d'API détaillé.

---

## ✅ Prérequis

1. **Git** — pour cloner le dépôt.
2. **Docker** + **Docker Compose** — [installation](https://docs.docker.com/get-docker/).
3. **Ollama** installé **sur votre machine (hôte)** — [ollama.com/download](https://ollama.com/download) —
   avec au moins un modèle téléchargé :
   ```bash
   ollama pull mistral:7b-instruct
   ```
   Vérifiez qu'Ollama tourne : `ollama list` doit afficher le modèle, et `ollama ps` son état.

> 💡 **GPU** : si vous avez une carte NVIDIA avec les pilotes CUDA, Ollama l'utilise automatiquement
> (énorme gain de vitesse). `ollama ps` doit afficher `100% GPU`. Sinon le modèle tourne sur CPU
> (plus lent) — voir la section *Performances*.

---

## 🚀 Installation & démarrage

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-compte>/D-D_AI_Companion.git
cd D-D_AI_Companion

# 2. (Recommandé) définir un secret JWT pour la production
echo "JWT_SECRET=$(openssl rand -hex 32)" > .env

# 3. S'assurer qu'Ollama tourne sur l'hôte avec le modèle
ollama pull mistral:7b-instruct

# 4. Construire et lancer toute la stack
docker compose up --build -d

# 5. Ouvrir l'application
#    Frontend  ->  http://localhost:8080
#    API/docs  ->  http://localhost:8000/docs
#    Grafana   ->  http://localhost:3000  (admin / admin)
#    Prometheus->  http://localhost:9090
```

Pour suivre les logs : `docker compose logs -f backend`
Pour tout arrêter : `docker compose down` (ajoutez `-v` pour effacer aussi la base).

### Première utilisation

1. Ouvrez **http://localhost:8080** et **créez un compte** (page de connexion).
2. Depuis **le Seuil** (accueil) :
   - **Créer un personnage** (choisissez classe, sous-classe et historique → les bonus s'appliquent).
   - **Forger une campagne** (Session Zéro) — vous en devenez le **Maître du Jeu**.
   - **La Table** pour mener la partie et discuter avec le LLM.
3. Retrouvez tout dans **Mon antre** (personnages, fiches, campagnes) et **La Bibliothèque**.

---

## ⚙️ Configuration (variables d'environnement)

À placer dans un fichier `.env` à la racine (lu par `docker compose`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `JWT_SECRET` | `dev-secret-...` | **À changer en production.** Clé de signature des jetons. |
| `DATABASE_URL` | `postgresql+psycopg://dnd:dnd@db:5432/dnd` | Connexion BDD (PostgreSQL en compose). |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | URL d'Ollama (hôte). |
| `OLLAMA_NUM_CTX` | `3072` | Fenêtre de contexte (plus petit = plus rapide). |
| `OLLAMA_NUM_PREDICT` | `1024` | Plafond de tokens générés. |
| `RAG_DEFAULT_SITES` | `https://www.aidedd.org/` | Sites indexés automatiquement au démarrage (séparés par des virgules). |
| `GRAFANA_USER` / `GRAFANA_PASSWORD` | `admin` / `admin` | Identifiants Grafana. |
| `OLLAMA_NUM_THREAD` | `0` (auto) | Threads CPU pour Ollama. |
| `CHAT_HISTORY_MAX_MESSAGES` | `12` | Messages d'historique envoyés au LLM (prefill plus court = plus rapide). |
| `OLLAMA_KEEP_ALIVE` | `30m` | Durée de maintien du modèle en mémoire. |

Les modèles disponibles dans le sélecteur sont définis dans `backend/config.py` (`AVAILABLE_MODELS`).

---

## ⚡ Performances

Le LLM tourne **localement**. Les leviers de vitesse, du plus efficace au moins :

1. **GPU** (sans perte de qualité) — installez les pilotes CUDA ; `ollama ps` doit montrer `100% GPU`.
2. **Modèle plus léger** — un modèle ~3B répond 2–3× plus vite qu'un 7B (qualité un peu moindre).
3. **Réglages intégrés** (déjà actifs) — détection d'intention (n'attache les outils MCP que si nécessaire),
   historique plafonné, modèle gardé en mémoire (`keep_alive`), démarrage non bloquant.
4. **Flash attention** côté hôte : `OLLAMA_FLASH_ATTENTION=1` et `OLLAMA_KV_CACHE_TYPE=q8_0`.

---

## 📊 Monitoring (Prometheus + Grafana)

Le backend expose des métriques Prometheus sur **`/metrics`** (latence, taux de requêtes,
statuts HTTP, requêtes par endpoint). La stack `docker compose` inclut :

- **Prometheus** (`http://localhost:9090`) — collecte les métriques du backend.
- **Grafana** (`http://localhost:3000`, `admin`/`admin`) — un **dashboard pré-provisionné**
  « D&D AI Companion — Backend » (requêtes/s, latence p50/p95/p99, erreurs 4xx/5xx, trafic par endpoint).

Le dashboard et la source de données sont provisionnés automatiquement
(`monitoring/grafana/provisioning/`). Aucune configuration manuelle n'est requise.

---

## 🧑‍💻 Développement local (sans Docker)

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000   # SQLite par défaut

# Frontend (autre terminal)
cd frontend
npm install
npm run dev    # http://localhost:5173 (proxy /api -> :8000)
```

---

## 📁 Structure du projet

```
backend/         API FastAPI (routes, services, modèles SQLModel, config)
frontend/        Interface React/Vite (pages, composants, lib de calcul de fiche)
src/             Serveurs & client MCP (règles, gamemaster)
rag/             Manuel de base indexé + documents uploadés
docs/demos/      Emplacement des captures/GIF du README
docker-compose.yml   Stack db + backend + frontend
ARCHITECTURE.md  Contrat d'API détaillé
```

---

## 🛠️ Dépannage

- **« Oracle injoignable » / pas de réponse au chat** : vérifiez qu'Ollama tourne sur l'hôte
  (`ollama ps`) et que le modèle est téléchargé (`ollama pull mistral:7b-instruct`).
- **Réponses lentes** : voir *Performances* — sur CPU, un 7B reste lent ; privilégiez le GPU ou un modèle plus léger.
- **Inscription impossible** : assurez-vous que le conteneur `db` est « healthy » (`docker compose ps`).
- **Port déjà utilisé** : modifiez les ports publiés dans `docker-compose.yml` (`8080`, `8000`, `5432`).
- **Repartir de zéro** : `docker compose down -v` (efface la base) puis `docker compose up --build -d`.

---

## 📜 Licence

Distribué sous licence MIT — voir [`LICENSE`](LICENSE).
Contenu D&D © Wizards of the Coast ; ce projet est un outil non officiel à usage personnel.
