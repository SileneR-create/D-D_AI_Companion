# Architecture — D&D AI Companion

Ce document decrit l'architecture cible du projet et **le contrat entre le
frontend et le backend**. Il sert de reference pour la migration progressive de
l'ancienne interface Streamlit vers React.

## Vue d'ensemble

L'application est decoupee en trois ensembles independants :

| Couche | Dossier | Role |
| --- | --- | --- |
| **Backend** | `backend/` | API HTTP (FastAPI). Orchestre Ollama, les serveurs MCP et le RAG. Unique source de logique metier. |
| **Frontend** | `frontend/` | Interface React (Vite + Tailwind). Ne contient aucune logique metier : tout passe par l'API. |
| **UI legacy** | `streamlit_legacy/` | Ancienne interface Streamlit, conservee le temps de la migration. Consomme les memes services que le backend. |

Le principe directeur : **toute la logique vit dans le backend**. Le frontend
(React comme Streamlit) n'est qu'un client de l'API. Migrer d'une UI a l'autre
ne touche jamais a la logique.

```
                +-------------------+
                |  Ollama (LLM)     |
                +---------+---------+
                          |
   +----------------------+----------------------+
   |                 backend/ (FastAPI)          |
   |  routes  ->  services/chat  ->  Ollama      |
   |                 |        \                   |
   |          services/mcp   services/rag        |
   +-----+----------------------------+----------+
         | HTTP /api (JSON + SSE)     |
   +-----+-----+              +-------+--------+
   | frontend/ |              | streamlit_     |
   |  (React)  |              |  legacy/       |
   +-----------+              +----------------+
```

## Backend (`backend/`)

Architecture en couches, du plus haut niveau au plus bas :

```
backend/
  main.py            # Application FastAPI, montage des routeurs, cycle de vie
  config.py          # Constantes centralisees (modeles, chemins, CORS)
  schemas.py         # Modeles Pydantic = contrat de donnees de l'API
  prompts.py         # Prompts systeme (Regles, Gamemaster)
  routes/
    rules.py         # Endpoints /api/rules/*    (assistant Regles + RAG)
    gamemaster.py    # Endpoints /api/gamemaster/* (Maitre du Jeu)
  services/
    chat.py          # Moteur de chat partage (LLM + tool-calling + SSE)
    mcp.py           # Connexion et acces aux serveurs MCP
    rag.py           # Indexation FAISS du manuel + recuperation de contexte
```

**Regle d'or :** les routes sont *fines*. Elles preparent un prompt systeme et
un gestionnaire MCP, puis delegent a `services/chat.stream_chat`. Toute la
mecanique d'inference, d'appel d'outils (natif + fallback textuel) et de
production du flux SSE est mutualisee dans `services/chat.py`. Ajouter un nouvel
assistant = une nouvelle route de ~10 lignes + un prompt.

## Frontend (`frontend/`)

L'interface s'inspire de la maquette **GrimoireCompanion** (theme parchemin /
or / braises), decoupee proprement en composants presentationnels et pages.

```
frontend/
  index.html
  vite.config.js       # Proxy /api -> http://localhost:8000 en dev
  src/
    main.jsx           # Point d'entree React
    App.jsx            # Racine : rail + vues, sonde l'API (braise du rail)
    index.css          # Tailwind (reset minimal ; le theme est en JS)
    theme.js           # Palette (T) + typographies (Cinzel / EB Garamond)
    api/
      client.js        # *** Contrat avec le backend *** (le seul a faire fetch)
      index.js
    hooks/
      useChatStream.js # Conversation en streaming (etat + SSE -> messages)
      useModels.js     # Liste des modeles Ollama d'un domaine
    components/
      atmosphere.jsx   # GlobalStyles, Atmosphere (fond), Crest (blason)
      ornaments.jsx    # Filigree, Divider, ScreenTitle, Tag, SectionLabel
      Rail.jsx         # Navigation laterale (Seuil / Grimoire / La Table)
      chat.jsx         # Exchange, Conversation, Inkwell
      panels.jsx       # ParchPanel, SigilPanel, Ledger, ModelSelect, ...
    pages/
      Threshold.jsx    # Accueil (le Seuil)
      Grimoire.jsx     # Regles -> branche sur DOMAINS.RULES
      Table.jsx        # Gamemaster -> branche sur DOMAINS.GAMEMASTER
    mockups/           # Maquettes d'origine (reference de design, non montees)
```

Le branchement sur l'API se fait dans les pages : `useChatStream(domain)` fournit
`messages / draft / streaming / toolName / send`, que `Conversation` traduit en
bulles "Oracle / Chercheur". Les panneaux decoratifs (Ledger, sources) restent
illustratifs ; les brancher sur l'etat reel de campagne (MCP Gamemaster) est la
prochaine etape.

**`src/api/client.js` est le pendant exact de `backend/schemas.py`.** Aucun
composant n'appelle `fetch` directement : tout transite par ce module. C'est le
point unique a maintenir si le contrat evolue.

## Contrat d'API

Base URL en dev : `http://localhost:8000` (proxifie via `/api` par Vite).
`{domain}` vaut `rules` ou `gamemaster`.

| Methode | Endpoint | Corps | Reponse |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{ "status": "ok" }` |
| GET | `/api/{domain}/models` | — | `{ "models": string[] }` |
| GET | `/api/gamemaster/state` | — | `CampaignState` (campagne active) |
| GET | `/api/rules/sources` | — | `{ sources: RagSource[] }` |
| POST | `/api/rules/documents` | fichier (multipart) | `{ sources: RagSource[] }` |
| POST | `/api/{domain}/chat` | `ChatRequest` | flux SSE `text/event-stream` |

### `ChatRequest` (JSON)

```json
{
  "message": "Comment fonctionne l'avantage ?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "model": "mistral:7b-instruct"
}
```

### Flux SSE de `/chat`

La reponse est un flux d'evenements `data: {json}\n\n`. Le champ `type` indique
la nature de chaque evenement :

| `type` | Charge utile | Signification cote UI |
| --- | --- | --- |
| `token` | `{ "content": "..." }` | Fragment de texte a concatener au message en cours. |
| `clear` | — | Vider le message en cours (un outil va etre appele). |
| `tool_call` | `{ "name": "..." }` | Un outil MCP est invoque (afficher un indicateur). |
| `done` | — | Fin du flux : figer le message assistant. |
| `error` | `{ "content": "..." }` | Une erreur est survenue. |

Ces noms sont definis a un seul endroit cote backend
(`schemas.SSEEventType`) et consommes par `client.js > streamChat`. Toute
evolution se fait des deux cotes simultanement.

## Flux d'une requete de chat

1. L'UI appelle `streamChat(domain, { message, history, model }, handlers)`.
2. `routes/{domain}.chat` construit le prompt systeme (+ contexte RAG pour
   `rules`) et delegue a `services/chat.stream_chat`.
3. `stream_chat` interroge Ollama en streaming, detecte d'eventuels appels
   d'outils MCP, les execute, puis stream la reponse finale — le tout sous forme
   d'evenements SSE.
4. `client.js` parse le flux et declenche les callbacks (`onToken`, `onToolCall`,
   `onDone`, ...). Le hook `useChatStream` les traduit en etat React.

## Performance (temps de reponse)

La latence vient surtout du LLM local (Ollama). Leviers cotes backend
(`backend/config.py` + `services/chat.py`) :

- **keep_alive** (`OLLAMA_KEEP_ALIVE`, defaut `30m`) : garde le modele charge en
  memoire entre les requetes et evite le rechargement a froid (plusieurs
  secondes) a chaque message.
- **warm-up au demarrage** (`WARMUP_ON_STARTUP`) : precharge le modele par defaut
  des le lancement de l'API, en tache de fond.
- **RAG non bloquant** : l'embedding de la question (CPU) tourne dans un thread
  (`asyncio.to_thread`) pour ne pas bloquer la boucle pendant la preparation.
- **num_ctx** (`OLLAMA_NUM_CTX`) : fenetre de contexte plus petite = prefill plus
  rapide. A ajuster selon la longueur des conversations.

Pour aller plus loin : choisir un modele plus leger/quantifie (selecteur "Esprit
invoque" dans l'UI), raccourcir le prompt systeme, ou reduire `RAG_TOP_K`.

## Lancer le projet

Backend (depuis la racine) :

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Frontend :

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

UI legacy (optionnelle, depuis la racine) :

```bash
streamlit run streamlit_legacy/run.py
```

Prerequis communs : un serveur **Ollama** local (`http://localhost:11434`) avec
les modeles listes dans `backend/config.py`.

## Etat de la migration & prochaines etapes

L'UI Grimoire (theme parchemin) est branchee sur le backend : les ecrans
**Grimoire** (regles) et **La Table** (gamemaster) dialoguent en streaming via
`useChatStream`. La maquette `DnDCompanionApp` (theme moderne) reste disponible
dans `src/mockups/` comme alternative de design.

Pour ajouter ou rebrancher un ecran :

```jsx
import { useChatStream } from "../hooks/useChatStream.js";
import { DOMAINS } from "../api";

function MonEcran() {
  const { messages, draft, streaming, toolName, send } = useChatStream(DOMAINS.RULES);
  // `draft` = reponse en cours de streaming ; `toolName` = outil MCP en cours.
  ...
}
```

Restent a faire :

1. Brancher les panneaux de campagne (Ledger, sources) sur l'etat reel du MCP
   Gamemaster plutot que sur des donnees illustratives.
2. Persister l'historique de conversation (actuellement en memoire de session).
3. Une fois l'UI React validee en usage, retirer `streamlit_legacy/`.

> Note : `maquette_frontend/` (a la racine) et `frontend/src/mockups/` conservent
> les prototypes d'origine comme reference de design.


## Docker

Deux services orchestres par `docker-compose.yml` ; **Ollama tourne sur l'hote**
(GPU) et le backend l'atteint via `host.docker.internal`.

- `backend` : image FastAPI (+ serveurs MCP, `uv` installe), port 8000. Variable
  `OLLAMA_HOST` pointant vers l'hote. Volume `rag/uploads` persiste les documents.
- `frontend` : build Vite servi par nginx (port 8080). nginx proxifie `/api`
  vers `backend:8000` (meme origine cote navigateur, pas de CORS ; `proxy_buffering
  off` pour le streaming SSE).

```bash
# Prerequis : Ollama lance sur l'hote avec le modele voulu.
docker compose up --build
# Frontend : http://localhost:8080   |   API : http://localhost:8000/api/health
```

## Roles (joueur / MD)

`ChatRequest.role` (`dm` | `player`) filtre les outils MCP exposes au LLM
(`PLAYER_ALLOWED_TOOLS` dans `config.py`). Le MD a tout ; le joueur a un
sous-ensemble (lancer de des, gestion de SON personnage, lecture de la campagne,
regles). En phase 2, ce role decoulera de l'authentification et de la propriete
de campagne (le createur d'une campagne en est le MD).


## Authentification

Comptes utilisateurs avec JWT. PostgreSQL en production (compose), SQLite en
repli pour le dev local (`DATABASE_URL`).

- Modeles (`backend/models.py`) : `User`, `Campaign` (appartient a un MD),
  `CampaignMember` (role par campagne + 1 personnage / joueur).
- `backend/security.py` : hachage bcrypt + jetons JWT (`JWT_SECRET`).
- Routes `POST /api/auth/register`, `POST /api/auth/login` (-> jeton),
  `GET /api/auth/me`.
- Les routes `/api/rules/*` et `/api/gamemaster/*` exigent un jeton
  (`Depends(get_current_user)`) ; `/api/health` et `/api/auth/*` restent ouvertes.
- Cote frontend : `AuthProvider` + `AuthScreen` (login/inscription), jeton
  stocke en localStorage et ajoute en en-tete `Authorization` par `client.js`.

Prochaine etape : deriver le role chat (`dm`/`player`) de la propriete de
campagne (createur = MD) au lieu de la valeur envoyee par le client.


## Wizard de creation (campagnes & personnages)

Parcours guide en 3 etapes (`frontend/src/pages/Wizard.jsx`) :
1. **Type d'aventure** -> `POST /api/gamemaster/campaigns` : cree la campagne en
   base (`dm_id = utilisateur`), l'enregistre cote MCP (`create_campaign`) et
   ajoute le createur comme membre **MD**.
2. **Personnages** -> `POST /api/gamemaster/characters` : formulaire structure
   (race, classe, niveau, caracteristiques) sauvegarde via `create_character`.
   Regle : 1 joueur = 1 personnage / campagne (le MD peut en creer plusieurs).
3. **Lancement** -> bascule sur La Table.

Le role chat (`dm`/`player`) provient desormais de la campagne active
(`ActiveCampaignContext`) : le createur est MD et a acces a tous les outils ; un
joueur n'a que le sous-ensemble autorise. Endpoints associes :
`GET /api/gamemaster/campaigns`, `POST /api/gamemaster/campaigns/{id}/activate`.


## Sorts par classe (creation de personnage)

Les classes lanceuses (barde, clerc, druide, ensorceleur, magicien, paladin,
rodeur, occultiste) proposent une selection de sorts a la creation. Le niveau de
sort maximal depend de la classe et du niveau du personnage (`services/spells.py`
-> `max_spell_level`). Les listes sont recuperees depuis l'API D&D 5e
(dnd5eapi.co), avec cache memoire et mode degrade hors ligne (liste vide).

Endpoint : `GET /api/characters/spells?cls=Magicien&level=3`. Les sorts choisis
sont stockes dans `Character.spells` (JSON) et affiches dans "Mon antre".
