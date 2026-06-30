"""Configuration centralisee du backend.

Toutes les constantes (modeles, chemins, CORS, valeurs Ollama) sont definies ici
afin que le reste du code n'ait jamais a les coder en dur.
"""
import os
from pathlib import Path

# Racine du projet (... / D-D_AI_Companion)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Modeles Ollama disponibles ---------------------------------------------
DEFAULT_MODEL = "mistral:7b-instruct"
AVAILABLE_MODELS = [
    "mistral:7b-instruct",
    "qwen3:latest",
    "incept5/llama3.1-claude",
    "deepseek-r1:latest",
]

# --- Performance Ollama ------------------------------------------------------
# keep_alive : duree pendant laquelle le modele reste charge en memoire entre
# deux requetes. Evite le rechargement a froid (plusieurs secondes) a chaque
# message. "-1" = garder indefiniment ; "30m" = 30 minutes.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Taille de la fenetre de contexte. Plus petit = prefill plus rapide.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "3072"))

# Borne le nombre de tokens generes : evite les reponses interminables qui
# donnent l'impression que "le LLM est lent". -1 = illimite.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))

# Nombre de threads CPU utilises par Ollama (0 = auto-detection d'Ollama).
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "0"))

# Options passees a chaque appel ollama.chat.
OLLAMA_OPTIONS = {"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT}
if OLLAMA_NUM_THREAD > 0:
    OLLAMA_OPTIONS["num_thread"] = OLLAMA_NUM_THREAD

# Precharger le modele par defaut au demarrage de l'API (warm-up).
WARMUP_ON_STARTUP = True

# Nombre max de messages d'historique renvoyes au LLM a chaque tour.
# Le *prefill* (lecture du prompt) coute proportionnellement a la longueur :
# tronquer l'historique garde des reponses rapides meme sur une longue partie.
# (l'historique complet reste stocke en base et affiche a l'ecran)
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "12"))

# --- Serveurs MCP -----------------------------------------------------------
MCP_RULES_CONFIG = BASE_DIR / "server_config.json"
MCP_GAMEMASTER_CONFIG = BASE_DIR / "server_config_gamemaster.json"

# --- RAG --------------------------------------------------------------------
RAG_PDF_PATH = BASE_DIR / "rag" / "basic-rules-fr.pdf"
RAG_UPLOAD_DIR = BASE_DIR / "rag" / "uploads"
RAG_INDEX_DIR = BASE_DIR / "rag" / "index"          # index FAISS persiste
RAG_CRAWL_MAX_PAGES = 40                              # pages max lors d'un crawl
# Sites indexes automatiquement au demarrage (une seule fois, en arriere-plan).
RAG_DEFAULT_SITES = [s for s in os.getenv("RAG_DEFAULT_SITES", "https://www.aidedd.org/").split(",") if s.strip()]
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K = 3

# --- API --------------------------------------------------------------------
API_TITLE = "D&D AI Companion API"
API_VERSION = "1.0.0"

# Origines autorisees pour le frontend React (Vite : 5173, CRA : 3000)
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# --- Roles & permissions d'outils -------------------------------------------
# Un joueur n'accede qu'a un sous-ensemble d'outils MCP ; le MD a tout.
# (noms d'outils *non qualifies*, cf. serveur gamemaster-mcp)
PLAYER_ALLOWED_TOOLS = {
    # Lancer de des
    "roll_dice",
    # Creer / gerer SON personnage
    "create_character", "update_character", "add_item_to_character", "get_character",
    # Consulter les elements joues de la campagne (lecture)
    "list_characters", "get_campaign_info", "get_game_state",
    "list_quests", "list_npcs", "get_npc", "list_locations", "get_location",
    "get_sessions", "get_events", "calculate_experience",
}

# --- Base de donnees & authentification -------------------------------------
# PostgreSQL en production (compose) ; SQLite en repli pour le dev local sans BDD.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# JWT : a surcharger via la variable d'env JWT_SECRET en production !
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-a-changer-en-production")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 h

# --- Detection d'intention (vitesse) ----------------------------------------
# N'attache les outils MCP au LLM que si le message exprime une action.
# Evite d'envoyer ~30 schemas d'outils a chaque tour narratif -> prompt plus
# court -> generation plus rapide.
INTENT_TOOL_GATING = True
ACTION_KEYWORDS = {
    "cree", "creer", "crée", "créer", "ajoute", "ajouter", "genere", "génère", "generer", "générer",
    "lance", "lancer", "lancé", "jette", "combat", "attaque", "de", "dé", "des", "dés", "roll", "dice",
    "recherche", "cherche", "chercher", "trouve", "trouver", "liste", "lister", "montre", "affiche",
    "sort", "sorts", "monstre", "monstres", "equipement", "équipement", "tresor", "trésor",
    "pnj", "npc", "quete", "quête", "lieu", "lieux", "niveau", "xp", "experience", "expérience",
    "statut", "etat", "état", "sauvegarde", "attribue", "calcule", "donne", "recrute", "supprime",
    "demarre", "démarre", "tour", "initiative", "session", "evenement", "événement",
}
