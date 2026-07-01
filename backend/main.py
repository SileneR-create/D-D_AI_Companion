"""Point d'entree de l'API FastAPI du D&D AI Companion."""
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, API_VERSION, CORS_ORIGINS, WARMUP_ON_STARTUP
from backend.db import init_db
from backend.deps import get_current_user
from backend.routes import arsenal, auth, characters, forge, gamemaster, library, rules, solo
from backend.schemas import HealthResponse
from backend.services.chat import warmup
from backend.services.mcp import init_mcp_gamemaster, init_mcp_rules
from backend.services.rag import crawl_default_sites, init_rag


async def _startup_heavy() -> None:
    """Initialisations lourdes (MCP, RAG, warm-up) lancees en arriere-plan.

    Elles ne doivent PAS bloquer le demarrage : le telechargement du modele
    d'embeddings (RAG) et la connexion des serveurs MCP peuvent prendre du temps.
    Pendant ce temps, l'API repond deja (sante, auth) ; le chat/RAG se contentent
    de fonctionner en mode degrade jusqu'a ce que ces services soient prets.
    """
    try:
        await init_mcp_rules()
        await init_mcp_gamemaster()
        await init_rag()
        asyncio.create_task(crawl_default_sites())   # sites par defaut, sans bloquer
        if WARMUP_ON_STARTUP:
            await warmup()
        print("Services (MCP, RAG) prets.")
    except Exception as e:  # noqa: BLE001
        print(f"Initialisation des services en arriere-plan interrompue: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Demarrage D&D AI Companion API...")
    init_db()  # rapide et indispensable a l'auth -> on l'attend.
    asyncio.create_task(_startup_heavy())  # le reste ne bloque pas le service.
    print("API prete (services lourds en cours de chargement).")
    yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

# --- Monitoring Prometheus (expose /metrics, non authentifie) ---------------
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    print("Metrics Prometheus exposees sur /metrics")
except Exception as e:  # noqa: BLE001 -- le backend fonctionne sans monitoring
    print(f"Instrumentation Prometheus indisponible: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
# Les routes metier exigent une authentification ; /api/auth et /api/health non.
_auth = [Depends(get_current_user)]
app.include_router(rules.router, prefix="/api/rules", tags=["rules"], dependencies=_auth)
app.include_router(gamemaster.router, prefix="/api/gamemaster", tags=["gamemaster"], dependencies=_auth)
app.include_router(characters.router, prefix="/api/characters", tags=["characters"], dependencies=_auth)
app.include_router(forge.router, prefix="/api/forge", tags=["forge"], dependencies=_auth)
app.include_router(library.router, prefix="/api/library", tags=["library"], dependencies=_auth)
app.include_router(arsenal.router, prefix="/api/arsenal", tags=["arsenal"], dependencies=_auth)
app.include_router(solo.router, prefix="/api/solo", tags=["solo"], dependencies=_auth)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
