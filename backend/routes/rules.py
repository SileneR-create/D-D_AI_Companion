"""Route /api/rules -- assistant Regles D&D 5e (RAG) + gestion des sources."""
import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.config import AVAILABLE_MODELS
from backend.prompts import build_rules_prompt
from backend.schemas import ChatRequest, CrawlRequest, ModelsResponse, RagSourcesResponse, RuleDetail, RulesIndex
from backend.services import equipment, rag
from backend.services.chat import stream_chat
from backend.services.mcp import get_mcp_rules

router = APIRouter()

# En-tetes necessaires pour un flux SSE fiable derriere un proxy.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=AVAILABLE_MODELS)


@router.get("/reference", response_model=RulesIndex)
async def reference_index() -> RulesIndex:
    """Index leger pour la recherche instantanee (sorts, monstres, equipement, conditions)."""
    data = await equipment.get_rules_index()
    return RulesIndex(**data)


@router.get("/reference/{type_}/{index}", response_model=RuleDetail)
async def reference_detail(type_: str, index: str) -> RuleDetail:
    """Fiche detaillee normalisee d'une entree de regle."""
    return RuleDetail(**await equipment.get_rules_detail(type_, index))


@router.get("/sources", response_model=RagSourcesResponse)
def list_sources() -> RagSourcesResponse:
    """Documents actuellement indexes dans le RAG."""
    return RagSourcesResponse(sources=rag.list_sources())


@router.post("/documents", response_model=RagSourcesResponse)
async def upload_document(file: UploadFile = File(...)) -> RagSourcesResponse:
    """Ajoute un document (PDF / txt / md) a l'index RAG."""
    data = await file.read()
    try:
        await rag.add_document(file.filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RagSourcesResponse(sources=rag.list_sources())


@router.post("/crawl", response_model=RagSourcesResponse)
async def crawl(data: CrawlRequest) -> RagSourcesResponse:
    """Aspire tout un site (meme domaine) et l'indexe dans le RAG."""
    try:
        await rag.crawl_site(data.url, data.max_pages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Crawl impossible: {e}")
    return RagSourcesResponse(sources=rag.list_sources())


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    # Le retrieve RAG (embedding CPU) tourne dans un thread pour ne pas bloquer
    # la boucle asyncio pendant la preparation de la reponse.
    rag_context = await asyncio.to_thread(rag.retrieve, req.message)
    system_prompt = build_rules_prompt(rag_context)
    return StreamingResponse(
        stream_chat(req, system_prompt, get_mcp_rules()),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
