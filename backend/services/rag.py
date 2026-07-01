"""Service RAG -- indexation de documents + crawl de site, avec index persistant.

Optimisations vitesse :
- Index FAISS *persiste sur disque* (RAG_INDEX_DIR) : au demarrage on recharge
  l'index au lieu de tout re-embedder, sauf si les sources/modele ont change.
- Modele d'embeddings rapide (MiniLM multilingue).
"""
import asyncio
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

# NB : faiss / langchain / HuggingFaceEmbeddings (torch) sont importes PARESSEUSEMENT
# dans les fonctions ci-dessous. Sinon leur chargement (plusieurs secondes) bloquerait
# le demarrage d'uvicorn -> nginx renvoie 502 tant que le backend n'ecoute pas.

from backend.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CRAWL_MAX_PAGES,
    RAG_DEFAULT_SITES,
    RAG_EMBEDDING_MODEL,
    RAG_INDEX_DIR,
    RAG_PDF_PATH,
    RAG_TOP_K,
    RAG_UPLOAD_DIR,
)
from backend.schemas import RagSource

SUPPORTED_EXT = {".pdf", ".txt", ".md"}
_MANIFEST = RAG_INDEX_DIR / "manifest.json"

_vectorstore = None          # FAISS, charge paresseusement
_embeddings = None           # HuggingFaceEmbeddings, charge paresseusement
_sources: list[RagSource] = []


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=RAG_EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    return _embeddings


def _empty_store():
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    embeddings = _get_embeddings()
    dim = len(embeddings.embed_query("test"))
    return FAISS(embedding_function=embeddings, index=faiss.IndexFlatL2(dim),
                 docstore=InMemoryDocstore({}), index_to_docstore_id={})


def _source_files() -> list[Path]:
    files = [RAG_PDF_PATH] if RAG_PDF_PATH.exists() else []
    if RAG_UPLOAD_DIR.exists():
        files += [f for f in sorted(RAG_UPLOAD_DIR.iterdir()) if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
    return files


def _fingerprint() -> dict:
    """Empreinte des sources + modele : si elle change, on reconstruit l'index."""
    return {
        "model": RAG_EMBEDDING_MODEL,
        "files": {f.name: f.stat().st_size for f in _source_files()},
    }


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from PyPDF2 import PdfReader
        with open(path, "rb") as f:
            return "".join(p.extract_text() or "" for p in PdfReader(f).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _index_file(path: Path, label: str | None = None) -> RagSource:
    global _vectorstore
    from langchain_text_splitters import CharacterTextSplitter
    if _vectorstore is None:
        _vectorstore = _empty_store()
    splitter = CharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
    chunks = splitter.split_text(_extract_text(path))
    if chunks:
        _vectorstore.add_texts(chunks)
    source = RagSource(name=label or path.name, chunks=len(chunks))
    _sources.append(source)
    return source


def _save_index() -> None:
    try:
        RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore.save_local(str(RAG_INDEX_DIR))
        _MANIFEST.write_text(json.dumps({
            **_fingerprint(),
            "sources": [{"name": s.name, "chunks": s.chunks} for s in _sources],
        }), encoding="utf-8")
        print("RAG : index sauvegarde sur disque")
    except Exception as e:  # noqa: BLE001
        print(f"RAG : sauvegarde index impossible: {e}")


def _try_load() -> bool:
    """Recharge l'index persiste si l'empreinte correspond. True si charge."""
    global _vectorstore, _sources
    if not _MANIFEST.exists():
        return False
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("model") != RAG_EMBEDDING_MODEL or manifest.get("files") != _fingerprint()["files"]:
            return False
        from langchain_community.vectorstores import FAISS
        _vectorstore = FAISS.load_local(str(RAG_INDEX_DIR), _get_embeddings(), allow_dangerous_deserialization=True)
        _sources = [RagSource(**s) for s in manifest.get("sources", [])]
        print(f"RAG : index recharge depuis le disque ({len(_sources)} source(s))")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"RAG : rechargement index impossible, reconstruction. ({e})")
        return False


def _build_index() -> None:
    global _vectorstore, _sources
    if _try_load():
        return
    _vectorstore = _empty_store()
    _sources = []
    for f in _source_files():
        _index_file(f)
        print(f"RAG : indexe {f.name}")
    _save_index()


async def init_rag() -> None:
    print("Chargement des sources RAG...")
    await asyncio.get_event_loop().run_in_executor(None, _build_index)
    print(f"RAG pret ({len(_sources)} source(s))")


async def add_document(filename: str, data: bytes) -> RagSource:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Format non supporte : {ext}. Acceptes : {', '.join(sorted(SUPPORTED_EXT))}")
    RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAG_UPLOAD_DIR / Path(filename).name
    dest.write_bytes(data)
    src = await asyncio.get_event_loop().run_in_executor(None, _index_file, dest, None)
    await asyncio.get_event_loop().run_in_executor(None, _save_index)
    return src


# --- Crawl d'un site --------------------------------------------------------
def _html_to_text_and_links(html: str, base_url: str, domain: str) -> tuple[str, list[str]]:
    try:
        from bs4 import BeautifulSoup
    except Exception:  # noqa: BLE001
        return "", []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    links = []
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"]).split("#")[0]
        if urlparse(u).netloc == domain and u.startswith("http"):
            links.append(u)
    return text, links


def _crawl_blocking(start_url: str, max_pages: int) -> RagSource:
    domain = urlparse(start_url).netloc
    seen: set[str] = set()
    queue = [start_url]
    texts: list[str] = []
    with httpx.Client(timeout=15.0, follow_redirects=True, headers={"User-Agent": "DnDCompanion/1.0"}) as client:
        while queue and len(seen) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                r = client.get(url)
                if "html" not in r.headers.get("content-type", ""):
                    continue
                text, links = _html_to_text_and_links(r.text, url, domain)
            except Exception:  # noqa: BLE001
                continue
            if text:
                texts.append(text)
            for link in links:
                if link not in seen and link not in queue:
                    queue.append(link)
    combined = "\n\n".join(texts)
    RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAG_UPLOAD_DIR / f"crawl-{domain}.txt"
    dest.write_text(combined, encoding="utf-8")
    src = _index_file(dest, f"{domain} ({len(seen)} pages)")
    _save_index()
    return src


async def crawl_site(start_url: str, max_pages: int | None = None) -> RagSource:
    """Aspire un site entier (meme domaine) et l'indexe dans le RAG."""
    if not start_url.startswith(("http://", "https://")):
        raise ValueError("URL invalide (doit commencer par http:// ou https://).")
    limit = min(max_pages or RAG_CRAWL_MAX_PAGES, 200)
    return await asyncio.get_event_loop().run_in_executor(None, _crawl_blocking, start_url, limit)


def list_sources() -> list[RagSource]:
    return list(_sources)


def retrieve(query: str, k: int = RAG_TOP_K) -> str:
    if _vectorstore is None:
        return ""
    try:
        return "\n".join(d.page_content for d in _vectorstore.similarity_search(query, k=k))
    except Exception:  # noqa: BLE001
        return ""


async def crawl_default_sites() -> None:
    """Indexe les sites par defaut (ex: aidedd.org) une seule fois, en arriere-plan.

    Best-effort : si le crawl echoue (reseau) on ignore ; si le site est deja
    indexe (fichier crawl present), on ne recommence pas.
    """
    for url in RAG_DEFAULT_SITES:
        try:
            domain = urlparse(url).netloc
            dest = RAG_UPLOAD_DIR / f"crawl-{domain}.txt"
            if dest.exists():
                print(f"RAG : {domain} deja indexe, crawl ignore")
                continue
            print(f"RAG : indexation du site {url} (arriere-plan)...")
            src = await crawl_site(url)
            print(f"RAG : {url} indexe ({src.chunks} fragments)")
        except Exception as e:  # noqa: BLE001
            print(f"RAG : crawl de {url} impossible: {e}")
