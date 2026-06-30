# --- Backend FastAPI + serveurs MCP ---
FROM python:3.12-slim

WORKDIR /app

# Force l'affichage immediat des logs Python (sinon les print() restent en buffer
# et n'apparaissent jamais dans `docker compose logs`).
ENV PYTHONUNBUFFERED=1

# Dependances systeme (build de faiss/sentence-transformers, git, curl)
RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# uv : necessaire pour lancer le serveur MCP "gamemaster" (uv run gamemaster-mcp)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel uv

# Dependances Python du backend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY . .

# Pre-installe le serveur MCP "gamemaster" dans l'environnement Python.
# Sinon `uv run gamemaster-mcp` le *construit au lancement* et ecrit la sortie de
# build sur stdout, ce qui corrompt le protocole MCP (stdio) -> "Connection closed".
# --no-deps : on n'autorise PAS la mise a jour des deps partagees du backend
# (fastmcp/pydantic/mcp deja installees) ; on ajoute seulement ses deps propres.
# Une fois installe, il se lance via la commande `gamemaster-mcp`.
RUN pip install --no-cache-dir --no-deps ./src/mcp_servers/gamemaster-mcp \
 && pip install --no-cache-dir shortuuid python-dotenv

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl --fail http://localhost:8000/api/health || exit 1

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
