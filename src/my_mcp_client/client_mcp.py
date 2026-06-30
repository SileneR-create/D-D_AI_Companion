"""MCP Server Management — boucle asyncio persistante par instance.

IMPORTANT (concurrence structuree anyio) :
Les context managers `stdio_client(...)` et `ClientSession(...)` ouvrent des
*cancel scopes* anyio. Ils DOIVENT etre ouverts ET fermes dans la MEME tache.
On garde donc, pour chaque serveur, une tache de fond unique et persistante qui
ouvre les deux `async with` et les maintient ouverts jusqu'a l'arret (sinon :
"Attempted to exit cancel scope in a different task than it was entered in").
"""
import json
import threading
import asyncio
import sys
import os
from typing import List, Dict
import httpx
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPManager:
    """Gestionnaire de serveurs MCP avec une boucle asyncio dediee et persistante."""

    def __init__(self, ollama_url: str = "http://localhost:11434", system_prompt: str = None):
        self.sessions: Dict[str, ClientSession] = {}
        self.all_tools: List[dict] = []
        self.ollama_url = ollama_url
        self.system_prompt = system_prompt
        self.http_client = httpx.AsyncClient()

        # Taches de fond + signaux d'arret par serveur.
        self._runners: Dict[str, asyncio.Task] = {}
        self._stops: Dict[str, asyncio.Event] = {}

        # Boucle dediee tournant dans un thread de fond.
        if sys.platform == "win32":
            self._loop = asyncio.ProactorEventLoop()
        else:
            self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # API synchrone (Streamlit / threads quelconques)
    # ------------------------------------------------------------------
    def run_sync(self, coro, timeout: float = 120):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def call_tool_sync(self, tool_name: str, arguments: dict, timeout: float = 120):
        return self.run_sync(self.call_tool(tool_name, arguments), timeout=timeout)

    # ------------------------------------------------------------------
    # API utilisable depuis une boucle asyncio externe (FastAPI)
    # ------------------------------------------------------------------
    async def load_servers_from_external(self, config_path: str):
        future = asyncio.run_coroutine_threadsafe(self.load_servers(config_path), self._loop)
        return await asyncio.wrap_future(future)

    async def call_tool_from_external(self, tool_name: str, arguments: dict):
        future = asyncio.run_coroutine_threadsafe(self.call_tool(tool_name, arguments), self._loop)
        return await asyncio.wrap_future(future)

    # ------------------------------------------------------------------
    # Interne (s'execute sur self._loop)
    # ------------------------------------------------------------------
    async def load_servers(self, config_path: str):
        abs_path = os.path.abspath(config_path)
        logger.info(f"Loading MCP config: {abs_path}")
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Config not found: {abs_path}")

        config_dir = os.path.dirname(abs_path)
        with open(abs_path, encoding="utf-8") as f:
            config = json.load(f)

        for name, server_cfg in config["mcpServers"].items():
            resolved = dict(server_cfg)
            cwd = resolved.get("cwd")
            if not cwd:
                resolved["cwd"] = config_dir
            elif not os.path.isabs(cwd):
                resolved["cwd"] = os.path.normpath(os.path.join(config_dir, cwd))
            try:
                await self._connect_server(name, resolved)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Server '{name}' failed to load: {e}")

    async def _connect_server(self, name: str, config: dict):
        """Lance la tache persistante du serveur et attend qu'il soit pret."""
        logger.info(f"🔌 Connecting to '{name}' (cwd={config.get('cwd')})…")
        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
            cwd=config.get("cwd"),
        )
        ready: asyncio.Future = self._loop.create_future()
        stop = asyncio.Event()
        self._stops[name] = stop
        self._runners[name] = self._loop.create_task(self._run_server(name, params, ready, stop))
        try:
            await asyncio.wait_for(asyncio.shield(ready), timeout=45.0)
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Cannot connect to '{name}': {e}")
            logger.warning(f"⚠️  Continuing without '{name}'")

    async def _run_server(self, name: str, params: StdioServerParameters,
                          ready: asyncio.Future, stop: asyncio.Event):
        """Maintient ouverts stdio + session pour TOUTE la duree de vie (meme tache)."""
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                    meta = await session.list_tools()
                    self.sessions[name] = session
                    for tool in meta.tools:
                        self.all_tools.append({
                            "type": "function",
                            "function": {
                                "name": f"{name}.{tool.name}",
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                            "server": name,
                            "original_name": tool.name,
                        })
                    logger.info(f"✅ '{name}' — {len(meta.tools)} tools")
                    if not ready.done():
                        ready.set_result(True)
                    await stop.wait()   # garde le contexte ouvert jusqu'a l'arret
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Server task '{name}' stopped: {e}")
            if not ready.done():
                ready.set_exception(e)
        finally:
            self.sessions.pop(name, None)

    async def call_tool(self, tool_name: str, arguments: dict):
        tool_info = next(
            (t for t in self.all_tools if t["function"]["name"] == tool_name), None
        )
        if not tool_info:
            available = [t["function"]["name"] for t in self.all_tools]
            raise ValueError(f"Tool '{tool_name}' not found. Available: {available}")

        server_name = tool_info["server"]
        original_name = tool_info["original_name"]
        session = self.sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"Session for '{server_name}' is not connected")

        logger.info(f"🔧 {tool_name}({arguments})")
        result = await session.call_tool(original_name, arguments)
        logger.info("✅ Tool returned a result")
        return result.content[0].text

    async def cleanup(self):
        """Signale l'arret aux taches (qui ferment leurs contextes dans LEUR tache)."""
        for stop in self._stops.values():
            stop.set()
        for task in self._runners.values():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
        await self.http_client.aclose()
