"""MCP Server Management - Débug avancé"""
import json
from typing import List, Dict
import os
import httpx
import asyncio
import sys
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPManager:
    """Manager for MCP servers, handling tool definitions and session management."""

    def __init__(self, ollama_url: str = "http://localhost:11434", system_prompt: str = None):
        """Initialize MCP Manager"""
        self.sessions: Dict[str, ClientSession] = {}
        self.all_tools: List[dict] = []
        self.transports: Dict[str, tuple] = {}
        self.ollama_url = ollama_url
        self.system_prompt = system_prompt
        self.http_client = httpx.AsyncClient()

    async def load_servers(self, config_path: str):
        """Load and connect to all MCP servers from config"""
        # Résoudre le chemin absolu du fichier config
        abs_config_path = os.path.abspath(config_path)
        logger.info(f"Loading config from: {abs_config_path}")
        
        if not os.path.exists(abs_config_path):
            logger.error(f"Config file not found: {abs_config_path}")
            raise FileNotFoundError(f"Config file not found: {abs_config_path}")
        
        config_dir = os.path.dirname(abs_config_path)
        with open(abs_config_path, encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Config content: {json.dumps(config, indent=2)}")
        
        for name, server_config in config['mcpServers'].items():
            resolved_config = dict(server_config)
            if 'cwd' not in resolved_config or not resolved_config['cwd']:
                resolved_config['cwd'] = config_dir
    
            try:
                await self._connect_server(name, resolved_config)
            except Exception as e:
                logger.error(f"Error loading server '{name}': {e}")
                # Continuer avec les autres serveurs

    async def _connect_server(self, name: str, config: dict):
        """Connect to a single MCP server"""
        try:
            logger.info(f"🔌 Attempting to connect to '{name}'...")
            logger.debug(f"Command: {config['command']}")
            logger.debug(f"Args: {config['args']}")
            logger.debug(f"CWD: {config.get('cwd')}")
            
            params = StdioServerParameters(
                command=config['command'],
                args=config['args'],
                env=config.get('env'),
                cwd=config.get('cwd')
            )
            
            # Créer le transport avec timeout
            try:
                stdio_transport = await asyncio.wait_for(
                    stdio_client(params).__aenter__(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout connecting to '{name}' after 30 seconds")
                raise
            
            stdio, write = stdio_transport
            logger.debug(f"✓ Transport created for '{name}'")
            
            # Créer et initialiser la session
            session = ClientSession(stdio, write)
            await session.__aenter__()
            logger.debug(f"✓ Session created for '{name}'")
            
            await asyncio.wait_for(session.initialize(), timeout=30.0)
            logger.debug(f"✓ Session initialized for '{name}'")
            
            # Stocker
            self.sessions[name] = session
            self.transports[name] = (stdio_transport, session)
            
            # Récupérer les tools
            meta = await session.list_tools()
            logger.info(f"✅ Connected to '{name}' with {len(meta.tools)} tools")
            
            for tool in meta.tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": f"{name}.{tool.name}",
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    },
                    "server": name,
                    "original_name": tool.name
                }
                self.all_tools.append(tool_def)
                logger.debug(f"  - Tool: {name}.{tool.name}")
        
        except Exception as e:
            logger.error(f"❌ Failed to connect to '{name}': {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # NE PAS relever l'exception, juste logger
            logger.warning(f"⚠️ Continuing without server '{name}'")

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a specific tool by name with provided arguments."""
        tool_info = next((t for t in self.all_tools if t["function"]["name"] == tool_name), None)
        if not tool_info:
            raise ValueError(f"Tool {tool_name} not found. Available tools: {[t['function']['name'] for t in self.all_tools]}")
        
        server_name = tool_info["server"]
        original_name = tool_info["original_name"]
        
        if server_name not in self.sessions:
            raise RuntimeError(f"Session for server '{server_name}' is not connected")
        
        session = self.sessions[server_name]
        
        try:
            logger.info(f"🔧 Calling tool: {tool_name} with args: {arguments}")
            result = await session.call_tool(original_name, arguments)
            logger.info(f"✅ Tool result: {result.content[0].text}")
            return result.content[0].text
        except Exception as e:
            logger.error(f"❌ Error calling tool '{tool_name}': {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup all sessions and close HTTP client."""
        try:
            for name, (transport, session) in self.transports.items():
                await session.__aexit__(None, None, None)
                await transport.__aexit__(None, None, None)
                logger.info(f"Closed connection to '{name}'")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            await self.http_client.aclose()