"""Service MCP -- initialisation et acces aux gestionnaires de serveurs MCP.

Deux gestionnaires distincts : un pour les *Regles* (recherche de sources D&D),
un pour le *Gamemaster* (gestion de campagne). Chacun est un singleton initialise
au demarrage de l'API (voir `main.lifespan`).
"""
from src.my_mcp_client.client_mcp import MCPManager

from backend.config import MCP_GAMEMASTER_CONFIG, MCP_RULES_CONFIG

_mcp_rules: MCPManager | None = None
_mcp_gamemaster: MCPManager | None = None


async def init_mcp_rules() -> None:
    global _mcp_rules
    _mcp_rules = MCPManager()
    try:
        await _mcp_rules.load_servers_from_external(str(MCP_RULES_CONFIG))
        print("MCP DnD Rules connecte")
    except Exception as e:  # noqa: BLE001
        print(f"MCP DnD Rules non disponible: {e}")


async def init_mcp_gamemaster() -> None:
    global _mcp_gamemaster
    _mcp_gamemaster = MCPManager()
    try:
        await _mcp_gamemaster.load_servers_from_external(str(MCP_GAMEMASTER_CONFIG))
        print("MCP Gamemaster connecte")
    except Exception as e:  # noqa: BLE001
        print(f"MCP Gamemaster non disponible: {e}")


def get_mcp_rules() -> MCPManager | None:
    return _mcp_rules


def get_mcp_gamemaster() -> MCPManager | None:
    return _mcp_gamemaster
