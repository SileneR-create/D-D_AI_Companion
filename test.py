import asyncio
from src.my_mcp_client.client import MCPManager

async def main():
    manager = MCPManager()
    try:
        await manager.load_servers("server_config_gamemaster.json")
        print("Connecté !")
    except Exception as e:
        print("Erreur MCP:", e)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
