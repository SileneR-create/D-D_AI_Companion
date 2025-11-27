#!/usr/bin/env python3
"""
D&D Knowledge Navigator - Main server entry point.

This script starts the FastMCP server that provides D&D 5e information
through the Model Context Protocol (MCP).
"""

import logging
import sys
import traceback
import os
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Maintenant importer après avoir ajusté le path
from mcp.server.fastmcp import FastMCP

try:
    from src.core import api_helpers
    from src.core import formatters
    from src.core import prompts
    from src.core import tools
    from src.core import resources
    from src.core.cache import APICache
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    print(f"Script directory: {current_dir}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Configure more detailed logging
log_dir = os.path.join(current_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "dnd_mcp_server.log")

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the D&D Knowledge Navigator server."""
    # Add debug output
    print("=" * 60, file=sys.stderr)
    print("Starting D&D Knowledge Navigator with FastMCP...", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    print(f"Script directory: {current_dir}", file=sys.stderr)
    print(f"Logs will be saved to: {os.path.abspath(log_file)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.stderr.flush()  # Flush stderr to ensure output is visible

    try:
        # Create FastMCP server
        print("Creating FastMCP server...", file=sys.stderr)
        sys.stderr.flush()
        app = FastMCP("dnd-knowledge-navigator")
        print("✓ FastMCP server created successfully", file=sys.stderr)
        sys.stderr.flush()

        # Create shared cache with 24-hour TTL and persistence
        cache_dir = os.path.join(current_dir, "cache")
        cache = APICache(ttl_hours=24, persistent=True, cache_dir=cache_dir)
        print(f"✓ API cache initialized (24-hour TTL, cache in {cache_dir})", file=sys.stderr)
        sys.stderr.flush()

        # Register components
        print("Registering resources...", file=sys.stderr)
        sys.stderr.flush()
        try:
            resources.register_resources(app, cache)
            print("✓ Resources registered", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error registering resources: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise
        sys.stderr.flush()

        print("Registering tools...", file=sys.stderr)
        sys.stderr.flush()
        try:
            tools.register_tools(app, cache)
            print("✓ Tools registered", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error registering tools: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise
        sys.stderr.flush()

        print("Registering prompts...", file=sys.stderr)
        sys.stderr.flush()
        try:
            prompts.register_prompts(app)
            print("✓ Prompts registered", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error registering prompts: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise
        sys.stderr.flush()

        # Run the app
        print("=" * 60, file=sys.stderr)
        print("Running FastMCP app...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.stderr.flush()
        
        app.run()
        
        print("App run completed", file=sys.stderr)
        sys.stderr.flush()
        return 0
    except KeyboardInterrupt:
        print("Server interrupted by user", file=sys.stderr)
        sys.stderr.flush()
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1


# For direct execution, we use the main() function
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)