"""Ancienne UI Streamlit (legacy) du D&D AI Companion.

Conservee en parallele de la migration vers React. Les pages consomment
directement les memes services (MCP, RAG, Ollama) que le backend FastAPI.

Lancement depuis la RACINE du projet :
    streamlit run streamlit_legacy/run.py
"""
import sys
from pathlib import Path

# La racine du projet doit etre sur sys.path pour importer `src.my_mcp_client`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="D&D AI Companion", layout="wide")
st.title("D&D AI Companion")
st.markdown("Votre assistant intelligent pour Donjons & Dragons 5e")

page = st.navigation([
    st.Page("home.py", title="Accueil"),
    st.Page("pages/dnd_chat.py", title="D&D Rules"),
    st.Page("pages/gamemaster_chat.py", title="Gamemaster"),
])

page.run()
