import streamlit as st
import ollama
import asyncio
import threading
import json
import os
from queue import Queue
import sys

from src.my_mcp_client.client import MCPManager


# Queue pour communiquer entre threads
mcp_queue = Queue()

def init_mcp_in_thread():
    """Initialiser MCP dans un thread séparé"""
    try:
        print("DEBUG: Starting MCP initialization...", flush=True)
        config_path = "C:\\Users\\Utilisateur\\Desktop\\projet_dnd\\server_config_gamemaster.json"
        
        if not os.path.exists(config_path):
            print(f"DEBUG: Config file not found: {config_path}", flush=True)
            mcp_queue.put(("error", f"Config file not found: {config_path}"))
            return
        
        print("DEBUG: Config file found, loading...", flush=True)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            servers = list(config.get('mcpServers', {}).keys())
            print(f"DEBUG: Servers found: {servers}", flush=True)
        
        print("DEBUG: Creating event loop...", flush=True)
        # FIX: Utiliser ProactorEventLoop sur Windows
        if sys.platform == 'win32':
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("DEBUG: Creating MCPManager...", flush=True)
        mcp_manager = MCPManager()
        
        print("DEBUG: Loading servers...", flush=True)
        loop.run_until_complete(mcp_manager.load_servers(config_path))
        
        print("DEBUG: Success! Servers loaded", flush=True)
        mcp_queue.put(("success", mcp_manager))
        st.sidebar.success(f"✅ MCP connecté: {servers}")
    
    except Exception as e:
        print(f"DEBUG: Exception occurred: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        mcp_queue.put(("error", str(e)))
        st.sidebar.error(f"❌ Erreur MCP: {str(e)}")



@st.cache_resource
def get_mcp_manager():
    """Obtenir le MCPManager (lancé une seule fois)"""
    st.sidebar.info("⏳ Initialisation MCP...")
    
    # Lancer le thread d'initialisation
    thread = threading.Thread(target=init_mcp_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=60)  # Attendre max 60 secondes
    
    if mcp_queue.empty():
        st.sidebar.error("❌ Timeout MCP (> 30s)")
        st.stop()
    
    status, result = mcp_queue.get()
    
    if status == "error":
        st.sidebar.error(f"❌ {result}")
        st.stop()
    
    return result


mcp_manager = get_mcp_manager()

st.title("🎲 Gamemaster")
st.markdown("Je peux vous aidez avant et pendant votre campagne")

ollama_model_name = st.sidebar.selectbox(
    "Modèle Ollama",
    ["mistral:7b-instruct", "qwen3:latest", "incept5/llama3.1-claude", "deepseek-r1:latest"]
)


def call_mcp_tool_sync(tool_name: str, args: dict):
    """Appeler un tool MCP de manière synchrone"""
    try:
        print(f"DEBUG: Calling tool '{tool_name}' with args: {args}", flush=True)
        import sys
        
        # Créer une boucle d'événements pour cet appel
        if sys.platform == 'win32':
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print(f"DEBUG: Event loop created, calling tool...", flush=True)
        # Augmentez le timeout à 120 secondes (2 minutes)
        result = loop.run_until_complete(
            asyncio.wait_for(
                mcp_manager.call_tool(tool_name, args),
                timeout=120.0
            )
        )
        print(f"DEBUG: Tool completed, result length: {len(str(result))}", flush=True)
        loop.close()
        return result
    except asyncio.TimeoutError:
        print(f"DEBUG: Tool timeout after 120 seconds", flush=True)
        return f"❌ Timeout: le tool a pris trop de temps (>120s). Essayez une requête plus spécifique."
    except Exception as e:
        print(f"DEBUG: Error calling tool: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return f"❌ Erreur tool MCP: {str(e)}"


def generate_response(input_text):
    """Générer une réponse avec Ollama"""
    print(f"DEBUG: generate_response called with: {input_text[:50]}", flush=True)  # ← AJOUTEZ CETTE LIGNE

    try:
        # Construire la liste des outils disponibles
        tools_list = "\n".join([
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in mcp_manager.all_tools
        ])
        
        system_prompt = f"""Tu es un assistant D&D expert avec accès à ces outils:
{tools_list}

RÈGLES D’UTILISATION DES OUTILS :
1. Tu NE DOIS JAMAIS expliquer quel tool tu vas utiliser.
2. Tu NE DOIS JAMAIS décrire le fonctionnement d’un tool.
3. Tu NE DOIS PAS dire "nous devons utiliser l’outil…" ou "voici comment je le ferais".
4. Tu n’utilises QUE les outils pour répondre plus précisément à la demande.
5. Tu ne donnes jamais de code block, jamais de backticks.
6. Tu ne fais jamais semblant d’appeler un outil : tu n’écris tool_call: que si tu appelles réellement un tool MCP.
7. Si aucun outil ne peux répondre à la demande, tu réponds normalement en expliquant que tu n'es pas certain de la réponse en français.

Quand tu reçois la réponse d’un tool MCP, tu NE DOIS PAS afficher le JSON brut à l’utilisateur.
Tu dois toujours reformuler la réponse de manière naturelle, en langage humain.
Seul le tool call doit être en JSON (structurel) ; le rendu final NE doit jamais contenir de JSON.
Quand un tool MCP renvoie du JSON, tu ne dois jamais afficher le JSON brut à l’utilisateur.
Tu dois convertir automatiquement la donnée JSON en un format lisible :

- Si le JSON contient une liste, tu la convertis en liste bullet points.
- Si le JSON contient un tableau d’objets, tu convertis en un tableau propre Markdown.
- Sinon, un résumé clair et reformulé

Tu choisis toujours le format de présentation le plus clair selon le contenu."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        response = ollama.chat(model=ollama_model_name, messages=messages)
        content = response.message.content.strip()
        
        print(f"DEBUG: Model response: {content[:200]}", flush=True)

        # Chercher le premier tool_call valide
        if "tool_call:" in content:
            # Extraire la première ligne avec tool_call
            lines = content.split('\n')
            tool_call_line = None
            for line in lines:
                if "tool_call:" in line:
                    tool_call_line = line.strip()
                    break
            
            if tool_call_line:
                print(f"DEBUG: Found tool_call: {tool_call_line}", flush=True)
                try:
                    call_str = tool_call_line.replace("tool_call:", "").strip()
                    # Nettoyer les caractères inutiles
                    call_str = call_str.split('\n')[0].split('```')[0].strip()
                    
                    tool_name, arg_str = call_str.split("(", 1)
                    arg_str = arg_str.rstrip(")")
                    
                    print(f"DEBUG: Tool name: {tool_name}, Args: {arg_str}", flush=True)
                    
                    args = {}
                    if arg_str.strip():
                        for item in arg_str.split(","):
                            if "=" in item:
                                k, v = item.split("=", 1)
                                args[k.strip()] = v.strip().strip("'\"")
                    
                    print(f"DEBUG: Parsed args: {args}", flush=True)
                    tool_result = call_mcp_tool_sync(tool_name, args)
                    return tool_result
                except (ValueError, IndexError) as ve:
                    print(f"DEBUG: Parse error: {ve}", flush=True)
                    return f"❌ Erreur parsing tool_call: {str(ve)}"
        
        return content
    
    except Exception as e:
        print(f"DEBUG: Exception in generate_response: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return f"❌ Erreur Ollama: {str(e)}"


def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]


if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("⏳ Génération en cours..."):
            response = generate_response(prompt)
        st.write(response)
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": response})