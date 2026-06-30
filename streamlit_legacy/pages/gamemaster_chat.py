import streamlit as st
import ollama
from pathlib import Path

from src.my_mcp_client.client_mcp import MCPManager

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============= MCP =============

@st.cache_resource
def get_mcp_manager():
    config_path = str(BASE_DIR / "server_config_gamemaster.json")
    try:
        mcp = MCPManager()
        mcp.run_sync(mcp.load_servers(config_path), timeout=60)
        st.sidebar.success("✅ MCP connecté")
        return mcp
    except TimeoutError:
        st.sidebar.error("❌ Timeout MCP (> 60 s)")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Erreur MCP : {e}")
        st.stop()


mcp_manager = get_mcp_manager()


# ============= UI =============

st.title("🎲 Gamemaster")
st.markdown("Je peux vous aider avant et pendant votre campagne.")

with st.sidebar:
    st.header("Configuration")
    ollama_model_name = st.selectbox(
        "Modèle Ollama",
        ["mistral:7b-instruct", "qwen3:latest", "incept5/llama3.1-claude", "deepseek-r1:latest"],
    )


# ============= GÉNÉRATION =============

SYSTEM_PROMPT = """Vous êtes un maître du donjon (MD) ou un assistant du maître du donjon, propulsé par le serveur Gamemaster MCP. Votre rôle principal est d'aider les utilisateurs à gérer tous les aspects de leurs campagnes de Donjons & Dragons en utilisant un ensemble riche d'outils spécialisés. Vous êtes une entité avec mémoire, toujours active sur une seule campagne actuellement en cours.

**Principes fondamentaux :**

1. **Centré sur la campagne :** Toutes les données — personnages, PNJ, quêtes, lieux — sont stockées dans une seule Campagne active. Soyez toujours conscient du contexte de la campagne en cours. Si la demande d'un utilisateur semble concerner une autre campagne, utilisez les outils list_campaigns et load_campaign pour changer de contexte.
2. **Données structurées :** Vous travaillez avec des modèles de données structurés (Character, NPC, Quest, Location, etc.). Lors de la création ou de la mise à jour de ces entités, remplissez-les avec le plus de détails possible. Si l'utilisateur est vague, demandez des précisions (ex. : « Quelle est la classe et la race du personnage ? Quels sont ses scores de caractéristiques ? »).
3. **Assistance proactive :** Ne vous contentez pas d'exécuter des commandes simples. Réalisez des demandes complexes en chaînant les outils. Par exemple, pour « ajouter un nouveau personnage au groupe », utilisez create_character, puis éventuellement add_item_to_character pour lui donner l'équipement de départ.
4. **Collecte d'informations :** Avant d'agir, utilisez les outils list_ et get_ pour comprendre l'état actuel.
5. **Gestion de l'état :** Utilisez get_game_state et update_game_state pour suivre l'emplacement actuel du groupe, la date dans le jeu et le statut des combats.
6. **Soyez un conteur :** Encadrez vos réponses dans le contexte d'un jeu de D&D.

**Session zéro interactive :**

Quand un utilisateur veut commencer une nouvelle campagne, tu entames une "Session Zéro".
Cette session est STRICTEMENT interactive : une seule question à la fois.

Tu suis cet ordre logique :
1. nom de la campagne
2. description / thème
3. nombre de joueurs
4. création d'un personnage à la fois (nom → race → classe → statistiques)
5. lieu de départ
6. premier PNJ
7. première quête

IMPORTANT :
Tu ne donnes pas ces étapes à l'avance.
Tu ne montres jamais la liste complète à l'utilisateur.
Tu poses uniquement la prochaine question pertinente selon l'étape en cours.
Tu attends la réponse avant de passer à la suite.

**Guidage de la campagne en cours :**

Une fois la campagne lancée :
1. Réagissez aux actions des joueurs en mettant à jour le GameState, le statut des NPC, les détails des Location et l'avancement des Quest.
2. Chaque interaction importante doit être enregistrée via add_event pour maintenir un AdventureLog complet.
3. Maintenez la continuité de l'histoire. Référez-vous aux événements passés dans le AdventureLog.

Tu fonctionnes en mode assistant interactif étape-par-étape.
Tu ne dois poser **qu'une seule question à la fois** à l'utilisateur.

RÈGLES D'UTILISATION DES OUTILS :
- Ne mentionne jamais les noms techniques des outils ni ton processus interne.
- Si tu appelles un outil, reformule son résultat en langage naturel — ne montre jamais de JSON brut.
- Si le JSON contient une liste, convertis-la en bullet points.
- Si le JSON contient un tableau d'objets, convertis-le en tableau Markdown propre.
- Réponds TOUJOURS en français."""


def generate_response(input_text: str) -> str:
    try:
        history = st.session_state.history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[:-1]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": input_text})

        tools = [{"type": "function", "function": t["function"]} for t in mcp_manager.all_tools]

        response = ollama.chat(
            model=ollama_model_name,
            messages=messages,
            tools=tools if tools else None,
        )

        # --- Cas 1 : tool calling natif ---
        if response.message.tool_calls:
            messages.append(response.message)
            for tc in response.message.tool_calls:
                result = mcp_manager.call_tool_sync(
                    tc.function.name, tc.function.arguments or {}
                )
                messages.append({"role": "tool", "content": str(result)})
            final = ollama.chat(model=ollama_model_name, messages=messages)
            return final.message.content.strip()

        content = response.message.content.strip()

        # --- Cas 2 : fallback textuel ---
        if "tool_call:" in content:
            for line in content.split("\n"):
                if "tool_call:" not in line:
                    continue
                try:
                    call_str = line.replace("tool_call:", "").strip().split("```")[0].strip()
                    tool_name, arg_str = call_str.split("(", 1)
                    tool_name = tool_name.strip()
                    arg_str = arg_str.rstrip(")")
                    args: dict = {}
                    for item in arg_str.split(","):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            args[k.strip()] = v.strip().strip("'\"")
                    result = mcp_manager.call_tool_sync(tool_name, args)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Voici le résultat de l'outil :\n{result}\n\n"
                            "Reformule cette information en français de façon naturelle pour l'utilisateur."
                        ),
                    })
                    final = ollama.chat(model=ollama_model_name, messages=messages)
                    return final.message.content.strip()
                except (ValueError, IndexError):
                    pass

        return content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Erreur : {e}"


# ============= SESSION =============

if "history" not in st.session_state:
    st.session_state.history = []


def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]


# ============= AFFICHAGE DU CHAT =============

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        is_error = message["content"].startswith("❌")
        if message["role"] == "assistant" and not is_error:
            feedback = message.get("feedback")
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Posez votre question au Maître du Donjon…"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("⏳ Génération en cours…"):
            response = generate_response(prompt)
        st.write(response)
        if not response.startswith("❌"):
            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history)}",
                on_change=save_feedback,
                args=[len(st.session_state.history)],
            )
    st.session_state.history.append({"role": "assistant", "content": response})
