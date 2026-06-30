import streamlit as st
import ollama
import os
import faiss
from pathlib import Path
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.my_mcp_client.client_mcp import MCPManager

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============= RAG =============

def process_pdf_to_vectorstore(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )
        dim = len(embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(dim)
        vs = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        vs.add_texts(chunks)
        return vs
    except Exception as e:
        st.error(f"Erreur traitement PDF : {e}")
        return None


def retrieve_from_rag(query: str, vector_store, k: int = 3) -> str:
    if vector_store is None:
        return ""
    try:
        results = vector_store.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in results)
    except Exception:
        return ""


# ============= MCP =============

@st.cache_resource
def get_mcp_manager():
    config_path = str(BASE_DIR / "server_config.json")
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


# ============= RAG auto-chargement =============

@st.cache_resource
def load_rag_vectorstore():
    pdf_path = BASE_DIR / "rag" / "basic-rules-fr.pdf"
    if not pdf_path.exists():
        st.sidebar.warning(f"⚠️ PDF non trouvé : {pdf_path}")
        return None
    with st.spinner("📚 Chargement du manuel D&D…"):
        with open(pdf_path, "rb") as f:
            vs = process_pdf_to_vectorstore(f)
        if vs:
            st.sidebar.success("✅ Manuel D&D chargé")
        return vs


rag_vector_store = load_rag_vectorstore()


# ============= UI =============

st.title("🐉 D&D Rules")
st.markdown(
    "Je connais les règles basiques de la 5e édition. "
    "Pose-moi une question sur les mécaniques, la création de personnage, les sorts…"
)

with st.sidebar:
    st.header("Configuration")
    ollama_model_name = st.selectbox(
        "Modèle Ollama",
        ["mistral:7b-instruct", "qwen3:latest", "incept5/llama3.1-claude", "deepseek-r1:latest"],
    )


# ============= GÉNÉRATION =============

def generate_response(input_text: str) -> str:
    try:
        rag_context = retrieve_from_rag(input_text, rag_vector_store, k=3)

        system_prompt = f"""Tu es un assistant expert des règles de Donjons & Dragons 5e. Réponds TOUJOURS en français.

EXTRAIT DU MANUEL (D&D 5e règles de base) :
{rag_context if rag_context else "Aucun extrait disponible pour cette question."}

CONSIGNES :
- Appuie-toi sur l'extrait ci-dessus quand il est pertinent.
- Si tu appelles un outil, reformule son résultat en langage naturel — ne montre jamais de JSON brut.
- Ne mentionne jamais les noms techniques des outils ni ton processus interne."""

        # Historique complet (le message courant est déjà dans history_rules[-1])
        history = st.session_state.history_rules
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[:-1]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": input_text})

        # Outils au format OpenAI (sans les clés internes server/original_name)
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

        # --- Cas 2 : fallback textuel (modèles sans function calling) ---
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

if "history_rules" not in st.session_state:
    st.session_state.history_rules = []


def save_feedback(index):
    st.session_state.history_rules[index]["feedback"] = st.session_state[f"feedback_{index}"]


# ============= AFFICHAGE DU CHAT =============

for i, message in enumerate(st.session_state.history_rules):
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

if prompt := st.chat_input("Posez votre question sur D&D…"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history_rules.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("⏳ Génération en cours…"):
            response = generate_response(prompt)
        st.write(response)
        if not response.startswith("❌"):
            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history_rules)}",
                on_change=save_feedback,
                args=[len(st.session_state.history_rules)],
            )
    st.session_state.history_rules.append({"role": "assistant", "content": response})
