"""Moteur de chat partage -- orchestration LLM (Ollama) + outils MCP en streaming.

C'est *le fichier backend clair* : toute la logique d'inference, de tool-calling
(natif et fallback textuel) et de production du flux SSE vit ici, a un seul endroit.
Les routes (`routes/rules.py`, `routes/gamemaster.py`) se contentent de fournir
un prompt systeme et un gestionnaire MCP, puis deleguent a `stream_chat`.
"""
import json
import traceback
from typing import AsyncIterator, Optional

import ollama

from backend.config import (
    ACTION_KEYWORDS,
    CHAT_HISTORY_MAX_MESSAGES,
    DEFAULT_MODEL,
    INTENT_TOOL_GATING,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_OPTIONS,
    PLAYER_ALLOWED_TOOLS,
)
from backend.schemas import ChatRequest


def sse(type_: str, **kwargs) -> str:
    """Serialise un evenement au format Server-Sent Events."""
    return f"data: {json.dumps({'type': type_, **kwargs}, ensure_ascii=False)}\n\n"


async def warmup(model: str = DEFAULT_MODEL) -> None:
    """Precharge le modele en memoire pour eviter le demarrage a froid (best-effort)."""
    try:
        client = ollama.AsyncClient()
        await client.chat(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            keep_alive=OLLAMA_KEEP_ALIVE,
            options=OLLAMA_OPTIONS,
        )
        print(f"Modele '{model}' prechauffe (keep_alive={OLLAMA_KEEP_ALIVE})")
    except Exception as e:  # noqa: BLE001
        print(f"Warm-up impossible ({model}): {e}")


def _build_messages(system_prompt: str, req: ChatRequest) -> list[dict]:
    """Construit la liste de messages (system + historique + message courant)."""
    messages = [{"role": "system", "content": system_prompt}]
    # On ne garde que les derniers messages : le prefill (lecture du prompt)
    # coute proportionnellement a sa longueur -> historique court = reponse rapide.
    recent = req.history[-CHAT_HISTORY_MAX_MESSAGES:] if CHAT_HISTORY_MAX_MESSAGES > 0 else req.history
    for m in recent:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})
    return messages


def _wants_tools(message: str) -> bool:
    """Heuristique instantanee : le message demande-t-il une action (outil) ?"""
    if not INTENT_TOOL_GATING:
        return True
    words = set(message.lower().replace("'", " ").split())
    return bool(words & ACTION_KEYWORDS)


def _tools_payload(manager, role: str = "dm", message: str = "") -> Optional[list]:
    """Expose les outils MCP au format Ollama, filtres selon le role de l'utilisateur.

    Le MD (`dm`) a acces a tout ; le joueur (`player`) seulement au sous-ensemble
    autorise (cf. `PLAYER_ALLOWED_TOOLS`).
    """
    if not manager or not manager.all_tools or not _wants_tools(message):
        return None
    tools = manager.all_tools
    if role == "player":
        tools = [t for t in tools if t.get("original_name") in PLAYER_ALLOWED_TOOLS]
    if not tools:
        return None
    return [{"type": "function", "function": t["function"]} for t in tools]


async def _chat(client, model, messages, **kwargs):
    """Wrapper ollama.chat appliquant systematiquement keep_alive + options."""
    return await client.chat(
        model=model,
        messages=messages,
        keep_alive=OLLAMA_KEEP_ALIVE,
        options=OLLAMA_OPTIONS,
        **kwargs,
    )


def _as_dict(args) -> dict:
    """Normalise les arguments d'un appel d'outil (dict ou JSON string) en dict."""
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else {}
        except (ValueError, TypeError):
            return {}
    return args if isinstance(args, dict) else {}


async def stream_chat(
    req: ChatRequest,
    system_prompt: str,
    manager,
    on_tool=None,
) -> AsyncIterator[str]:
    """Genere le flux SSE d'une conversation.

    Args:
        req: requete de chat (message, historique, modele).
        system_prompt: prompt systeme deja construit par la route.
        manager: gestionnaire MCP (ou None si indisponible).
        on_tool: callback synchrone (tool_name, args_dict) appele apres chaque
            outil execute -- permet a la route de refleter l'effet dans la base
            (ex: persister un PNJ cree par le MD dans les Archives).

    Yields:
        Des chaines SSE pretes a etre renvoyees par `StreamingResponse`.
    """
    try:
        messages = _build_messages(system_prompt, req)
        tools = _tools_payload(manager, req.role, req.message)

        client = ollama.AsyncClient()
        full_content = ""
        tool_calls = None

        # --- 1er passage : generation + detection d'appels d'outils ---------
        stream = await _chat(client, req.model, messages, tools=tools, stream=True)
        async for chunk in stream:
            if chunk.message.content:
                full_content += chunk.message.content
                yield sse("token", content=chunk.message.content)
            if getattr(chunk.message, "tool_calls", None):
                tool_calls = chunk.message.tool_calls

        # --- Cas 1 : tool calling natif ------------------------------------
        if tool_calls and manager:
            async for ev in _handle_native_tools(
                client, req.model, messages, full_content, tool_calls, manager, on_tool
            ):
                yield ev

        # --- Cas 2 : fallback textuel (`tool_call: nom(args)`) -------------
        elif "tool_call:" in full_content and manager:
            async for ev in _handle_text_fallback(
                client, req.model, messages, full_content, manager, on_tool
            ):
                yield ev

        yield sse("done")

    except Exception as e:  # noqa: BLE001 -- on remonte l'erreur au client SSE
        traceback.print_exc()
        yield sse("error", content=str(e))


def _fire_on_tool(on_tool, name, args) -> None:
    """Declenche le callback de reflet en base, sans jamais casser le flux."""
    if not on_tool:
        return
    try:
        on_tool(name, _as_dict(args))
    except Exception:  # noqa: BLE001 -- best-effort : un echec de reflet ne doit pas couper le chat
        traceback.print_exc()


async def _handle_native_tools(
    client, model, messages, full_content, tool_calls, manager, on_tool=None
) -> AsyncIterator[str]:
    """Execute les outils detectes nativement puis stream la reponse finale."""
    yield sse("clear")
    messages.append(
        {
            "role": "assistant",
            "content": full_content,
            "tool_calls": [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments or {}}}
                for tc in tool_calls
            ],
        }
    )
    for tc in tool_calls:
        yield sse("tool_call", name=tc.function.name)
        result = await manager.call_tool_from_external(tc.function.name, tc.function.arguments or {})
        messages.append({"role": "tool", "content": str(result)})
        _fire_on_tool(on_tool, tc.function.name, tc.function.arguments or {})

    stream2 = await _chat(client, model, messages, stream=True)
    async for chunk in stream2:
        if chunk.message.content:
            yield sse("token", content=chunk.message.content)


async def _handle_text_fallback(
    client, model, messages, full_content, manager, on_tool=None
) -> AsyncIterator[str]:
    """Parse une ligne `tool_call: nom(arg=val, ...)` pour les modeles sans function calling."""
    for line in full_content.split("\n"):
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

            yield sse("clear")
            yield sse("tool_call", name=tool_name)
            result = await manager.call_tool_from_external(tool_name, args)
            _fire_on_tool(on_tool, tool_name, args)

            messages.append({"role": "assistant", "content": full_content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Resultat de l'outil :\n{result}\n\nReformule en francais pour l'utilisateur.",
                }
            )
            stream2 = await _chat(client, model, messages, stream=True)
            async for chunk in stream2:
                if chunk.message.content:
                    yield sse("token", content=chunk.message.content)
            break
        except (ValueError, IndexError):
            pass
