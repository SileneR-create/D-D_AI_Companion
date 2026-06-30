"""Service Campagne -- expose l'etat du serveur MCP Gamemaster en JSON structure.

Les outils MCP (`get_campaign_info`, `list_characters`, ...) renvoient du texte
formate. Ce module les appelle puis parse leur sortie pour alimenter le panneau
"Grimoire de campagne" du frontend, sans que celui-ci ait a parler MCP.
"""
import asyncio
import re
from typing import Optional

from backend.schemas import CampaignState, CampaignStateQuest, Companion


def _resolve(manager, original_name: str) -> Optional[str]:
    """Retrouve le nom qualifie (`serveur.outil`) d'un outil MCP."""
    if manager is None or not getattr(manager, "all_tools", None):
        return None
    for t in manager.all_tools:
        if t.get("original_name") == original_name:
            return t["function"]["name"]
    return None


async def _call(manager, original_name: str, args: dict | None = None) -> str:
    """Appelle un outil MCP par son nom court ; renvoie '' en cas d'echec."""
    qualified = _resolve(manager, original_name)
    if not qualified:
        return ""
    try:
        return str(await manager.call_tool_from_external(qualified, args or {}))
    except Exception:  # noqa: BLE001
        return ""


def _parse_kv(text: str) -> dict:
    """Transforme des lignes `**Cle:** valeur` en dictionnaire (cles minuscules)."""
    out = {}
    for m in re.finditer(r"\*\*(.+?):\*\*\s*(.+)", text):
        out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def _to_int(value: str | None) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"-?\d+", value)
    return int(m.group()) if m else None


def _parse_companions(text: str) -> list[Companion]:
    companions: list[Companion] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("**") or not line.startswith(("•", "-", "*")):
            continue
        body = line.lstrip("•-* ").strip()
        # Format attendu : "Nom (Level N Race Classe)"
        m = re.match(r"^(.*?)\s*\(Level\s*(\d+)\s*(.*?)\)\s*$", body)
        if m:
            name, level, rest = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
            detail = f"Niveau {level}" + (f" — {rest}" if rest else "")
            companions.append(Companion(name=name, detail=detail, level=level))
        elif body:
            companions.append(Companion(name=body))
    return companions


def _parse_quests(text: str) -> list[CampaignStateQuest]:
    quests: list[CampaignStateQuest] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("**") or not line.startswith(("•", "-", "*")):
            continue
        body = line.lstrip("•-* ").strip()
        m = re.match(r"^(.*?)\s*\[(.+?)\]\s*$", body)
        if m:
            quests.append(CampaignStateQuest(title=m.group(1).strip(), status=m.group(2).strip()))
        elif body:
            quests.append(CampaignStateQuest(title=body))
    return quests


async def get_campaign_state(manager) -> CampaignState:
    """Construit l'etat synthetique de la campagne active."""
    if manager is None or not manager.all_tools:
        return CampaignState(active=False)

    info_txt, chars_txt, quests_txt, state_txt = await asyncio.gather(
        _call(manager, "get_campaign_info"),
        _call(manager, "list_characters"),
        _call(manager, "list_quests"),
        _call(manager, "get_game_state"),
    )

    # Pas de campagne chargee : les outils renvoient un message explicite.
    if not info_txt or "no active campaign" in info_txt.lower():
        return CampaignState(active=False)

    info = _parse_kv(info_txt)
    state = _parse_kv(state_txt)

    counts = {
        "companions": _to_int(info.get("character count")) or 0,
        "npcs": _to_int(info.get("npc count")) or 0,
        "quests": _to_int(info.get("quest count")) or 0,
        "sessions": _to_int(info.get("session count")) or 0,
    }

    omen = state.get("notes")
    if omen and omen.lower().startswith("no current notes"):
        omen = None

    return CampaignState(
        active=True,
        name=info.get("name") or info.get("campaign"),
        description=info.get("description"),
        party_level=_to_int(info.get("party level")),
        in_combat=str(info.get("in combat", "")).strip().lower() in ("true", "yes", "oui"),
        counts=counts,
        companions=_parse_companions(chars_txt),
        quests=_parse_quests(quests_txt),
        omen=omen,
    )


# --- Actions (wizard de creation) -------------------------------------------
async def mcp_create_campaign(manager, name: str, description: str, dm_name: str | None) -> str:
    """Cree la campagne cote MCP (et la rend active)."""
    return await _call(manager, "create_campaign", {
        "name": name, "description": description, "dm_name": dm_name,
    })


async def mcp_load_campaign(manager, name: str) -> str:
    """Charge une campagne existante comme campagne active du serveur MCP."""
    return await _call(manager, "load_campaign", {"name": name})


async def mcp_create_character(manager, data: dict) -> str:
    """Cree un personnage cote MCP (la campagne active doit etre chargee)."""
    return await _call(manager, "create_character", data)


async def mcp_create_quest(manager, title: str, description: str, giver: str | None, reward: str | None) -> str:
    """Cree une quete cote MCP (la campagne active doit etre chargee)."""
    return await _call(manager, "create_quest", {
        "title": title, "description": description or "", "giver": giver, "reward": reward,
    })


async def mcp_create_npc(manager, name, description, race, occupation, attitude) -> str:
    return await _call(manager, "create_npc", {
        "name": name, "description": description, "race": race,
        "occupation": occupation, "attitude": attitude,
    })


async def mcp_create_location(manager, name, description) -> str:
    return await _call(manager, "create_location", {"name": name, "description": description})


async def mcp_add_item_to_character(manager, character_name, item_name, description, item_type, value, quantity) -> str:
    return await _call(manager, "add_item_to_character", {
        "character_name_or_id": character_name, "item_name": item_name,
        "description": description, "item_type": item_type, "value": value, "quantity": quantity,
    })
