"""Sorts par classe -- via l'API D&D 5e (dnd5eapi.co), avec cache et mode degrade.

Le formulaire de creation propose les sorts de la classe choisie, filtres selon
le niveau de sort accessible au personnage. En cas d'indisponibilite reseau, on
renvoie une liste vide (la creation reste possible, sans sorts).
"""
import httpx

API_BASE = "https://www.dnd5eapi.co/api/2014"

# Classe (FR, telle qu'affichee) -> index API (EN). Seules les classes
# lanceuses de sorts sont presentes.
CLASS_INDEX = {
    "Barde": "bard",
    "Clerc": "cleric",
    "Druide": "druid",
    "Ensorceleur": "sorcerer",
    "Magicien": "wizard",
    "Paladin": "paladin",
    "Rodeur": "ranger",
    "Occultiste": "warlock",
}
FULL = {"bard", "cleric", "druid", "sorcerer", "wizard"}
HALF = {"paladin", "ranger"}

_cache: dict[str, list[dict]] = {}


def is_caster(class_fr: str) -> bool:
    return class_fr in CLASS_INDEX


def max_spell_level(class_fr: str, level: int) -> int:
    """Niveau de sort maximal accessible (0 = sorts mineurs uniquement)."""
    idx = CLASS_INDEX.get(class_fr)
    if not idx:
        return 0
    if idx in FULL:
        return min(9, (level + 1) // 2)
    if idx in HALF:
        return 0 if level < 2 else min(5, (level - 1) // 4 + 1)
    if idx == "warlock":
        return min(5, (level + 1) // 2)
    return 0


async def _fetch_class_spells(idx: str) -> list[dict]:
    if idx in _cache:
        return _cache[idx]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{API_BASE}/classes/{idx}/spells")
            r.raise_for_status()
            results = r.json().get("results", [])
            spells = [{"index": s["index"], "name": s["name"], "level": s.get("level", 0)} for s in results]
            _cache[idx] = spells
            return spells
    except Exception:  # noqa: BLE001 -- mode degrade hors ligne
        return []


async def get_class_spells(class_fr: str, level: int) -> dict:
    """Renvoie {caster, max_spell_level, spells:[{index,name,level}]} filtre par niveau."""
    if not is_caster(class_fr):
        return {"caster": False, "max_spell_level": 0, "spells": []}
    idx = CLASS_INDEX[class_fr]
    cap = max_spell_level(class_fr, level)
    spells = [s for s in await _fetch_class_spells(idx) if s["level"] <= cap]
    spells.sort(key=lambda s: (s["level"], s["name"]))
    return {"caster": True, "max_spell_level": cap, "spells": spells}


# --- Limites de sorts connus (doit refleter frontend/src/lib/spellLimits.js) -
_WIS = {"Clerc", "Druide", "Rodeur"}
_INT = {"Magicien"}
_CHA = {"Barde", "Ensorceleur", "Occultiste", "Paladin"}

_CANTRIPS = {
    "Barde": [2, 3, 4], "Clerc": [3, 4, 5], "Druide": [2, 3, 4],
    "Ensorceleur": [4, 5, 6], "Occultiste": [2, 3, 4], "Magicien": [3, 4, 5],
}
_KNOWN = {
    "Barde": [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 15, 16, 18, 19, 19, 20, 22, 22, 22],
    "Ensorceleur": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15, 15],
    "Rodeur": [0, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11],
    "Occultiste": [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15],
}


def _cast_mod(class_fr: str, scores: dict) -> int:
    def m(k: str) -> int:
        return (scores.get(k, 10) - 10) // 2
    if class_fr in _WIS:
        return m("wisdom")
    if class_fr in _INT:
        return m("intelligence")
    if class_fr in _CHA:
        return m("charisma")
    return 0


def cantrips_known(class_fr: str, level: int) -> int:
    t = _CANTRIPS.get(class_fr)
    if not t:
        return 0
    return t[2] if level >= 10 else t[1] if level >= 4 else t[0]


def spells_known(class_fr: str, level: int, scores: dict) -> int:
    lvl = max(1, min(20, level))
    if class_fr in _KNOWN:
        return _KNOWN[class_fr][lvl - 1]
    m = _cast_mod(class_fr, scores)
    if class_fr in ("Clerc", "Druide"):
        return max(1, lvl + m)
    if class_fr == "Magicien":
        return 6 + 2 * (lvl - 1)
    if class_fr == "Paladin":
        return 0 if lvl < 2 else max(1, lvl // 2 + m)
    return 0


async def validate_selection(class_fr: str, level: int, scores: dict, selected: list[str]) -> None:
    """Verifie que le nombre de sorts/sorts mineurs choisis respecte les limites."""
    if not selected:
        return
    idx = CLASS_INDEX.get(class_fr)
    if not idx:
        raise ValueError("Cette classe ne lance pas de sorts.")
    levels = {s["index"]: s["level"] for s in await _fetch_class_spells(idx)}
    if not levels:
        return  # liste injoignable (hors ligne) : on ne bloque pas
    cantrips = sum(1 for i in selected if levels.get(i, 1) == 0)
    spells = sum(1 for i in selected if levels.get(i, 1) > 0)
    if cantrips > cantrips_known(class_fr, level):
        raise ValueError(f"Trop de sorts mineurs ({cantrips}/{cantrips_known(class_fr, level)}).")
    if spells > spells_known(class_fr, level, scores):
        raise ValueError(f"Trop de sorts ({spells}/{spells_known(class_fr, level, scores)}).")


# --- Detail d'un sort -------------------------------------------------------
_detail_cache: dict[str, dict] = {}


async def get_spell_detail(index: str) -> dict:
    """Renvoie le detail d'un sort (nom, niveau, ecole, portee, duree, description)."""
    if index in _detail_cache:
        return _detail_cache[index]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{API_BASE}/spells/{index}")
            r.raise_for_status()
            d = r.json()
            detail = {
                "index": d.get("index", index),
                "name": d.get("name", index),
                "level": d.get("level", 0),
                "school": (d.get("school") or {}).get("name"),
                "casting_time": d.get("casting_time"),
                "range": d.get("range"),
                "duration": d.get("duration"),
                "desc": "\n\n".join(d.get("desc", []) or []),
            }
            _detail_cache[index] = detail
            return detail
    except Exception:  # noqa: BLE001
        return {"index": index, "name": index, "level": 0, "desc": "Description indisponible (hors ligne)."}
