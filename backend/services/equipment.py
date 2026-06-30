"""Catalogue d'objets D&D (SRD) -- via l'API D&D 5e (dnd5eapi.co), avec cache.

Alimente "L'Arsenal" : armes, armures, equipement d'aventurier, outils et objets
magiques. La liste par categorie est peu couteuse ; les details (cout, poids,
proprietes, description) sont charges a la demande. Mode degrade hors ligne :
on renvoie des listes vides plutot que d'echouer.
"""
import httpx

API_BASE = "https://www.dnd5eapi.co/api/2014"

# Categories exposees : (cle, libelle FR, "kind" pour le detail, endpoint liste)
CATEGORIES = [
    ("weapon", "Armes", "equipment", "/equipment-categories/weapon"),
    ("armor", "Armures", "equipment", "/equipment-categories/armor"),
    ("gear", "Equipement d'aventurier", "equipment", "/equipment-categories/adventuring-gear"),
    ("tools", "Outils", "equipment", "/equipment-categories/tools"),
    ("magic", "Objets magiques", "magic", "/magic-items"),
]

_COIN = {"gp": "po", "sp": "pa", "cp": "pc", "ep": "pe", "pp": "pp"}

_cache: dict[str, object] = {}


async def _get(path: str) -> dict | None:
    if path in _cache:
        return _cache[path]
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            r = await client.get(f"{API_BASE}{path}")
            r.raise_for_status()
            data = r.json()
            _cache[path] = data
            return data
    except Exception:  # noqa: BLE001 -- mode degrade
        return None


async def get_catalog() -> dict:
    """Renvoie {available, categories:[{key,label,kind,items:[{index,name}]}]}."""
    categories = []
    available = False
    for key, label, kind, path in CATEGORIES:
        data = await _get(path)
        raw = (data or {}).get("equipment") or (data or {}).get("results") or []
        items = [{"index": it["index"], "name": it["name"]} for it in raw]
        if items:
            available = True
        categories.append({"key": key, "label": label, "kind": kind, "items": items})
    return {"available": available, "categories": categories}


def _fmt_cost(cost: dict | None) -> str:
    if not cost:
        return ""
    unit = _COIN.get(cost.get("unit", ""), cost.get("unit", ""))
    return f"{cost.get('quantity', 0)} {unit}".strip()


async def get_item_detail(kind: str, index: str) -> dict:
    """Detail normalise d'un objet (kind = 'equipment' | 'magic')."""
    path = f"/magic-items/{index}" if kind == "magic" else f"/equipment/{index}"
    d = await _get(path)
    if not d:
        return {"name": index, "category": "", "desc": "Detail indisponible (hors ligne)."}

    desc = d.get("desc")
    if isinstance(desc, list):
        desc = "\n".join(desc)

    out = {
        "name": d.get("name", index),
        "category": (d.get("equipment_category") or {}).get("name", "") if kind != "magic" else "Objet magique",
        "cost": _fmt_cost(d.get("cost")),
        "weight": f"{d['weight']} lb" if d.get("weight") else "",
        "rarity": (d.get("rarity") or {}).get("name", "") if kind == "magic" else "",
        "properties": [p.get("name", "") for p in (d.get("properties") or [])],
        "damage": "",
        "armor_class": "",
        "desc": desc or "",
    }
    dmg = d.get("damage")
    if dmg:
        out["damage"] = f"{dmg.get('damage_dice', '')} {(dmg.get('damage_type') or {}).get('name', '')}".strip()
    ac = d.get("armor_class")
    if ac:
        bonus = " + mod. Dex" if ac.get("dex_bonus") else ""
        out["armor_class"] = f"{ac.get('base', '')}{bonus}"
    return out


# --- Monstres (pour PNJ adversaires) ----------------------------------------
async def get_monsters() -> list[dict]:
    """Liste des monstres SRD : [{index, name}]."""
    data = await _get("/monsters")
    return [{"index": m["index"], "name": m["name"]} for m in (data or {}).get("results", [])]


async def get_monster_detail(index: str) -> dict:
    """Stats clefs d'un monstre : CA, PV, FP (challenge rating)."""
    d = await _get(f"/monsters/{index}")
    if not d:
        return {"name": index, "armor_class": None, "hit_points": None, "challenge_rating": None}
    ac = d.get("armor_class")
    if isinstance(ac, list) and ac:
        ac = ac[0].get("value")
    cr = d.get("challenge_rating")
    # FP fractionnaire usuel (0.125 -> 1/8, 0.25 -> 1/4, 0.5 -> 1/2)
    cr_map = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}
    cr_str = cr_map.get(cr, str(int(cr)) if isinstance(cr, (int, float)) and cr == int(cr) else str(cr)) if cr is not None else None
    return {
        "name": d.get("name", index),
        "armor_class": ac if isinstance(ac, int) else None,
        "hit_points": d.get("hit_points"),
        "challenge_rating": cr_str,
    }


# --- Recherche de regles (index unifie + detail normalise) ------------------
def _join_desc(desc) -> str:
    if isinstance(desc, list):
        return "\n".join(str(x) for x in desc)
    return desc or ""


async def get_rules_index() -> dict:
    """Index plat et leger (nom+index+type) pour la recherche instantanee."""
    items: list[dict] = []

    sp = await _get("/spells")
    items += [{"name": s["name"], "index": s["index"], "type": "spell"} for s in (sp or {}).get("results", [])]

    mo = await _get("/monsters")
    items += [{"name": m["name"], "index": m["index"], "type": "monster"} for m in (mo or {}).get("results", [])]

    cat = await get_catalog()
    for c in cat["categories"]:
        t = "magic" if c["kind"] == "magic" else "equipment"
        items += [{"name": it["name"], "index": it["index"], "type": t} for it in c["items"]]

    co = await _get("/conditions")
    items += [{"name": c["name"], "index": c["index"], "type": "condition"} for c in (co or {}).get("results", [])]

    return {"available": bool(items), "items": items}


async def get_rules_detail(type_: str, index: str) -> dict:
    """Detail normalise : {name, type, meta:[lignes], desc}."""
    if type_ == "spell":
        d = await _get(f"/spells/{index}") or {}
        meta = [
            f"Niveau {d.get('level', 0)}" + (f" · {(d.get('school') or {}).get('name','')}" if d.get('school') else ""),
            f"Incantation : {d.get('casting_time','')}" if d.get('casting_time') else "",
            f"Portee : {d.get('range','')}" if d.get('range') else "",
            f"Duree : {d.get('duration','')}" if d.get('duration') else "",
        ]
        return {"name": d.get("name", index), "type": "spell", "meta": [m for m in meta if m], "desc": _join_desc(d.get("desc"))}

    if type_ == "monster":
        d = await _get(f"/monsters/{index}") or {}
        ac = d.get("armor_class")
        if isinstance(ac, list) and ac:
            ac = ac[0].get("value")
        meta = [
            f"{d.get('size','')} {d.get('type','')}".strip(),
            f"CA {ac}" if ac is not None else "",
            f"PV {d.get('hit_points','')}" if d.get("hit_points") is not None else "",
            f"FP {d.get('challenge_rating','')}" if d.get("challenge_rating") is not None else "",
        ]
        actions = d.get("actions") or []
        desc = "\n\n".join(f"{a.get('name','')} : {a.get('desc','')}" for a in actions[:6])
        return {"name": d.get("name", index), "type": "monster", "meta": [m for m in meta if m], "desc": desc}

    if type_ == "condition":
        d = await _get(f"/conditions/{index}") or {}
        return {"name": d.get("name", index), "type": "condition", "meta": [], "desc": _join_desc(d.get("desc"))}

    # equipment | magic
    det = await get_item_detail("magic" if type_ == "magic" else "equipment", index)
    meta = [x for x in [det.get("category"), det.get("rarity"), det.get("cost"), det.get("weight"),
                        det.get("damage"), (f"CA {det['armor_class']}" if det.get("armor_class") else ""),
                        ", ".join(det.get("properties") or [])] if x]
    return {"name": det.get("name", index), "type": type_, "meta": meta, "desc": det.get("desc", "")}
