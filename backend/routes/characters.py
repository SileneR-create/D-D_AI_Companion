"""Route /api/characters -- fiches de personnage (avec ou sans campagne) + sorts."""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from backend.db import get_session
from backend.deps import get_current_user
from backend.models import Campaign, CampaignElement, CampaignMember, Character, Item, ItemAssignment, User
from backend.rules5e import ABILITIES, validate_scores
from backend.schemas import (
    CharacterCreate,
    CharacterMoneyUpdate,
    CharacterRead,
    CharacterSheetUpdate,
    CharactersResponse,
    ClassSpellsResponse,
    SpellDetail,
)
from backend.services import campaign as camp
from backend.services import spells as spellsvc
from backend.services.mcp import get_mcp_gamemaster

router = APIRouter()


def _json_list(raw) -> list:
    try:
        v = json.loads(raw) if raw else []
        return v if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


def _to_read(c: Character, campaign_name: str | None = None, campaigns: list[int] | None = None) -> CharacterRead:
    return CharacterRead(
        id=c.id, name=c.name, race=c.race, character_class=c.character_class,
        class_level=c.class_level, alignment=c.alignment, background=c.background,
        description=c.description, strength=c.strength, dexterity=c.dexterity,
        constitution=c.constitution, intelligence=c.intelligence, wisdom=c.wisdom,
        charisma=c.charisma, gold=c.gold or 0, silver=c.silver or 0, copper=c.copper or 0,
        electrum=c.electrum or 0, platinum=c.platinum or 0,
        subclass=c.subclass, size=c.size, speed=c.speed if c.speed is not None else 9,
        max_hp=c.max_hp or 0, current_hp=c.current_hp or 0, temp_hp=c.temp_hp or 0,
        hit_dice_used=c.hit_dice_used or 0, death_succ=c.death_succ or 0, death_fail=c.death_fail or 0,
        inspiration=bool(c.inspiration), armor_class=c.armor_class, spell_ability=c.spell_ability,
        save_proficiencies=_json_list(c.save_proficiencies),
        skill_proficiencies=_json_list(c.skill_proficiencies),
        languages=c.languages, feats=c.feats, species_traits=c.species_traits,
        class_features=c.class_features, other_proficiencies=c.other_proficiencies,
        personality=c.personality,
        spells=_json_list(c.spells), campaign_id=c.campaign_id, campaign_name=campaign_name,
        campaigns=campaigns or [],
    )


@router.get("/spells", response_model=ClassSpellsResponse)
async def class_spells(cls: str, level: int = 1) -> ClassSpellsResponse:
    """Sorts disponibles pour une classe a un niveau donne (vide si non-lanceur)."""
    data = await spellsvc.get_class_spells(cls, level)
    return ClassSpellsResponse(**data)


@router.get("/spell/{index}", response_model=SpellDetail)
async def spell_detail(index: str) -> SpellDetail:
    """Detail d'un sort (pour aider au choix)."""
    return SpellDetail(**await spellsvc.get_spell_detail(index))


@router.post("", response_model=CharacterRead, status_code=201)
async def create_character(
    data: CharacterCreate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CharacterRead:
    """Cree un personnage. La campagne est facultative (tout le monde peut en creer)."""
    scores = {ab: getattr(data, ab) for ab in ABILITIES}
    try:
        validate_scores(data.class_level, scores)
        await spellsvc.validate_selection(data.character_class, data.class_level, scores, data.spells)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Verifie l'appartenance si une campagne est fournie.
    campaign = None
    if data.campaign_id is not None:
        campaign = session.get(Campaign, data.campaign_id)
        member = session.exec(
            select(CampaignMember).where(
                CampaignMember.campaign_id == data.campaign_id,
                CampaignMember.user_id == user.id,
            )
        ).first()
        if not campaign or not member:
            raise HTTPException(status_code=404, detail="Campagne introuvable ou acces refuse.")

    # Applique le bonus de caracteristiques de l'historique (2024) par-dessus le point-buy.
    final = dict(scores)
    for ab, bonus in (data.ability_bonus or {}).items():
        if ab in final:
            final[ab] = final[ab] + max(0, min(2, int(bonus)))

    character = Character(
        owner_id=user.id, campaign_id=data.campaign_id,
        name=data.name, race=data.race, character_class=data.character_class,
        class_level=data.class_level, alignment=data.alignment, background=data.background,
        description=data.description, subclass=data.subclass,
        gold=data.gold, silver=data.silver, copper=data.copper,
        skill_proficiencies=json.dumps(data.skill_proficiencies) if data.skill_proficiencies else None,
        feats=data.feats or None, other_proficiencies=data.other_proficiencies or None,
        spells=json.dumps(data.spells) if data.spells else None,
        **final,
    )
    session.add(character)
    session.commit()
    session.refresh(character)

    # Inventaire de depart : attribue les objets de la Forge choisis (si a l'utilisateur).
    if data.starting_items:
        for iid in data.starting_items:
            it = session.get(Item, iid)
            if it and it.owner_id == user.id:
                session.add(ItemAssignment(item_id=iid, owner_id=user.id,
                                           target_type="character", target_id=character.id))
        session.commit()

    if campaign is not None:
        manager = get_mcp_gamemaster()
        await camp.mcp_load_campaign(manager, campaign.name)
        await camp.mcp_create_character(manager, {
            "name": data.name, "race": data.race, "character_class": data.character_class,
            "class_level": data.class_level, "player_name": user.username,
            "background": data.background, "alignment": data.alignment, **final,
        })

    return _to_read(character, campaign.name if campaign else None)


@router.get("", response_model=CharactersResponse)
def list_characters(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CharactersResponse:
    """Liste les personnages de l'utilisateur."""
    chars = session.exec(
        select(Character).where(Character.owner_id == user.id).order_by(Character.created_at.desc())
    ).all()
    # Campagnes que l'utilisateur peut reellement voir (membre OU createur).
    member_ids = set(session.exec(
        select(CampaignMember.campaign_id).where(CampaignMember.user_id == user.id)).all())
    accessible = {c.id: c.name for c in session.exec(select(Campaign)).all()
                  if c.id in member_ids or c.dm_id == user.id}
    # Un perso rattache a une campagne inaccessible (supprimee, autre compte) est
    # detache : il redevient "sans campagne" et pourra etre re-recrute ailleurs.
    healed = False
    for c in chars:
        if c.campaign_id and c.campaign_id not in accessible:
            c.campaign_id = None
            session.add(c)
            healed = True
    if healed:
        session.commit()
    # Campagnes ou chaque PJ participe (relation N-N via CampaignElement).
    elems = session.exec(select(CampaignElement).where(
        CampaignElement.owner_id == user.id, CampaignElement.element_type == "character")).all()
    by_char: dict[int, list[int]] = {}
    for e in elems:
        if e.campaign_id in accessible:
            by_char.setdefault(e.element_id, []).append(e.campaign_id)
    return CharactersResponse(characters=[
        _to_read(c, accessible.get(c.campaign_id), by_char.get(c.id, [])) for c in chars
    ])


@router.get("/{character_id}", response_model=CharacterRead)
def get_character(
    character_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CharacterRead:
    c = session.get(Character, character_id)
    if not c or c.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Personnage introuvable.")
    cname = None
    if c.campaign_id:
        camp_row = session.get(Campaign, c.campaign_id)
        cname = camp_row.name if camp_row else None
    cids = [e.campaign_id for e in session.exec(select(CampaignElement).where(
        CampaignElement.owner_id == user.id, CampaignElement.element_type == "character",
        CampaignElement.element_id == c.id)).all()]
    return _to_read(c, cname, cids)


@router.patch("/{character_id}/sheet", response_model=CharacterRead)
def update_sheet(
    character_id: int,
    data: CharacterSheetUpdate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CharacterRead:
    """Met a jour les champs editables de la feuille de personnage."""
    c = session.get(Character, character_id)
    if not c or c.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Personnage introuvable.")
    payload = data.model_dump(exclude_unset=True)
    for field in ("save_proficiencies", "skill_proficiencies"):
        if field in payload:
            payload[field] = json.dumps(payload[field] or [])
    for k, v in payload.items():
        setattr(c, k, v)
    session.add(c); session.commit(); session.refresh(c)
    cname = None
    if c.campaign_id:
        camp_row = session.get(Campaign, c.campaign_id)
        cname = camp_row.name if camp_row else None
    return _to_read(c, cname)


@router.patch("/{character_id}/money", response_model=CharacterRead)
def update_money(
    character_id: int,
    data: CharacterMoneyUpdate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CharacterRead:
    """Met a jour la bourse (po/pa/pc) d'un personnage."""
    c = session.get(Character, character_id)
    if not c or c.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Personnage introuvable.")
    c.gold = max(0, data.gold)
    c.silver = max(0, data.silver)
    c.copper = max(0, data.copper)
    session.add(c); session.commit(); session.refresh(c)
    cname = None
    if c.campaign_id:
        camp_row = session.get(Campaign, c.campaign_id)
        cname = camp_row.name if camp_row else None
    return _to_read(c, cname)


@router.delete("/{character_id}", status_code=204)
def delete_character(
    character_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> None:
    c = session.get(Character, character_id)
    if not c or c.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Personnage introuvable.")
    session.delete(c)
    session.commit()
