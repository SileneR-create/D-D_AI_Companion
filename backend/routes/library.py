"""Route /api/library -- bibliotheque transversale (PNJ, lieux, objets) + activation par campagne."""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from backend.db import get_session
from backend.deps import get_current_user
from backend.models import Campaign, CampaignElement, CampaignMember, Character, Item, Location, Npc, User
from backend.schemas import (
    ActionResult,
    ActivateRequest,
    CampaignSummary,
    ItemRead,
    LibraryElement,
    LibraryResponse,
    LocationCreate,
    NpcCreate,
)
from backend.services import campaign as camp
from backend.services.mcp import get_mcp_gamemaster

router = APIRouter()


def _active_map(session: Session, owner_id: int, element_type: str) -> dict[int, list[int]]:
    """element_id -> [campaign_id...] ou il est actif."""
    rows = session.exec(
        select(CampaignElement).where(
            CampaignElement.owner_id == owner_id, CampaignElement.element_type == element_type,
        )
    ).all()
    out: dict[int, list[int]] = {}
    for r in rows:
        out.setdefault(r.element_id, []).append(r.campaign_id)
    return out


@router.get("", response_model=LibraryResponse)
def get_library(user: User = Depends(get_current_user),
                session: Session = Depends(get_session)) -> LibraryResponse:
    """Tout ce que l'utilisateur a cree + ses campagnes (pour l'activation)."""
    rows = session.exec(
        select(Campaign, CampaignMember).join(CampaignMember, CampaignMember.campaign_id == Campaign.id)
        .where(CampaignMember.user_id == user.id)
    ).all()
    campaigns = [CampaignSummary(id=c.id, name=c.name, description=c.description, role=m.role, status=c.status) for c, m in rows]

    npc_active = _active_map(session, user.id, "npc")
    loc_active = _active_map(session, user.id, "location")

    npcs = session.exec(select(Npc).where(Npc.owner_id == user.id).order_by(Npc.created_at.desc())).all()
    locs = session.exec(select(Location).where(Location.owner_id == user.id).order_by(Location.created_at.desc())).all()
    items = session.exec(select(Item).where(Item.owner_id == user.id).order_by(Item.created_at.desc())).all()

    return LibraryResponse(
        campaigns=campaigns,
        npcs=[LibraryElement(id=n.id, name=n.name, summary=n.occupation or n.appearance,
                             active_in=npc_active.get(n.id, []), is_adversary=bool(n.is_adversary),
                             armor_class=n.armor_class, hit_points=n.hit_points,
                             challenge_rating=n.challenge_rating) for n in npcs],
        locations=[LibraryElement(id=l.id, name=l.name, summary=l.description, active_in=loc_active.get(l.id, [])) for l in locs],
        items=[ItemRead(id=i.id, name=i.name, item_type=i.item_type, rarity=i.rarity, value=i.value,
                        gp=i.gp or 0, sp=i.sp or 0, cp=i.cp or 0, description=i.description) for i in items],
    )


@router.post("/npc", response_model=LibraryElement, status_code=201)
def create_library_npc(data: NpcCreate, user: User = Depends(get_current_user),
                       session: Session = Depends(get_session)) -> LibraryElement:
    """Cree un PNJ dans la bibliotheque (sans campagne ; a activer ensuite)."""
    npc = Npc(owner_id=user.id, campaign_id=None, name=data.name, appearance=data.appearance,
              race=data.race, occupation=data.occupation, attitude=data.attitude, notes=data.notes,
              is_adversary=data.is_adversary, armor_class=data.armor_class, hit_points=data.hit_points,
              challenge_rating=data.challenge_rating, monster_ref=data.monster_ref)
    session.add(npc); session.commit(); session.refresh(npc)
    return LibraryElement(id=npc.id, name=npc.name, summary=npc.occupation or npc.appearance, active_in=[])


@router.post("/location", response_model=LibraryElement, status_code=201)
def create_library_location(data: LocationCreate, user: User = Depends(get_current_user),
                            session: Session = Depends(get_session)) -> LibraryElement:
    """Cree un lieu dans la bibliotheque (sans campagne ; a activer ensuite)."""
    loc = Location(owner_id=user.id, campaign_id=None, name=data.name,
                   description=data.description, region=data.region)
    session.add(loc); session.commit(); session.refresh(loc)
    return LibraryElement(id=loc.id, name=loc.name, summary=loc.description, active_in=[])


_ELEM_MODEL = {"npc": Npc, "location": Location, "character": Character}


def _check(session: Session, user: User, data: ActivateRequest):
    """Verifie que l'element et la campagne appartiennent bien a l'utilisateur.

    PNJ/lieux : reserves au MD. Personnages joueurs : tout membre de la campagne
    (le joueur fait entrer/sortir SON personnage).
    """
    model = _ELEM_MODEL.get(data.element_type)
    elem = session.get(model, data.element_id) if model else None
    if not elem or elem.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Element introuvable.")
    member = session.exec(select(CampaignMember).where(
        CampaignMember.campaign_id == data.campaign_id, CampaignMember.user_id == user.id)).first()
    campaign = session.get(Campaign, data.campaign_id)
    if not campaign or not member:
        raise HTTPException(status_code=403, detail="Campagne introuvable ou acces refuse.")
    if data.element_type in ("npc", "location") and member.role != "dm":
        raise HTTPException(status_code=403, detail="Action reservee au Maitre du Jeu.")
    return elem, campaign


@router.post("/activate", response_model=ActionResult)
async def activate(data: ActivateRequest, user: User = Depends(get_current_user),
                   session: Session = Depends(get_session)) -> ActionResult:
    elem, campaign = _check(session, user, data)
    exists = session.exec(select(CampaignElement).where(
        CampaignElement.campaign_id == data.campaign_id, CampaignElement.element_type == data.element_type,
        CampaignElement.element_id == data.element_id)).first()
    if not exists:
        session.add(CampaignElement(owner_id=user.id, campaign_id=data.campaign_id,
                                    element_type=data.element_type, element_id=data.element_id))
        session.commit()
        # Sync MCP best-effort : pousse l'element dans la campagne nouvellement activee.
        manager = get_mcp_gamemaster()
        await camp.mcp_load_campaign(manager, campaign.name)
        if data.element_type == "npc":
            await camp.mcp_create_npc(manager, elem.name, elem.appearance, elem.race, elem.occupation, None)
        elif data.element_type == "location":
            await camp.mcp_create_location(manager, elem.name, elem.description)
        else:  # character
            await camp.mcp_create_character(manager, {
                "name": elem.name, "race": elem.race, "character_class": elem.character_class,
                "class_level": elem.class_level, "player_name": user.username,
                "background": elem.background, "alignment": elem.alignment,
                "strength": elem.strength, "dexterity": elem.dexterity, "constitution": elem.constitution,
                "intelligence": elem.intelligence, "wisdom": elem.wisdom, "charisma": elem.charisma,
            })
    return ActionResult(ok=True, message=f"{elem.name} active dans {campaign.name}.")


@router.post("/deactivate", response_model=ActionResult)
def deactivate(data: ActivateRequest, user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> ActionResult:
    _check(session, user, data)
    row = session.exec(select(CampaignElement).where(
        CampaignElement.campaign_id == data.campaign_id, CampaignElement.element_type == data.element_type,
        CampaignElement.element_id == data.element_id)).first()
    if row:
        session.delete(row)
        session.commit()
    return ActionResult(ok=True, message="Element desactive.")
