"""Route /api/forge -- bibliotheque d'objets (forge) + assignation a PJ/PNJ/quete."""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from backend.db import get_session
from backend.deps import get_current_user
from backend.models import Campaign, CampaignMember, Character, Item, ItemAssignment, Npc, Quest, User
from backend.schemas import (
    ActionResult,
    AssignmentRead,
    AssignmentsResponse,
    AssignRequest,
    ItemCreate,
    ItemRead,
    ItemsResponse,
)
from backend.services import campaign as camp
from backend.services.mcp import get_mcp_gamemaster

router = APIRouter()

# Types d'objets cote MCP (add_item_to_character) : weapon|armor|consumable|misc.
_MCP_ITEM_TYPE = {"weapon": "weapon", "armor": "armor", "consumable": "consumable", "magic": "misc", "misc": "misc"}


def _item_read(i: Item) -> ItemRead:
    return ItemRead(id=i.id, name=i.name, item_type=i.item_type, rarity=i.rarity,
                    value=i.value, gp=i.gp or 0, sp=i.sp or 0, cp=i.cp or 0,
                    description=i.description)


# --- Bibliotheque -----------------------------------------------------------
@router.post("/items", response_model=ItemRead, status_code=201)
def create_item(data: ItemCreate, user: User = Depends(get_current_user),
                session: Session = Depends(get_session)) -> ItemRead:
    item = Item(owner_id=user.id, name=data.name, item_type=data.item_type, rarity=data.rarity,
                value=data.value, gp=data.gp, sp=data.sp, cp=data.cp, description=data.description)
    session.add(item); session.commit(); session.refresh(item)
    return _item_read(item)


@router.get("/items", response_model=ItemsResponse)
def list_items(user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> ItemsResponse:
    items = session.exec(select(Item).where(Item.owner_id == user.id).order_by(Item.created_at.desc())).all()
    return ItemsResponse(items=[_item_read(i) for i in items])


@router.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int, user: User = Depends(get_current_user),
                session: Session = Depends(get_session)) -> None:
    item = session.get(Item, item_id)
    if not item or item.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Objet introuvable.")
    # retire aussi les assignations
    for a in session.exec(select(ItemAssignment).where(ItemAssignment.item_id == item_id)).all():
        session.delete(a)
    session.delete(item); session.commit()


# --- Assignation ------------------------------------------------------------
def _check_target(session, user, target_type, target_id):
    """Verifie l'acces a la cible et renvoie (campaign, character|None)."""
    if target_type == "character":
        ch = session.get(Character, target_id)
        if not ch or ch.owner_id != user.id:
            raise HTTPException(status_code=404, detail="Personnage introuvable.")
        campaign = session.get(Campaign, ch.campaign_id) if ch.campaign_id else None
        return campaign, ch
    if target_type == "npc":
        npc = session.get(Npc, target_id)
        if not npc or npc.owner_id != user.id:
            raise HTTPException(status_code=404, detail="PNJ introuvable.")
        return (session.get(Campaign, npc.campaign_id) if npc.campaign_id else None), None
    quest = session.get(Quest, target_id)
    if not quest or quest.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Quete introuvable.")
    return (session.get(Campaign, quest.campaign_id) if quest.campaign_id else None), None


def _require_member(session, campaign_id, user):
    m = session.exec(select(CampaignMember).where(
        CampaignMember.campaign_id == campaign_id, CampaignMember.user_id == user.id)).first()
    if not m:
        raise HTTPException(status_code=403, detail="Acces refuse a cette campagne.")
    return m


@router.post("/assign", response_model=ActionResult)
async def assign_item(data: AssignRequest, user: User = Depends(get_current_user),
                      session: Session = Depends(get_session)) -> ActionResult:
    item = session.get(Item, data.item_id)
    if not item or item.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Objet introuvable.")
    campaign, character = _check_target(session, user, data.target_type, data.target_id)

    session.add(ItemAssignment(item_id=item.id, owner_id=user.id, target_type=data.target_type,
                               target_id=data.target_id, quantity=max(1, data.quantity)))
    session.commit()

    # Sync MCP : seulement pour un personnage rattache a une campagne.
    if data.target_type == "character" and character and campaign:
        manager = get_mcp_gamemaster()
        await camp.mcp_load_campaign(manager, campaign.name)
        await camp.mcp_add_item_to_character(
            manager, character.name, item.name, item.description,
            _MCP_ITEM_TYPE.get(item.item_type, "misc"), item.value, data.quantity,
        )
    return ActionResult(ok=True, message=f"{item.name} attribue.")


@router.get("/assignments/{target_type}/{target_id}", response_model=AssignmentsResponse)
def list_assignments(target_type: str, target_id: int, user: User = Depends(get_current_user),
                     session: Session = Depends(get_session)) -> AssignmentsResponse:
    rows = session.exec(select(ItemAssignment).where(
        ItemAssignment.target_type == target_type, ItemAssignment.target_id == target_id,
        ItemAssignment.owner_id == user.id)).all()
    out = []
    for a in rows:
        item = session.get(Item, a.item_id)
        if item:
            out.append(AssignmentRead(id=a.id, item=_item_read(item), quantity=a.quantity))
    return AssignmentsResponse(assignments=out)


@router.delete("/assignments/{assignment_id}", status_code=204)
def delete_assignment(assignment_id: int, user: User = Depends(get_current_user),
                      session: Session = Depends(get_session)) -> None:
    a = session.get(ItemAssignment, assignment_id)
    if not a or a.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Attribution introuvable.")
    session.delete(a); session.commit()
