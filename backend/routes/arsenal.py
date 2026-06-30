"""Route /api/arsenal -- catalogue d'objets D&D (SRD) + attribution + monstres."""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from backend.db import get_session
from backend.deps import get_current_user
from backend.models import Character, Item, ItemAssignment, Npc, Quest, User
from backend.schemas import (
    ActionResult,
    ArsenalAssign,
    ArsenalCatalog,
    ArsenalItem,
    ArsenalItemDetail,
    MonstersResponse,
    MonsterStats,
)
from backend.services import equipment

router = APIRouter()


@router.get("/catalog", response_model=ArsenalCatalog)
async def catalog() -> ArsenalCatalog:
    return ArsenalCatalog(**await equipment.get_catalog())


@router.get("/item/{kind}/{index}", response_model=ArsenalItemDetail)
async def item(kind: str, index: str) -> ArsenalItemDetail:
    return ArsenalItemDetail(**await equipment.get_item_detail(kind, index))


@router.get("/monsters", response_model=MonstersResponse)
async def monsters() -> MonstersResponse:
    return MonstersResponse(monsters=[ArsenalItem(**m) for m in await equipment.get_monsters()])


@router.get("/monster/{index}", response_model=MonsterStats)
async def monster(index: str) -> MonsterStats:
    return MonsterStats(**await equipment.get_monster_detail(index))


def _check_target(session: Session, user: User, target_type: str, target_id: int) -> None:
    """Verifie que la cible appartient bien a l'utilisateur."""
    model = {"character": Character, "npc": Npc, "quest": Quest}.get(target_type)
    if model is None:
        raise HTTPException(status_code=400, detail="Cible invalide.")
    target = session.get(model, target_id)
    if not target or target.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Cible introuvable.")


@router.post("/assign", response_model=ActionResult)
def assign(data: ArsenalAssign, user: User = Depends(get_current_user),
           session: Session = Depends(get_session)) -> ActionResult:
    """Attribue un objet a un PNJ / quete / personnage.

    L'objet est reutilise s'il existe deja (meme nom) pour ne pas dupliquer la
    bibliotheque : plusieurs cibles peuvent ainsi partager le meme objet.
    """
    _check_target(session, user, data.target_type, data.target_id)
    item = session.exec(
        select(Item).where(Item.owner_id == user.id, Item.name == data.name)
    ).first()
    if not item:
        item = Item(owner_id=user.id, name=data.name, item_type=data.item_type or "misc",
                    value=data.value or None, description=(data.description or None))
        session.add(item); session.commit(); session.refresh(item)
    session.add(ItemAssignment(item_id=item.id, owner_id=user.id,
                               target_type=data.target_type, target_id=data.target_id, quantity=1))
    session.commit()
    return ActionResult(ok=True, message=f"{data.name} attribue.")
