"""Route /api/solo -- aventures en solo (1 joueur + narrateur LLM).

Une aventure solo est une Campaign avec status="solo" : le joueur incarne SON
personnage, le LLM joue le reste. L'ossature (actes, antagoniste, objectif) est
generee par des regles (services.solo) puis stockee dans Campaign.solo_outline ;
le jeu se deroule via le chat /api/gamemaster/chat (qui detecte le mode solo).
"""
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from backend.db import get_session
from backend.deps import get_current_user
from backend.models import (
    Campaign, CampaignElement, CampaignMember, Character, Location, Npc, Quest, User,
)
from backend.schemas import (
    CampaignSummary, SoloAdventure, SoloAdventuresResponse, SoloStartRequest,
)
from backend.services import campaign as camp
from backend.services.mcp import get_mcp_gamemaster
from backend.services.solo import build_outline

router = APIRouter()


def _unique_name(session: Session, user_id: int, base: str) -> str:
    """Evite deux aventures du meme nom pour un meme joueur (confort d'affichage)."""
    member_ids = set(session.exec(
        select(CampaignMember.campaign_id).where(CampaignMember.user_id == user_id)).all())
    existing = {c.name for c in session.exec(select(Campaign)).all()
                if c.id in member_ids or c.dm_id == user_id}
    name = base
    i = 2
    while name in existing:
        name = f"{base} ({i})"
        i += 1
    return name


@router.post("/start", response_model=CampaignSummary, status_code=201)
async def start_solo(data: SoloStartRequest, user: User = Depends(get_current_user),
                     session: Session = Depends(get_session)) -> CampaignSummary:
    """Genere et lance une aventure solo autour d'un personnage existant."""
    character = session.get(Character, data.character_id)
    if not character or character.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Personnage introuvable.")

    outline = build_outline(data.genre, data.length, data.tone, data.pitch, character.name)

    # 1) Campagne solo
    name = _unique_name(session, user.id, outline["title"])
    campaign = Campaign(name=name, description=outline["synopsis"], dm_id=user.id,
                        status="solo", solo_outline=json.dumps(outline, ensure_ascii=False))
    session.add(campaign); session.commit(); session.refresh(campaign)
    session.add(CampaignMember(campaign_id=campaign.id, user_id=user.id, role="dm"))
    session.commit()

    # 2) Lieu de depart (actif)
    loc_d = outline["start_location"]
    loc = Location(owner_id=user.id, campaign_id=campaign.id, name=loc_d["name"],
                   description=loc_d["description"], region=loc_d.get("region"))
    session.add(loc); session.commit(); session.refresh(loc)
    session.add(CampaignElement(owner_id=user.id, campaign_id=campaign.id,
                                element_type="location", element_id=loc.id))

    # 3) Antagoniste (PNJ adversaire, actif)
    ant = outline["antagonist"]
    npc = Npc(owner_id=user.id, campaign_id=campaign.id, name=ant["name"], occupation=ant["occupation"],
              attitude="Hostile", is_adversary=True, armor_class=ant["armor_class"],
              hit_points=ant["hit_points"], challenge_rating=ant["challenge_rating"])
    session.add(npc); session.commit(); session.refresh(npc)
    session.add(CampaignElement(owner_id=user.id, campaign_id=campaign.id,
                                element_type="npc", element_id=npc.id))

    # 4) Quete principale (objectif)
    quest = Quest(campaign_id=campaign.id, owner_id=user.id, title="Quete principale",
                  kind="Principale", status="active", objective=outline["objective"],
                  description=outline["synopsis"])
    session.add(quest)

    # 5) Rattache le personnage joueur (direct + actif)
    character.campaign_id = campaign.id
    session.add(character)
    session.add(CampaignElement(owner_id=user.id, campaign_id=campaign.id,
                                element_type="character", element_id=character.id))
    session.commit()

    # 6) MCP best-effort : pousse la campagne + le perso dans le serveur de jeu
    try:
        manager = get_mcp_gamemaster()
        await camp.mcp_create_campaign(manager, name, outline["synopsis"], user.username)
        await camp.mcp_create_character(manager, {
            "name": character.name, "race": character.race, "character_class": character.character_class,
            "class_level": character.class_level, "player_name": user.username,
            "background": character.background, "alignment": character.alignment,
        })
    except Exception:  # noqa: BLE001 -- le mode solo fonctionne sans le serveur MCP
        pass

    return CampaignSummary(id=campaign.id, name=campaign.name, description=campaign.description,
                           role="dm", status=campaign.status)


@router.get("/adventures", response_model=SoloAdventuresResponse)
def list_adventures(user: User = Depends(get_current_user),
                    session: Session = Depends(get_session)) -> SoloAdventuresResponse:
    """Liste les aventures solo de l'utilisateur (les plus recentes d'abord)."""
    camps = session.exec(
        select(Campaign).where(Campaign.dm_id == user.id, Campaign.status == "solo")
        .order_by(Campaign.created_at.desc())
    ).all()
    out = []
    for c in camps:
        genre, acts, hero = None, 0, None
        try:
            o = json.loads(c.solo_outline) if c.solo_outline else {}
            genre = o.get("genre")
            acts = len(o.get("acts", []))
        except (ValueError, TypeError):
            pass
        pj = session.exec(select(Character).where(Character.campaign_id == c.id)).first()
        hero = pj.name if pj else None
        out.append(SoloAdventure(id=c.id, name=c.name, description=c.description, genre=genre,
                                 character_name=hero, acts=acts,
                                 created_at=c.created_at.isoformat() if c.created_at else None))
    return SoloAdventuresResponse(adventures=out)
