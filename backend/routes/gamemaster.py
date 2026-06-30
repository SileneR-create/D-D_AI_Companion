"""Route /api/gamemaster -- chat MD, etat, campagnes, PNJ, lieux, quetes, recrutement."""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from backend.config import AVAILABLE_MODELS
from backend.db import engine, get_session
from backend.deps import get_current_user
from datetime import datetime

from backend.models import (
    Campaign, CampaignElement, CampaignMember, Character, ItemAssignment, Location, Message, Npc, Quest, User,
)
from backend.prompts import GAMEMASTER_PROMPT
from backend.schemas import (
    ActionResult,
    ArchivedQuest,
    CampaignArchive,
    CampaignCreate,
    CampaignsResponse,
    CampaignState,
    CampaignSummary,
    ChatRequest,
    LocationCreate,
    LocationRead,
    LocationsResponse,
    MessageRead,
    MessagesResponse,
    ModelsResponse,
    NpcCreate,
    NpcRead,
    NpcsResponse,
    QuestCreate,
    QuestRead,
    QuestsResponse,
    QuestUpdate,
    RecruitRequest,
)
from backend.services import campaign as camp
from backend.services.chat import stream_chat
from backend.services.mcp import get_mcp_gamemaster

router = APIRouter()

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}

_ATTITUDE_EN = {"Amical": "friendly", "Neutre": "neutral", "Hostile": "hostile", "Inconnu": "unknown"}
_ATTITUDE_FR = {v: k for k, v in _ATTITUDE_EN.items()}


def _mirror_chat_tool(owner_id: int, campaign_id: int):
    """Fabrique un callback qui reflete dans la base les PNJ/lieux que le MD (LLM)
    cree pendant la partie via les outils MCP -- sinon ils restent dans le stockage
    interne du serveur MCP et n'apparaissent jamais dans les Archives.

    Idempotent : un element du meme nom (meme proprietaire/campagne) n'est pas duplique.
    """
    def mirror(tool_name: str, args: dict) -> None:
        if not isinstance(args, dict):
            return
        short = tool_name.split(".")[-1].split("-")[-1]
        name = (args.get("name") or "").strip()
        if not name:
            return
        with Session(engine) as s:
            if short == "create_npc":
                exists = s.exec(select(Npc).where(
                    Npc.owner_id == owner_id, Npc.campaign_id == campaign_id, Npc.name == name)).first()
                if exists:
                    return
                npc = Npc(owner_id=owner_id, campaign_id=campaign_id, name=name,
                          appearance=args.get("description"), race=args.get("race"),
                          occupation=args.get("occupation"),
                          attitude=_ATTITUDE_FR.get(args.get("attitude"), None),
                          notes=args.get("notes") or None)
                s.add(npc); s.commit(); s.refresh(npc)
                s.add(CampaignElement(owner_id=owner_id, campaign_id=campaign_id,
                                      element_type="npc", element_id=npc.id))
                s.commit()
            elif short == "create_location":
                exists = s.exec(select(Location).where(
                    Location.owner_id == owner_id, Location.campaign_id == campaign_id,
                    Location.name == name)).first()
                if exists:
                    return
                loc = Location(owner_id=owner_id, campaign_id=campaign_id, name=name,
                               description=args.get("description"), region=args.get("location_type"))
                s.add(loc); s.commit(); s.refresh(loc)
                s.add(CampaignElement(owner_id=owner_id, campaign_id=campaign_id,
                                      element_type="location", element_id=loc.id))
                s.commit()
    return mirror


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=AVAILABLE_MODELS)


@router.get("/state", response_model=CampaignState)
async def state() -> CampaignState:
    return await camp.get_campaign_state(get_mcp_gamemaster())


def _membership(session: Session, campaign_id: int, user_id: int) -> CampaignMember | None:
    return session.exec(
        select(CampaignMember).where(
            CampaignMember.campaign_id == campaign_id, CampaignMember.user_id == user_id,
        )
    ).first()


def _require_member(session, campaign_id, user, dm=False) -> tuple[Campaign, CampaignMember]:
    campaign = session.get(Campaign, campaign_id)
    member = _membership(session, campaign_id, user.id)
    if not campaign or not member:
        raise HTTPException(status_code=404, detail="Campagne introuvable ou acces refuse.")
    if dm and member.role != "dm":
        raise HTTPException(status_code=403, detail="Action reservee au Maitre du Jeu.")
    return campaign, member


# --- Campagnes --------------------------------------------------------------
@router.post("/campaigns", response_model=CampaignSummary, status_code=201)
async def create_campaign(data: CampaignCreate, user: User = Depends(get_current_user),
                          session: Session = Depends(get_session)) -> CampaignSummary:
    # Unicite du nom PAR UTILISATEUR (deux utilisateurs differents peuvent avoir
    # une campagne du meme nom : tant qu'ils ne jouent pas ensemble, aucun souci).
    member_ids = set(session.exec(
        select(CampaignMember.campaign_id).where(CampaignMember.user_id == user.id)).all())
    same_name = session.exec(select(Campaign).where(Campaign.name == data.name)).all()
    if any(c.dm_id == user.id or c.id in member_ids for c in same_name):
        raise HTTPException(status_code=400, detail="Vous avez deja une campagne portant ce nom.")
    campaign = Campaign(name=data.name, description=data.description, dm_id=user.id)
    session.add(campaign); session.commit(); session.refresh(campaign)
    session.add(CampaignMember(campaign_id=campaign.id, user_id=user.id, role="dm")); session.commit()
    await camp.mcp_create_campaign(get_mcp_gamemaster(), data.name, data.description, user.username)
    return CampaignSummary(id=campaign.id, name=campaign.name, description=campaign.description, role="dm", status=campaign.status)


@router.get("/campaigns", response_model=CampaignsResponse)
def list_campaigns(user: User = Depends(get_current_user),
                   session: Session = Depends(get_session)) -> CampaignsResponse:
    """Toutes les campagnes de l'utilisateur : celles dont il est membre ET celles
    qu'il a creees (dm_id), meme si la ligne d'adhesion manquait (auto-reparee)."""
    by_id: dict[int, tuple[Campaign, str]] = {}
    member_rows = session.exec(
        select(Campaign, CampaignMember).join(CampaignMember, CampaignMember.campaign_id == Campaign.id)
        .where(CampaignMember.user_id == user.id)
    ).all()
    for c, m in member_rows:
        by_id[c.id] = (c, m.role)

    owned = session.exec(select(Campaign).where(Campaign.dm_id == user.id)).all()
    healed = False
    for c in owned:
        if c.id not in by_id:
            # Campagne possedee sans adhesion -> on repare pour la rendre activable.
            session.add(CampaignMember(campaign_id=c.id, user_id=user.id, role="dm"))
            healed = True
            by_id[c.id] = (c, "dm")
    if healed:
        session.commit()

    camps = sorted(by_id.values(), key=lambda t: (t[0].created_at or datetime.min), reverse=True)
    return CampaignsResponse(campaigns=[
        CampaignSummary(id=c.id, name=c.name, description=c.description, role=role, status=c.status)
        for c, role in camps
    ])


@router.post("/campaigns/{campaign_id}/activate", response_model=CampaignSummary)
async def activate_campaign(campaign_id: int, user: User = Depends(get_current_user),
                            session: Session = Depends(get_session)) -> CampaignSummary:
    campaign, member = _require_member(session, campaign_id, user)
    await camp.mcp_load_campaign(get_mcp_gamemaster(), campaign.name)
    return CampaignSummary(id=campaign.id, name=campaign.name, description=campaign.description, role=member.role, status=campaign.status)


def _archive(session: Session, campaign: Campaign) -> CampaignArchive:
    quests = session.exec(select(Quest).where(Quest.campaign_id == campaign.id)).all()
    won, failed = [], []
    for q in quests:
        aq = ArchivedQuest(title=q.title, objective=q.objective or "", reward=q.reward,
                           outcome="won" if (q.status == "completed" or q.objective_done) else "failed")
        (won if aq.outcome == "won" else failed).append(aq)
    return CampaignArchive(id=campaign.id, name=campaign.name, status=campaign.status, won=won, failed=failed)


@router.post("/campaigns/{campaign_id}/end", response_model=CampaignArchive)
def end_campaign(campaign_id: int, user: User = Depends(get_current_user),
                 session: Session = Depends(get_session)) -> CampaignArchive:
    """Met fin a la campagne (MD). Tout est conserve ; on renvoie l'archive des quetes."""
    campaign, _ = _require_member(session, campaign_id, user, dm=True)
    campaign.status = "ended"
    campaign.ended_at = datetime.utcnow()
    session.add(campaign); session.commit(); session.refresh(campaign)
    return _archive(session, campaign)


@router.get("/campaigns/{campaign_id}/archive", response_model=CampaignArchive)
def campaign_archive(campaign_id: int, user: User = Depends(get_current_user),
                     session: Session = Depends(get_session)) -> CampaignArchive:
    """Archive des quetes reussies/echouees d'une campagne."""
    campaign, _ = _require_member(session, campaign_id, user)
    return _archive(session, campaign)


@router.delete("/campaigns/{campaign_id}", status_code=204)
def delete_campaign(campaign_id: int, user: User = Depends(get_current_user),
                    session: Session = Depends(get_session)) -> None:
    """Supprime definitivement une campagne (MD). Detache les persos, purge quetes/messages/membres/elements."""
    campaign, _ = _require_member(session, campaign_id, user, dm=True)
    # Detache les personnages (on garde les fiches dans Mon antre).
    for ch in session.exec(select(Character).where(Character.campaign_id == campaign_id)).all():
        ch.campaign_id = None
        session.add(ch)
    # Purge les quetes (et leurs attributions d'objets).
    for q in session.exec(select(Quest).where(Quest.campaign_id == campaign_id)).all():
        for a in session.exec(select(ItemAssignment).where(
                ItemAssignment.target_type == "quest", ItemAssignment.target_id == q.id)).all():
            session.delete(a)
        session.delete(q)
    for m in session.exec(select(Message).where(Message.campaign_id == campaign_id)).all():
        session.delete(m)
    for e in session.exec(select(CampaignElement).where(CampaignElement.campaign_id == campaign_id)).all():
        session.delete(e)
    for mem in session.exec(select(CampaignMember).where(CampaignMember.campaign_id == campaign_id)).all():
        session.delete(mem)
    session.delete(campaign)
    session.commit()


# --- PNJ --------------------------------------------------------------------
def _npc_read(n: Npc) -> NpcRead:
    return NpcRead(id=n.id, name=n.name, appearance=n.appearance, race=n.race,
                   occupation=n.occupation, attitude=n.attitude, notes=n.notes,
                   is_adversary=bool(n.is_adversary), armor_class=n.armor_class,
                   hit_points=n.hit_points, challenge_rating=n.challenge_rating,
                   monster_ref=n.monster_ref)


@router.post("/campaigns/{campaign_id}/npcs", response_model=NpcRead, status_code=201)
async def create_npc(campaign_id: int, data: NpcCreate, user: User = Depends(get_current_user),
                     session: Session = Depends(get_session)) -> NpcRead:
    campaign, _ = _require_member(session, campaign_id, user, dm=True)
    npc = Npc(campaign_id=campaign_id, owner_id=user.id, name=data.name, appearance=data.appearance,
              race=data.race, occupation=data.occupation, attitude=data.attitude, notes=data.notes,
              is_adversary=data.is_adversary, armor_class=data.armor_class, hit_points=data.hit_points,
              challenge_rating=data.challenge_rating, monster_ref=data.monster_ref)
    session.add(npc); session.commit(); session.refresh(npc)
    # Actif dans la campagne ou il est cree (apparait dans la liste de la Table).
    session.add(CampaignElement(owner_id=user.id, campaign_id=campaign_id, element_type="npc", element_id=npc.id))
    session.commit()
    manager = get_mcp_gamemaster()
    await camp.mcp_load_campaign(manager, campaign.name)
    await camp.mcp_create_npc(manager, data.name, data.appearance, data.race, data.occupation,
                              _ATTITUDE_EN.get(data.attitude or "", None))
    return _npc_read(npc)


@router.get("/campaigns/{campaign_id}/npcs", response_model=NpcsResponse)
def list_npcs(campaign_id: int, user: User = Depends(get_current_user),
              session: Session = Depends(get_session)) -> NpcsResponse:
    _require_member(session, campaign_id, user)
    ids = session.exec(select(CampaignElement.element_id).where(
        CampaignElement.campaign_id == campaign_id, CampaignElement.element_type == "npc")).all()
    npcs = session.exec(select(Npc).where(Npc.id.in_(ids)).order_by(Npc.created_at.desc())).all() if ids else []
    return NpcsResponse(npcs=[_npc_read(n) for n in npcs])


@router.delete("/campaigns/{campaign_id}/npcs/{npc_id}", status_code=204)
def delete_npc(campaign_id: int, npc_id: int, user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> None:
    npc = session.get(Npc, npc_id)
    if not npc or npc.owner_id != user.id:
        raise HTTPException(status_code=404, detail="PNJ introuvable.")
    session.delete(npc); session.commit()


# --- Lieux ------------------------------------------------------------------
@router.post("/campaigns/{campaign_id}/locations", response_model=LocationRead, status_code=201)
async def create_location(campaign_id: int, data: LocationCreate, user: User = Depends(get_current_user),
                          session: Session = Depends(get_session)) -> LocationRead:
    campaign, _ = _require_member(session, campaign_id, user, dm=True)
    loc = Location(campaign_id=campaign_id, owner_id=user.id, name=data.name,
                   description=data.description, region=data.region)
    session.add(loc); session.commit(); session.refresh(loc)
    session.add(CampaignElement(owner_id=user.id, campaign_id=campaign_id, element_type="location", element_id=loc.id))
    session.commit()
    manager = get_mcp_gamemaster()
    await camp.mcp_load_campaign(manager, campaign.name)
    await camp.mcp_create_location(manager, data.name, data.description)
    return LocationRead(id=loc.id, name=loc.name, description=loc.description, region=loc.region)


@router.get("/campaigns/{campaign_id}/locations", response_model=LocationsResponse)
def list_locations(campaign_id: int, user: User = Depends(get_current_user),
                   session: Session = Depends(get_session)) -> LocationsResponse:
    _require_member(session, campaign_id, user)
    ids = session.exec(select(CampaignElement.element_id).where(
        CampaignElement.campaign_id == campaign_id, CampaignElement.element_type == "location")).all()
    locs = session.exec(select(Location).where(Location.id.in_(ids)).order_by(Location.created_at.desc())).all() if ids else []
    return LocationsResponse(locations=[
        LocationRead(id=l.id, name=l.name, description=l.description, region=l.region) for l in locs
    ])


# --- Quetes -----------------------------------------------------------------
def _quest_read(q: Quest) -> QuestRead:
    try:
        enemies = json.loads(q.enemies) if q.enemies else []
    except (ValueError, TypeError):
        enemies = []
    return QuestRead(id=q.id, title=q.title, kind=q.kind, status=q.status, description=q.description,
                     objective=q.objective, objective_done=q.objective_done, enemies=enemies,
                     giver=q.giver, reward=q.reward)


@router.post("/campaigns/{campaign_id}/quests", response_model=QuestRead, status_code=201)
async def create_quest(campaign_id: int, data: QuestCreate, user: User = Depends(get_current_user),
                       session: Session = Depends(get_session)) -> QuestRead:
    if not data.objective.strip():
        raise HTTPException(status_code=400, detail="Un objectif est obligatoire.")
    campaign, _ = _require_member(session, campaign_id, user, dm=True)
    quest = Quest(campaign_id=campaign_id, owner_id=user.id, title=data.title, kind=data.kind,
                  status=data.status, description=data.description, objective=data.objective,
                  enemies=json.dumps(data.enemies) if data.enemies else None,
                  giver=data.giver, reward=data.reward)
    session.add(quest); session.commit(); session.refresh(quest)
    manager = get_mcp_gamemaster()
    await camp.mcp_load_campaign(manager, campaign.name)
    notes = f"Objectif: {data.objective}"
    if data.enemies:
        notes += f" | Ennemis: {', '.join(data.enemies)}"
    await camp.mcp_create_quest(manager, data.title, (data.description or "") + " — " + notes, data.giver, data.reward)
    return _quest_read(quest)


@router.get("/campaigns/{campaign_id}/quests", response_model=QuestsResponse)
def list_campaign_quests(campaign_id: int, user: User = Depends(get_current_user),
                         session: Session = Depends(get_session)) -> QuestsResponse:
    _require_member(session, campaign_id, user)
    quests = session.exec(select(Quest).where(Quest.campaign_id == campaign_id).order_by(Quest.created_at.desc())).all()
    return QuestsResponse(quests=[_quest_read(q) for q in quests])


@router.patch("/campaigns/{campaign_id}/quests/{quest_id}", response_model=QuestRead)
def update_quest(campaign_id: int, quest_id: int, data: QuestUpdate,
                 user: User = Depends(get_current_user), session: Session = Depends(get_session)) -> QuestRead:
    """Met a jour le statut et/ou valide l'objectif (MD)."""
    _require_member(session, campaign_id, user, dm=True)
    quest = session.get(Quest, quest_id)
    if not quest or quest.campaign_id != campaign_id:
        raise HTTPException(status_code=404, detail="Quete introuvable.")
    if data.status is not None:
        quest.status = data.status
    if data.objective_done is not None:
        quest.objective_done = data.objective_done
    session.add(quest); session.commit(); session.refresh(quest)
    return _quest_read(quest)


@router.delete("/campaigns/{campaign_id}/quests/{quest_id}", status_code=204)
def delete_quest(campaign_id: int, quest_id: int, user: User = Depends(get_current_user),
                 session: Session = Depends(get_session)) -> None:
    _require_member(session, campaign_id, user, dm=True)
    quest = session.get(Quest, quest_id)
    if not quest or quest.campaign_id != campaign_id:
        raise HTTPException(status_code=404, detail="Quete introuvable.")
    session.delete(quest); session.commit()


# --- Recrutement ------------------------------------------------------------
@router.post("/campaigns/{campaign_id}/recruit", response_model=ActionResult)
async def recruit_character(campaign_id: int, data: RecruitRequest, user: User = Depends(get_current_user),
                            session: Session = Depends(get_session)) -> ActionResult:
    campaign, _ = _require_member(session, campaign_id, user)
    character = session.get(Character, data.character_id)
    if not character or character.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Personnage introuvable.")
    if character.campaign_id:
        raise HTTPException(status_code=400, detail="Ce personnage appartient deja a une campagne.")
    character.campaign_id = campaign_id
    session.add(character); session.commit()
    manager = get_mcp_gamemaster()
    await camp.mcp_load_campaign(manager, campaign.name)
    await camp.mcp_create_character(manager, {
        "name": character.name, "race": character.race, "character_class": character.character_class,
        "class_level": character.class_level, "player_name": user.username,
        "background": character.background, "alignment": character.alignment,
        "strength": character.strength, "dexterity": character.dexterity,
        "constitution": character.constitution, "intelligence": character.intelligence,
        "wisdom": character.wisdom, "charisma": character.charisma,
    })
    return ActionResult(ok=True, message=f"{character.name} rejoint la campagne.")


# --- Historique de conversation ---------------------------------------------
def _save_message(campaign_id: int, role: str, content: str, author: str | None) -> None:
    with Session(engine) as s:
        s.add(Message(campaign_id=campaign_id, role=role, content=content, author=author))
        s.commit()


@router.get("/campaigns/{campaign_id}/messages", response_model=MessagesResponse)
def list_messages(campaign_id: int, user: User = Depends(get_current_user),
                  session: Session = Depends(get_session)) -> MessagesResponse:
    """Historique de conversation de la campagne."""
    _require_member(session, campaign_id, user)
    msgs = session.exec(select(Message).where(Message.campaign_id == campaign_id).order_by(Message.created_at)).all()
    return MessagesResponse(messages=[MessageRead(role=m.role, content=m.content, author=m.author) for m in msgs])


def _campaign_context(session: Session, campaign_id: int | None, username: str) -> str:
    """Bloc de contexte REEL injecte dans le prompt du MD (evite les hallucinations).

    Donne au LLM : le nom du joueur, le nom/synopsis de la campagne, les PJ qui
    participent, les PNJ actifs, les lieux et les quetes en cours.
    """
    base = (f"\n\nLE JOUEUR a la table s'appelle : {username}. "
            "Utilise ce nom, n'ecris JAMAIS le marqueur [nom_utilisateur].")
    if not campaign_id:
        return base
    campaign = session.get(Campaign, campaign_id)
    if not campaign:
        return base

    # PJ : rattaches directement OU actifs via CampaignElement.
    elem_char_ids = set(session.exec(select(CampaignElement.element_id).where(
        CampaignElement.campaign_id == campaign_id, CampaignElement.element_type == "character")).all())
    pjs, seen = [], set()
    for c in session.exec(select(Character).where(Character.campaign_id == campaign_id)).all():
        if c.id not in seen:
            seen.add(c.id); pjs.append(c)
    if elem_char_ids:
        for c in session.exec(select(Character).where(Character.id.in_(elem_char_ids))).all():
            if c.id not in seen:
                seen.add(c.id); pjs.append(c)

    npc_ids = session.exec(select(CampaignElement.element_id).where(
        CampaignElement.campaign_id == campaign_id, CampaignElement.element_type == "npc")).all()
    npcs = session.exec(select(Npc).where(Npc.id.in_(npc_ids))).all() if npc_ids else []
    loc_ids = session.exec(select(CampaignElement.element_id).where(
        CampaignElement.campaign_id == campaign_id, CampaignElement.element_type == "location")).all()
    locs = session.exec(select(Location).where(Location.id.in_(loc_ids))).all() if loc_ids else []
    quests = session.exec(select(Quest).where(
        Quest.campaign_id == campaign_id, Quest.status == "active")).all()

    def pj_line(c):
        return f"{c.name} ({c.race} {c.character_class} niv {c.class_level})"

    def npc_line(n):
        if n.is_adversary:
            stats = " ".join(filter(None, [f"CA {n.armor_class}" if n.armor_class else "",
                                           f"PV {n.hit_points}" if n.hit_points else "",
                                           f"FP {n.challenge_rating}" if n.challenge_rating else ""]))
            return f"{n.name} (ADVERSAIRE{(' ' + stats) if stats else ''})"
        return f"{n.name}{(' - ' + n.occupation) if n.occupation else ''}"

    lines = [
        "\n\nCONTEXTE REEL DE LA CAMPAGNE EN COURS (faits etablis, ne les invente PAS) :",
        f"- Nom de la campagne : {campaign.name}",
        f"- Synopsis : {campaign.description or 'non precise'}",
        f"- Joueur a la table : {username}",
        "- Personnages joueurs (TOUS participent au combat par defaut) : "
        + (", ".join(pj_line(c) for c in pjs) if pjs else "aucun pour l'instant"),
        "- PNJ actifs : " + (", ".join(npc_line(n) for n in npcs) if npcs else "aucun"),
        "- Lieux : " + (", ".join(l.name for l in locs) if locs else "aucun"),
        "- Quetes en cours : " + (", ".join(q.title for q in quests) if quests else "aucune"),
        "\nREGLES STRICTES :",
        "- N'invente JAMAIS le nom de la campagne ni de nouveaux joueurs : utilise ceux ci-dessus.",
        "- Pour un combat, inclus d'office TOUS les personnages joueurs listes comme participants ;",
        "  ne demande que les adversaires s'ils ne sont pas connus.",
        "- Ne te represente pas a chaque message : poursuis directement l'action demandee.",
    ]
    return base + "\n".join(lines)


# --- Chat -------------------------------------------------------------------
@router.post("/chat")
async def chat(req: ChatRequest, user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> StreamingResponse:
    # Persiste l'historique si la requete est rattachee a une campagne du membre.
    persist = None
    if req.campaign_id and _membership(session, req.campaign_id, user.id):
        persist = req.campaign_id
        if req.message.strip() and not req.resend:
            _save_message(persist, "user", req.message, user.username)

    system_prompt = GAMEMASTER_PROMPT + _campaign_context(session, req.campaign_id, user.username)
    # Reflet en base : les PNJ/lieux crees par le MD pendant la partie apparaissent
    # dans les Archives (sinon ils restent dans le stockage interne du serveur MCP).
    mirror = _mirror_chat_tool(user.id, persist) if persist else None
    gen = stream_chat(req, system_prompt, get_mcp_gamemaster(), on_tool=mirror)

    async def wrapped():
        reply = ""
        async for chunk in gen:
            line = chunk.strip()
            if line.startswith("data:"):
                try:
                    evt = json.loads(line[5:].strip())
                    t = evt.get("type")
                    if t == "token":
                        reply += evt.get("content", "")
                    elif t == "clear":
                        reply = ""
                    elif t == "done" and persist and reply.strip():
                        _save_message(persist, "assistant", reply, None)
                except Exception:  # noqa: BLE001
                    pass
            yield chunk

    return StreamingResponse(wrapped(), media_type="text/event-stream", headers=SSE_HEADERS)
