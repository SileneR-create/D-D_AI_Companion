"""Schemas Pydantic partages -- le *contrat de donnees* de l'API."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.config import DEFAULT_MODEL


# --- Authentification -------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UserRead(BaseModel):
    id: int
    username: str
    email: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Chat -------------------------------------------------------------------
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryMessage] = Field(default_factory=list)
    model: str = DEFAULT_MODEL
    role: Literal["dm", "player"] = "dm"
    campaign_id: Optional[int] = None            # pour persister l'historique
    resend: bool = False                         # relance : ne pas re-persister le message user


class ModelsResponse(BaseModel):
    models: List[str]


class HealthResponse(BaseModel):
    status: str = "ok"


# --- Sources RAG ------------------------------------------------------------
class RagSource(BaseModel):
    name: str
    chunks: int = 0


class CrawlRequest(BaseModel):
    url: str
    max_pages: Optional[int] = None


class RagSourcesResponse(BaseModel):
    sources: List[RagSource]


# --- Campagnes --------------------------------------------------------------
class CampaignCreate(BaseModel):
    name: str
    description: str = ""
    setting: Optional[str] = None


class CampaignSummary(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    role: Literal["dm", "player"]
    status: str = "active"                        # active | ended


class CampaignsResponse(BaseModel):
    campaigns: List[CampaignSummary]


class SoloStartRequest(BaseModel):
    genre: str                                    # heroique | sombre | enquete | donjon | intrigue | horreur
    length: str = "courte"                        # courte | moyenne | longue
    tone: str = "serieux"                         # serieux | heroique | leger
    pitch: Optional[str] = None                   # souhait libre du joueur
    character_id: int                             # le perso (existant) incarne en solo


class SoloAdventure(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    genre: Optional[str] = None
    character_name: Optional[str] = None
    acts: int = 0
    created_at: Optional[str] = None


class SoloAdventuresResponse(BaseModel):
    adventures: List[SoloAdventure]


class ArchivedQuest(BaseModel):
    title: str
    objective: str = ""
    outcome: str                                  # won | failed
    reward: Optional[str] = None


class CampaignArchive(BaseModel):
    id: int
    name: str
    status: str = "ended"
    won: List[ArchivedQuest] = Field(default_factory=list)
    failed: List[ArchivedQuest] = Field(default_factory=list)


# --- Personnages ------------------------------------------------------------
class CharacterCreate(BaseModel):
    name: str
    race: str
    character_class: str
    class_level: int = 1
    alignment: Optional[str] = None
    background: Optional[str] = None
    description: Optional[str] = None
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    gold: int = 0
    silver: int = 0
    copper: int = 0
    starting_items: List[int] = Field(default_factory=list)  # objets de la Forge donnes a la creation
    subclass: Optional[str] = None
    skill_proficiencies: List[str] = Field(default_factory=list)
    feats: Optional[str] = None
    other_proficiencies: Optional[str] = None
    ability_bonus: Dict[str, int] = Field(default_factory=dict)   # bonus d'historique (+2/+1)
    spells: List[str] = Field(default_factory=list)
    campaign_id: Optional[int] = None


class CharacterRead(BaseModel):
    id: int
    name: str
    race: str
    character_class: str
    class_level: int
    alignment: Optional[str] = None
    background: Optional[str] = None
    description: Optional[str] = None
    strength: int
    dexterity: int
    constitution: int
    intelligence: int
    wisdom: int
    charisma: int
    gold: int = 0
    silver: int = 0
    copper: int = 0
    electrum: int = 0
    platinum: int = 0
    subclass: Optional[str] = None
    size: Optional[str] = None
    speed: int = 9
    max_hp: int = 0
    current_hp: int = 0
    temp_hp: int = 0
    hit_dice_used: int = 0
    death_succ: int = 0
    death_fail: int = 0
    inspiration: bool = False
    armor_class: Optional[int] = None
    spell_ability: Optional[str] = None
    save_proficiencies: List[str] = Field(default_factory=list)
    skill_proficiencies: List[str] = Field(default_factory=list)
    languages: Optional[str] = None
    feats: Optional[str] = None
    species_traits: Optional[str] = None
    class_features: Optional[str] = None
    other_proficiencies: Optional[str] = None
    personality: Optional[str] = None
    spells: List[str] = Field(default_factory=list)
    campaign_id: Optional[int] = None
    campaign_name: Optional[str] = None
    campaigns: List[int] = Field(default_factory=list)   # campagnes ou le PJ participe


class CharacterMoneyUpdate(BaseModel):
    gold: int = 0
    silver: int = 0
    copper: int = 0


class CharacterSheetUpdate(BaseModel):
    """Mise a jour des champs editables de la feuille de personnage."""
    subclass: Optional[str] = None
    size: Optional[str] = None
    speed: Optional[int] = None
    max_hp: Optional[int] = None
    current_hp: Optional[int] = None
    temp_hp: Optional[int] = None
    hit_dice_used: Optional[int] = None
    death_succ: Optional[int] = None
    death_fail: Optional[int] = None
    inspiration: Optional[bool] = None
    armor_class: Optional[int] = None
    spell_ability: Optional[str] = None
    save_proficiencies: Optional[List[str]] = None
    skill_proficiencies: Optional[List[str]] = None
    languages: Optional[str] = None
    feats: Optional[str] = None
    species_traits: Optional[str] = None
    class_features: Optional[str] = None
    other_proficiencies: Optional[str] = None
    personality: Optional[str] = None
    alignment: Optional[str] = None
    background: Optional[str] = None
    gold: Optional[int] = None
    silver: Optional[int] = None
    copper: Optional[int] = None
    electrum: Optional[int] = None
    platinum: Optional[int] = None


class CharactersResponse(BaseModel):
    characters: List[CharacterRead]


# --- Sorts ------------------------------------------------------------------
class SpellRef(BaseModel):
    index: str
    name: str
    level: int = 0


class ClassSpellsResponse(BaseModel):
    caster: bool = False
    max_spell_level: int = 0
    spells: List[SpellRef] = Field(default_factory=list)


class SpellDetail(BaseModel):
    index: str
    name: str
    level: int = 0
    school: Optional[str] = None
    casting_time: Optional[str] = None
    range: Optional[str] = None
    duration: Optional[str] = None
    desc: str = ""


# --- PNJ --------------------------------------------------------------------
class NpcCreate(BaseModel):
    name: str
    appearance: Optional[str] = None       # description physique
    race: Optional[str] = None
    occupation: Optional[str] = None
    attitude: Optional[str] = None
    notes: Optional[str] = None
    is_adversary: bool = False
    armor_class: Optional[int] = None
    hit_points: Optional[int] = None
    challenge_rating: Optional[str] = None
    monster_ref: Optional[str] = None


class NpcRead(BaseModel):
    id: int
    name: str
    appearance: Optional[str] = None
    race: Optional[str] = None
    occupation: Optional[str] = None
    attitude: Optional[str] = None
    notes: Optional[str] = None
    is_adversary: bool = False
    armor_class: Optional[int] = None
    hit_points: Optional[int] = None
    challenge_rating: Optional[str] = None
    monster_ref: Optional[str] = None


class NpcsResponse(BaseModel):
    npcs: List[NpcRead]


# --- Lieux ------------------------------------------------------------------
class LocationCreate(BaseModel):
    name: str
    description: Optional[str] = None
    region: Optional[str] = None


class LocationRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    region: Optional[str] = None


class LocationsResponse(BaseModel):
    locations: List[LocationRead]


# --- Quetes -----------------------------------------------------------------
class QuestCreate(BaseModel):
    title: str
    objective: str                          # obligatoire
    kind: str = "Principale"
    status: str = "active"
    description: Optional[str] = None
    giver: Optional[str] = None             # nom du PNJ donneur
    reward: Optional[str] = None
    enemies: List[str] = Field(default_factory=list)


class QuestUpdate(BaseModel):
    status: Optional[str] = None
    objective_done: Optional[bool] = None


class QuestRead(BaseModel):
    id: int
    title: str
    kind: str
    status: str
    description: Optional[str] = None
    objective: str = ""
    objective_done: bool = False
    enemies: List[str] = Field(default_factory=list)
    giver: Optional[str] = None
    reward: Optional[str] = None


class QuestsResponse(BaseModel):
    quests: List[QuestRead]


class RecruitRequest(BaseModel):
    character_id: int


# --- Forge (objets) ---------------------------------------------------------
class ItemCreate(BaseModel):
    name: str
    item_type: str = "misc"                 # weapon | armor | magic | consumable | treasure | misc
    rarity: Optional[str] = None
    value: Optional[str] = None
    gp: int = 0                             # valeur en pieces d'or
    sp: int = 0                             # pieces d'argent
    cp: int = 0                             # pieces de cuivre
    description: Optional[str] = None


class ItemRead(BaseModel):
    id: int
    name: str
    item_type: str
    rarity: Optional[str] = None
    value: Optional[str] = None
    gp: int = 0
    sp: int = 0
    cp: int = 0
    description: Optional[str] = None


class ItemsResponse(BaseModel):
    items: List[ItemRead]


class AssignRequest(BaseModel):
    item_id: int
    target_type: Literal["character", "npc", "quest"]
    target_id: int
    quantity: int = 1


class AssignmentRead(BaseModel):
    id: int
    item: ItemRead
    quantity: int = 1


class AssignmentsResponse(BaseModel):
    assignments: List[AssignmentRead]


class LibraryElement(BaseModel):
    id: int
    name: str
    summary: Optional[str] = None
    active_in: List[int] = Field(default_factory=list)
    is_adversary: bool = False
    armor_class: Optional[int] = None
    hit_points: Optional[int] = None
    challenge_rating: Optional[str] = None


class LibraryResponse(BaseModel):
    campaigns: List[CampaignSummary]
    npcs: List[LibraryElement]
    locations: List[LibraryElement]
    items: List[ItemRead]


class ActivateRequest(BaseModel):
    element_type: Literal["npc", "location", "character"]
    element_id: int
    campaign_id: int


class MessageRead(BaseModel):
    role: str
    content: str
    author: Optional[str] = None


class MessagesResponse(BaseModel):
    messages: List[MessageRead]


class ActionResult(BaseModel):
    ok: bool = True
    message: str = ""


# --- Etat de campagne (GET /api/gamemaster/state) ---------------------------
class Companion(BaseModel):
    name: str
    detail: str = ""
    level: Optional[int] = None


class CampaignStateQuest(BaseModel):
    title: str
    status: str = ""


class CampaignState(BaseModel):
    active: bool = False
    name: Optional[str] = None
    party_level: Optional[int] = None
    in_combat: bool = False
    counts: Dict[str, int] = Field(default_factory=dict)
    companions: List[Companion] = Field(default_factory=list)
    quests: List[CampaignStateQuest] = Field(default_factory=list)
    omen: Optional[str] = None


# --- Recherche de regles (reference rapide) ---------------------------------
class RuleRef(BaseModel):
    name: str
    index: str
    type: str                                    # spell|monster|equipment|magic|condition


class RulesIndex(BaseModel):
    available: bool = False
    items: List[RuleRef] = Field(default_factory=list)


class RuleDetail(BaseModel):
    name: str
    type: str
    meta: List[str] = Field(default_factory=list)
    desc: str = ""


# --- Arsenal (catalogue d'objets D&D / SRD) ---------------------------------
class ArsenalItem(BaseModel):
    index: str
    name: str


class ArsenalCategory(BaseModel):
    key: str
    label: str
    kind: str                                    # equipment | magic
    items: List[ArsenalItem] = Field(default_factory=list)


class ArsenalCatalog(BaseModel):
    available: bool = False
    categories: List[ArsenalCategory] = Field(default_factory=list)


class ArsenalItemDetail(BaseModel):
    name: str
    category: str = ""
    cost: str = ""
    weight: str = ""
    rarity: str = ""
    damage: str = ""
    armor_class: str = ""
    properties: List[str] = Field(default_factory=list)
    desc: str = ""


class MonsterStats(BaseModel):
    name: str
    armor_class: Optional[int] = None
    hit_points: Optional[int] = None
    challenge_rating: Optional[str] = None


class MonstersResponse(BaseModel):
    monsters: List[ArsenalItem] = Field(default_factory=list)


class ArsenalAssign(BaseModel):
    """Attribue un objet (catalogue ou forge) a un PNJ / quete / personnage."""
    name: str
    item_type: str = "misc"
    value: Optional[str] = None
    description: Optional[str] = None
    target_type: Literal["npc", "quest", "character"]
    target_id: int


# --- Evenements SSE ---------------------------------------------------------
SSEEventType = Literal["token", "clear", "tool_call", "done", "error"]
