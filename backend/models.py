"""Modeles de base de donnees (SQLModel)."""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel, UniqueConstraint


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: Optional[str] = Field(default=None)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Campaign(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)   # unicite par utilisateur, pas globale
    description: Optional[str] = Field(default=None)
    dm_id: int = Field(foreign_key="user.id")
    status: str = "active"                        # active | ended | solo
    ended_at: Optional[datetime] = Field(default=None)
    solo_outline: Optional[str] = Field(default=None)  # JSON : ossature d'aventure solo (actes, antagoniste...)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CampaignMember(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("campaign_id", "user_id", name="uq_member"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_id: int = Field(foreign_key="campaign.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    role: str = "player"
    character_name: Optional[str] = Field(default=None)
    joined_at: datetime = Field(default_factory=datetime.utcnow)


class Character(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    campaign_id: Optional[int] = Field(default=None, foreign_key="campaign.id", index=True)
    name: str
    race: str
    character_class: str
    class_level: int = 1
    alignment: Optional[str] = None
    background: Optional[str] = None
    description: Optional[str] = None
    spells: Optional[str] = None                # JSON
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    # --- Bourse (pieces) ---
    gold: int = 0                                # po
    silver: int = 0                              # pa
    copper: int = 0                              # pc
    electrum: int = 0                            # pe
    platinum: int = 0                            # pp
    # --- Feuille de personnage (2024) ---
    subclass: Optional[str] = None               # sous-classe
    size: Optional[str] = None                   # taille (P/M/G...)
    speed: int = 9                               # vitesse en metres
    max_hp: int = 0                              # points de vie max
    current_hp: int = 0                          # PV actuels
    temp_hp: int = 0                             # PV temporaires
    hit_dice_used: int = 0                       # des de vie depenses
    death_succ: int = 0                          # jets contre la mort : succes
    death_fail: int = 0                          # jets contre la mort : echecs
    inspiration: bool = False                    # inspiration heroique
    armor_class: Optional[int] = None            # CA (None = calculee 10 + mod DEX)
    spell_ability: Optional[str] = None          # carac d'incantation (None = selon classe)
    save_proficiencies: Optional[str] = None     # JSON : maitrises de sauvegarde
    skill_proficiencies: Optional[str] = None    # JSON : maitrises de competences
    languages: Optional[str] = None              # langues
    feats: Optional[str] = None                  # dons
    species_traits: Optional[str] = None         # traits d'espece
    class_features: Optional[str] = None         # capacites de classe
    other_proficiencies: Optional[str] = None    # armures/armes/outils
    personality: Optional[str] = None            # histoire & personnalite
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Npc(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_id: Optional[int] = Field(default=None, foreign_key="campaign.id", index=True)  # provenance, non restrictif
    owner_id: int = Field(foreign_key="user.id", index=True)
    name: str
    appearance: Optional[str] = None            # description physique
    race: Optional[str] = None
    occupation: Optional[str] = None
    attitude: Optional[str] = None              # friendly | neutral | hostile | unknown
    notes: Optional[str] = None
    # --- Combat (PNJ adversaire) ---
    is_adversary: bool = False                  # ennemi des heros ?
    armor_class: Optional[int] = None           # classe d'armure
    hit_points: Optional[int] = None            # points de vie
    challenge_rating: Optional[str] = None      # facteur de puissance (FP)
    monster_ref: Optional[str] = None           # monstre SRD associe (nom)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Location(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_id: Optional[int] = Field(default=None, foreign_key="campaign.id", index=True)  # provenance, non restrictif
    owner_id: int = Field(foreign_key="user.id", index=True)
    name: str
    description: Optional[str] = None
    region: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Quest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_id: int = Field(foreign_key="campaign.id", index=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    title: str
    kind: str = "Principale"
    status: str = "active"                       # active | completed | failed | on_hold
    description: Optional[str] = None
    objective: str = ""                          # objectif (obligatoire a la creation)
    objective_done: bool = False                 # valide par le MD en fin de quete
    enemies: Optional[str] = None                # JSON liste d'ennemis
    giver: Optional[str] = None                  # PNJ donneur (nom)
    reward: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Item(SQLModel, table=True):
    """Objet de la 'forge' : bibliotheque reutilisable de l'utilisateur."""
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    name: str
    item_type: str = "misc"                      # weapon | armor | magic | consumable | treasure | misc
    rarity: Optional[str] = None                 # commun | peu commun | rare | tres rare | legendaire
    value: Optional[str] = None                  # libelle libre (ex: "50 po"), legacy/optionnel
    gp: int = 0                                  # valeur : pieces d'or (po)
    sp: int = 0                                  # pieces d'argent (pa)
    cp: int = 0                                  # pieces de cuivre (pc)
    description: Optional[str] = None             # effet / description
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ItemAssignment(SQLModel, table=True):
    """Attache un objet a un PJ, un PNJ ou une quete."""
    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    target_type: str                             # character | npc | quest
    target_id: int
    quantity: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
    """Un message de conversation, rattache a une campagne (historique persistant)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    campaign_id: int = Field(foreign_key="campaign.id", index=True)
    role: str                                    # user | assistant
    content: str
    author: Optional[str] = None                 # pseudo de l'auteur (messages user)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CampaignElement(SQLModel, table=True):
    """Active un PNJ ou un lieu dans une campagne (relation N-N, selection/deselection)."""
    __table_args__ = (UniqueConstraint("campaign_id", "element_type", "element_id", name="uq_campelem"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id", index=True)
    campaign_id: int = Field(foreign_key="campaign.id", index=True)
    element_type: str                            # npc | location | character
    element_id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # (Relation N-N campagne <-> element ; uq_campelem evite les doublons.)
