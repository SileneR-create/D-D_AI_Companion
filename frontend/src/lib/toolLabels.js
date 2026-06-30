/**
 * Traduction des noms techniques d'outils MCP en libelles lisibles (FR).
 * Le backend envoie des noms qualifies (`serveur.outil`) ; on retire le prefixe
 * puis on cherche un libelle ; a defaut, on "humanise" le nom brut.
 */
const LABELS = {
  // --- Serveur Regles (DnD) ---
  search_all_categories: "Recherche dans les sources",
  filter_spells_by_level: "Recherche de sorts",
  find_monsters_by_challenge_rating: "Recherche de monstres",
  generate_treasure_hoard: "Generation de tresor",
  get_class_starting_equipment: "Equipement de depart",
  search_equipment_by_cost: "Recherche d'equipement",
  verify_with_api: "Verification des regles",
  check_api_health: "Verification des sources",

  // --- Serveur Maitre du Jeu (gamemaster) ---
  create_campaign: "Creation de campagne",
  get_campaign_info: "Lecture de la campagne",
  list_campaigns: "Liste des campagnes",
  load_campaign: "Chargement de campagne",
  create_character: "Creation de personnage",
  get_character: "Lecture d'un personnage",
  update_character: "Mise a jour du personnage",
  bulk_update_characters: "Mise a jour des personnages",
  add_item_to_character: "Ajout d'objet",
  list_characters: "Liste des personnages",
  create_npc: "Creation de PNJ",
  get_npc: "Lecture d'un PNJ",
  list_npcs: "Liste des PNJ",
  create_location: "Creation de lieu",
  get_location: "Lecture d'un lieu",
  list_locations: "Liste des lieux",
  create_quest: "Creation de quete",
  update_quest: "Mise a jour de quete",
  list_quests: "Liste des quetes",
  update_game_state: "Mise a jour de la partie",
  get_game_state: "Lecture de l'etat de jeu",
  start_combat: "Debut du combat",
  end_combat: "Fin du combat",
  next_turn: "Tour suivant",
  add_session_note: "Note de session",
  get_sessions: "Lecture des sessions",
  add_event: "Ajout d'evenement",
  get_events: "Lecture des evenements",
  roll_dice: "Lancer de des",
  calculate_experience: "Calcul d'experience",
};

function humanize(raw) {
  return raw.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Libelle lisible d'un nom d'outil (qualifie ou non). */
export function toolLabel(name) {
  if (!name) return "";
  const bare = name.includes(".") ? name.split(".").pop() : name;
  return LABELS[bare] || humanize(bare);
}
