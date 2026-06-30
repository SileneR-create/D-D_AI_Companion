/** Conseils instantanes de creation de personnage (par classe). Hors-ligne. */
export const CLASS_ADVICE = {
  Barbare:     { role: "Combattant de melee robuste", priorities: ["FOR", "CON"], skills: ["Athletisme", "Intimidation"], tip: "Sans armure, ta CA = 10 + mod. DEX + mod. CON. Mise tout sur la Force et la Constitution." },
  Barde:       { role: "Soutien polyvalent et lanceur", priorities: ["CHA", "DEX"], skills: ["Persuasion", "Representation"], tip: "Le Charisme alimente tes sorts ; la Dexterite ameliore CA et initiative." },
  Clerc:       { role: "Soigneur et soutien divin", priorities: ["SAG", "CON"], skills: ["Intuition", "Religion"], tip: "La Sagesse fixe ton DD de sorts. La Constitution protege ta concentration." },
  Druide:      { role: "Lanceur de la nature, formes sauvages", priorities: ["SAG", "CON"], skills: ["Nature", "Survie"], tip: "Sagesse avant tout ; evite le metal (regle des armures du druide)." },
  Ensorceleur: { role: "Lanceur spontane et explosif", priorities: ["CHA", "CON"], skills: ["Arcanes", "Persuasion"], tip: "Charisme pour la puissance, Constitution pour tenir la concentration." },
  Guerrier:    { role: "Combattant adaptable", priorities: ["FOR", "CON"], skills: ["Athletisme", "Perception"], tip: "Choisis FOR (melee lourde) OU DEX (finesse/distance), puis Constitution." },
  Magicien:    { role: "Lanceur erudit aux nombreux sorts", priorities: ["INT", "CON"], skills: ["Arcanes", "Investigation"], tip: "Intelligence pour tes sorts ; Constitution pour la concentration." },
  Moine:       { role: "Artiste martial agile", priorities: ["DEX", "SAG"], skills: ["Acrobaties", "Discretion"], tip: "Sans armure, CA = 10 + mod. DEX + mod. SAG. Repartis sur ces deux carac." },
  Paladin:     { role: "Guerrier sacre, auras et soins", priorities: ["FOR", "CHA"], skills: ["Athletisme", "Persuasion"], tip: "Force pour frapper, Charisme pour tes sorts et tes auras." },
  Rodeur:      { role: "Pisteur, archer et explorateur", priorities: ["DEX", "SAG"], skills: ["Discretion", "Survie"], tip: "Dexterite pour l'arc, Sagesse pour tes sorts de rodeur." },
  Roublard:    { role: "Specialiste furtif (attaque sournoise)", priorities: ["DEX", "INT"], skills: ["Discretion", "Escamotage"], tip: "Dexterite avant tout. La maitrise de Discretion est presque obligatoire." },
  Occultiste:  { role: "Pactisien aux invocations occultes", priorities: ["CHA", "CON"], skills: ["Arcanes", "Tromperie"], tip: "Charisme pour tes sorts d'occultiste ; Constitution pour encaisser." },
};

export const getAdvice = (cls) => CLASS_ADVICE[cls] || null;
