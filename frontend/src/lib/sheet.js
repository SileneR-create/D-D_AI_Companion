/**
 * Moteur de calcul de la feuille de personnage (regles D&D 2024 / basic rules FR).
 *
 * Formules (sourcees des basic rules) :
 *  - Modificateur = floor((valeur - 10) / 2)
 *  - Bonus de maitrise = +2 (niv 1-4), +3 (5-8), +4 (9-12), +5 (13-16), +6 (17-20)
 *  - Initiative = mod. DEX
 *  - CA sans armure = 10 + mod. DEX (sinon valeur saisie)
 *  - Perception passive = 10 + mod. SAG (+ maitrise si Perception maitrisee)
 *  - Sauvegarde / competence = mod. carac (+ maitrise si maitrisee)
 *  - DD de sort = 8 + maitrise + mod. carac d'incantation
 *  - Attaque de sort = maitrise + mod. carac d'incantation
 */

// Cle interne -> libelle court de caracteristique.
export const ABILITY_KEYS = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"];
export const ABILITY_FR = {
  strength: "FOR", dexterity: "DEX", constitution: "CON",
  intelligence: "INT", wisdom: "SAG", charisma: "CHA",
};

// 18 competences 2024 -> caracteristique associee.
export const SKILLS = [
  ["acrobaties", "Acrobaties", "dexterity"],
  ["arcanes", "Arcanes", "intelligence"],
  ["athletisme", "Athletisme", "strength"],
  ["discretion", "Discretion", "dexterity"],
  ["dressage", "Dressage", "wisdom"],
  ["escamotage", "Escamotage", "dexterity"],
  ["histoire", "Histoire", "intelligence"],
  ["intimidation", "Intimidation", "charisma"],
  ["intuition", "Intuition", "wisdom"],
  ["investigation", "Investigation", "intelligence"],
  ["medecine", "Medecine", "wisdom"],
  ["nature", "Nature", "intelligence"],
  ["perception", "Perception", "wisdom"],
  ["persuasion", "Persuasion", "charisma"],
  ["religion", "Religion", "intelligence"],
  ["representation", "Representation", "charisma"],
  ["survie", "Survie", "wisdom"],
  ["tromperie", "Tromperie", "charisma"],
];

// Donnees par classe : de de vie, sauvegardes maitrisees, carac d'incantation.
export const CLASS_DATA = {
  Barbare:     { hit: 12, saves: ["strength", "constitution"], spell: null },
  Barde:       { hit: 8,  saves: ["dexterity", "charisma"],     spell: "charisma" },
  Clerc:       { hit: 8,  saves: ["wisdom", "charisma"],        spell: "wisdom" },
  Druide:      { hit: 8,  saves: ["intelligence", "wisdom"],    spell: "wisdom" },
  Ensorceleur: { hit: 6,  saves: ["constitution", "charisma"],  spell: "charisma" },
  Guerrier:    { hit: 10, saves: ["strength", "constitution"],  spell: null },
  Magicien:    { hit: 6,  saves: ["intelligence", "wisdom"],    spell: "intelligence" },
  Moine:       { hit: 8,  saves: ["strength", "dexterity"],     spell: null },
  Paladin:     { hit: 10, saves: ["wisdom", "charisma"],        spell: "charisma" },
  Rodeur:      { hit: 10, saves: ["strength", "dexterity"],     spell: "wisdom" },
  Roublard:    { hit: 8,  saves: ["dexterity", "intelligence"], spell: null },
  Occultiste:  { hit: 8,  saves: ["wisdom", "charisma"],        spell: "charisma" },
};

// Vitesse (m) / taille par espece (valeurs par defaut, editables).
export const SPECIES_DATA = {
  Humain: { speed: 9, size: "M" }, Elfe: { speed: 9, size: "M" }, "Demi-elfe": { speed: 9, size: "M" },
  Nain: { speed: 7.5, size: "M" }, Halfelin: { speed: 7.5, size: "P" }, "Demi-orc": { speed: 9, size: "M" },
  Tieffelin: { speed: 9, size: "M" }, Drakeide: { speed: 9, size: "M" }, Gnome: { speed: 7.5, size: "P" },
};

export const mod = (score) => Math.floor(((Number(score) || 10) - 10) / 2);
export const fmtMod = (m) => (m >= 0 ? `+${m}` : `${m}`);

export function profBonus(level) {
  const l = Number(level) || 1;
  return 2 + Math.floor((Math.max(1, Math.min(20, l)) - 1) / 4);
}

export function classData(cls) {
  return CLASS_DATA[cls] || { hit: 8, saves: [], spell: null };
}

/** Sauvegardes maitrisees : choix stocke, sinon defaut de la classe. */
export function effectiveSaves(c) {
  if (Array.isArray(c.save_proficiencies) && c.save_proficiencies.length) return c.save_proficiencies;
  return classData(c.character_class).saves;
}

/** Caracteristique d'incantation effective (override sinon selon la classe). */
export function spellAbility(c) {
  return c.spell_ability || classData(c.character_class).spell;
}

/** Calcule l'ensemble des valeurs derivees de la feuille. */
export function computeSheet(c) {
  const pb = profBonus(c.class_level);
  const mods = Object.fromEntries(ABILITY_KEYS.map((k) => [k, mod(c[k])]));
  const saves = effectiveSaves(c);
  const skillProf = c.skill_proficiencies || [];

  const initiative = mods.dexterity;
  const ac = (c.armor_class ?? null) !== null ? c.armor_class : 10 + mods.dexterity;
  const perceptionProf = skillProf.includes("perception");
  const passivePerception = 10 + mods.wisdom + (perceptionProf ? pb : 0);

  const savingThrows = ABILITY_KEYS.map((k) => ({
    key: k, label: ABILITY_FR[k], proficient: saves.includes(k),
    value: mods[k] + (saves.includes(k) ? pb : 0),
  }));

  const skills = SKILLS.map(([key, label, ability]) => ({
    key, label, ability, proficient: skillProf.includes(key),
    value: mods[ability] + (skillProf.includes(key) ? pb : 0),
  }));

  const sa = spellAbility(c);
  const caster = !!sa;
  const spellMod = caster ? mods[sa] : 0;
  const spellSaveDC = caster ? 8 + pb + spellMod : null;
  const spellAttack = caster ? pb + spellMod : null;

  return {
    profBonus: pb, mods, initiative, ac, passivePerception,
    savingThrows, skills,
    caster, spellAbility: sa, spellSaveDC, spellAttack,
    hitDie: classData(c.character_class).hit,
  };
}
