/**
 * Options de creation (D&D 2024) : sous-classes par classe et historiques.
 * Les historiques portent leurs maitrises de competences, un outil, un don d'origine
 * et 3 caracteristiques eligibles au bonus (+2 / +1). Cles competences/carac alignees sur sheet.js.
 */

// Sous-classes par classe (noms FR, PHB 2024).
export const SUBCLASSES = {
  Barbare: ["Voie de l'Arbre-monde", "Voie du Coeur sauvage", "Voie du Berserker", "Voie du Zelateur"],
  Barde: ["College de la Danse", "College du Savoir", "College de la Seduction", "College de la Vaillance"],
  Clerc: ["Domaine de la Vie", "Domaine de la Lumiere", "Domaine de la Ruse", "Domaine de la Guerre"],
  Druide: ["Cercle de la Terre", "Cercle de la Lune", "Cercle des Mers", "Cercle des Etoiles"],
  Ensorceleur: ["Sorcellerie draconique", "Sorcellerie sauvage", "Sorcellerie aberrante", "Ame d'horloge"],
  Guerrier: ["Champion", "Chevalier occultiste", "Maitre de guerre", "Soldat psi"],
  Magicien: ["Abjurateur", "Devin", "Evocateur", "Illusionniste"],
  Moine: ["Credo de la Main ouverte", "Credo des Elements", "Credo de la Misericorde", "Credo de l'Ombre"],
  Paladin: ["Serment de devotion", "Serment de gloire", "Serment des anciens", "Serment de vengeance"],
  Rodeur: ["Belluaire", "Chasseur", "Traqueur des tenebres", "Vagabond feerique"],
  Roublard: ["Voleur", "Ame aceree", "Arnaqueur arcanique", "Assassin"],
  Occultiste: ["Protecteur Archifee", "Protecteur Celeste", "Protecteur Fielon", "Protecteur Grand Ancien"],
};

// Historiques 2024 : abilities (3 eligibles), skills (2), tool, feat (don d'origine).
export const BACKGROUNDS = [
  { name: "Acolyte", abilities: ["intelligence", "wisdom", "charisma"], skills: ["intuition", "religion"], tool: "Materiel de calligraphie", feat: "Initie a la magie (Clerc)" },
  { name: "Artisan", abilities: ["strength", "dexterity", "intelligence"], skills: ["investigation", "persuasion"], tool: "Outils d'artisan", feat: "Artisan" },
  { name: "Artiste", abilities: ["strength", "dexterity", "charisma"], skills: ["acrobaties", "representation"], tool: "Instrument de musique", feat: "Musicien" },
  { name: "Charlatan", abilities: ["dexterity", "constitution", "charisma"], skills: ["escamotage", "tromperie"], tool: "Materiel de faussaire", feat: "Doue" },
  { name: "Criminel", abilities: ["dexterity", "constitution", "intelligence"], skills: ["discretion", "escamotage"], tool: "Outils de voleur", feat: "Vigilance" },
  { name: "Ermite", abilities: ["constitution", "wisdom", "charisma"], skills: ["medecine", "religion"], tool: "Kit d'herboriste", feat: "Guerisseur" },
  { name: "Fermier", abilities: ["strength", "constitution", "wisdom"], skills: ["dressage", "nature"], tool: "Outils de menuisier", feat: "Robuste" },
  { name: "Garde", abilities: ["strength", "intelligence", "wisdom"], skills: ["athletisme", "perception"], tool: "Jeu de societe", feat: "Vigilance" },
  { name: "Guide", abilities: ["dexterity", "constitution", "wisdom"], skills: ["discretion", "survie"], tool: "Materiel de cartographie", feat: "Initie a la magie (Druide)" },
  { name: "Marchand", abilities: ["constitution", "intelligence", "charisma"], skills: ["dressage", "persuasion"], tool: "Outils de navigateur", feat: "Chanceux" },
  { name: "Marin", abilities: ["strength", "dexterity", "wisdom"], skills: ["acrobaties", "perception"], tool: "Outils de navigateur", feat: "Bagarreur de taverne" },
  { name: "Noble", abilities: ["strength", "intelligence", "charisma"], skills: ["histoire", "persuasion"], tool: "Jeu de societe", feat: "Doue" },
  { name: "Sage", abilities: ["constitution", "intelligence", "wisdom"], skills: ["arcanes", "histoire"], tool: "Materiel de calligraphie", feat: "Initie a la magie (Magicien)" },
  { name: "Scribe", abilities: ["dexterity", "intelligence", "wisdom"], skills: ["investigation", "perception"], tool: "Materiel de calligraphie", feat: "Doue" },
  { name: "Soldat", abilities: ["strength", "dexterity", "constitution"], skills: ["athletisme", "intimidation"], tool: "Jeu de societe", feat: "Attaquant sauvage" },
  { name: "Voyageur", abilities: ["dexterity", "wisdom", "charisma"], skills: ["discretion", "intuition"], tool: "Outils de voleur", feat: "Chanceux" },
];

export const backgroundByName = (n) => BACKGROUNDS.find((b) => b.name === n) || null;
