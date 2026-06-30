/**
 * Limites de sorts connus par classe et niveau (5e) -- doit refleter
 * backend/services/spells.py. Cantrips et sorts comptes separement.
 */
const WIS = ["Clerc", "Druide", "Rodeur"];
const INT = ["Magicien"];
const CHA = ["Barde", "Ensorceleur", "Occultiste", "Paladin"];

const mod = (score) => Math.floor(((score || 10) - 10) / 2);

function castMod(cls, scores) {
  if (WIS.includes(cls)) return mod(scores.wisdom);
  if (INT.includes(cls)) return mod(scores.intelligence);
  if (CHA.includes(cls)) return mod(scores.charisma);
  return 0;
}

// Sorts mineurs connus : [niv1, niv4+, niv10+]
const CANTRIPS = {
  Barde: [2, 3, 4], Clerc: [3, 4, 5], Druide: [2, 3, 4],
  Ensorceleur: [4, 5, 6], Occultiste: [2, 3, 4], Magicien: [3, 4, 5],
};

// Sorts connus (lanceurs "connus") par niveau 1..20.
const KNOWN = {
  Barde: [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 15, 16, 18, 19, 19, 20, 22, 22, 22],
  Ensorceleur: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15, 15],
  Rodeur: [0, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11],
  Occultiste: [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15],
};

export function cantripsKnown(cls, level) {
  const t = CANTRIPS[cls];
  if (!t) return 0;
  return level >= 10 ? t[2] : level >= 4 ? t[1] : t[0];
}

export function spellsKnown(cls, level, scores) {
  const lvl = Math.max(1, Math.min(20, level));
  if (KNOWN[cls]) return KNOWN[cls][lvl - 1];
  const m = castMod(cls, scores);
  if (cls === "Clerc" || cls === "Druide") return Math.max(1, lvl + m);       // prepares
  if (cls === "Magicien") return 6 + 2 * (lvl - 1);                            // grimoire
  if (cls === "Paladin") return lvl < 2 ? 0 : Math.max(1, Math.floor(lvl / 2) + m);
  return 0;
}
