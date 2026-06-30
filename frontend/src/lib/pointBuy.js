/**
 * Point-buy 5e (cote client) -- doit refleter backend/rules5e.py.
 * Empeche les fiches boostees au niveau 1 ; les ameliorations (ASI) sont
 * debloquees aux niveaux 4/8/12/16/19 (2 points chacune, max 20).
 */
export const COST = { 8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7, 15: 9 };
export const BUDGET = 27;
export const STANDARD_ARRAY = [15, 14, 13, 12, 10, 8];
export const ABILITIES = [
  ["strength", "FOR"], ["dexterity", "DEX"], ["constitution", "CON"],
  ["intelligence", "INT"], ["wisdom", "SAG"], ["charisma", "CHA"],
];

export const asiPoints = (level) => 2 * [4, 8, 12, 16, 19].filter((l) => level >= l).length;

/** Modificateur de caracteristique. */
export const modifier = (score) => Math.floor((score - 10) / 2);
export const fmtMod = (score) => { const m = modifier(score); return m >= 0 ? `+${m}` : `${m}`; };

/** Bilan d'une repartition : points d'achat et d'amelioration utilises + validite. */
export function audit(level, scores) {
  let budgetUsed = 0;
  let bonusUsed = 0;
  let bounds = true;
  for (const v of Object.values(scores)) {
    if (v < 8 || v > 20) bounds = false;
    budgetUsed += COST[Math.min(v, 15)] ?? 99;
    bonusUsed += Math.max(0, v - 15);
  }
  const asi = asiPoints(level);
  const valid = bounds && budgetUsed <= BUDGET && bonusUsed <= asi;
  return { budgetUsed, bonusUsed, asi, budget: BUDGET, valid };
}
