"""Regles 5e pour la creation de personnage : point-buy + coherence de niveau.

Objectif : empecher les fiches "boostees" des le niveau 1. Les caracteristiques
de base suivent l'achat de points (point-buy, budget 27, scores 8-15). Les
ameliorations de caracteristiques (ASI) ne sont accordees qu'aux paliers de
niveau 4/8/12/16/19 ; chacune vaut +2 points repartissables (max 20 par score).
"""
# Cout du point-buy par score (D&D 5e standard).
POINTBUY_COST = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7, 15: 9}
POINTBUY_BUDGET = 27
STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]
ASI_LEVELS = (4, 8, 12, 16, 19)

ABILITIES = ("strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma")


def asi_points(level: int) -> int:
    """Points d'amelioration disponibles a ce niveau (2 par palier atteint)."""
    return 2 * sum(1 for lv in ASI_LEVELS if level >= lv)


def validate_scores(level: int, scores: dict[str, int]) -> None:
    """Verifie qu'une repartition est legale pour ce niveau. Leve ValueError sinon."""
    budget_used = 0
    bonus_used = 0
    for ab in ABILITIES:
        v = scores.get(ab, 10)
        if not isinstance(v, int) or v < 8 or v > 20:
            raise ValueError(f"{ab}: score {v} hors limites (8 a 20).")
        base = min(v, 15)
        budget_used += POINTBUY_COST[base]
        bonus_used += max(0, v - 15)   # chaque point au-dela de 15 vient des ASI

    if budget_used > POINTBUY_BUDGET:
        raise ValueError(
            f"Repartition de base trop elevee : {budget_used}/{POINTBUY_BUDGET} points d'achat."
        )
    allowed = asi_points(level)
    if bonus_used > allowed:
        raise ValueError(
            f"Scores trop hauts pour le niveau {level} : {bonus_used} points d'amelioration "
            f"utilises pour {allowed} disponibles. (Pas de boost au niveau 1.)"
        )
