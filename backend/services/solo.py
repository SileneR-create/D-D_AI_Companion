"""Générateur d'aventure solo — approche HYBRIDE CADRÉE.

L'ossature (genre, actes, lieu de départ, antagoniste, objectif) est produite
par des RÈGLES déterministes : c'est fiable et sans dérive même sur un petit
modèle. Le LLM ne fait ensuite qu'incarner le narrateur scène par scène à
l'intérieur de cette ossature (cf. prompts.SOLO_PROMPT + contexte injecté).
"""
import random
from typing import Optional

# Nombre d'actes selon la durée choisie.
LENGTH_ACTS = {"courte": 3, "moyenne": 5, "longue": 8}

# Squelette narratif par nombre d'actes (titres d'actes, dans l'ordre).
ACT_SKELETONS = {
    3: ["L'accroche", "La confrontation", "Le dénouement"],
    5: ["L'accroche", "L'enquête", "La complication", "La confrontation", "Le dénouement"],
    8: ["L'accroche", "Les premiers indices", "La première épreuve", "La révélation",
        "La complication", "La course contre la montre", "La confrontation finale", "Le dénouement"],
}

# Bibliothèque de genres : décor, lieux de départ, antagonistes, objectifs, ambiance.
GENRES = {
    "heroique": {
        "label": "Heroic fantasy",
        "ambiance": "une aventure héroïque pleine d'espoir et de hauts faits",
        "regions": ["la Vallée du Couchant", "les Marches d'Argent", "le Royaume de Coralan"],
        "starts": [
            ("Le village de Tertre-Vert", "Un hameau paisible cerné de champs dorés, soudain troublé par une menace."),
            ("La cité libre de Hautregard", "Une ville prospère aux remparts blancs où courent les premières rumeurs de péril."),
        ],
        "antagonists": [
            ("Vorghal le Conquérant", "seigneur de guerre", 16, 75, "5"),
            ("la Prêtresse Maledys", "cheffe d'un culte guerrier", 15, 58, "4"),
        ],
        "objectives": [
            "rassembler les héros et repousser la menace qui pèse sur la région",
            "récupérer une relique sacrée avant qu'elle ne tombe en de mauvaises mains",
        ],
    },
    "sombre": {
        "label": "Dark fantasy",
        "ambiance": "une aventure sombre où l'espoir se paie cher",
        "regions": ["les Terres Grises", "le Comté des Brumes", "la Marche Noire"],
        "starts": [
            ("Le bourg de Cendrelune", "Un village exsangue sous un ciel perpétuellement gris, terré derrière ses palissades."),
            ("L'abbaye en ruine de Saint-Rouvre", "Des murs lépreux où survit une poignée d'âmes apeurées."),
        ],
        "antagonists": [
            ("le Comte Mortfeu", "seigneur vampire", 16, 90, "6"),
            ("la Sorcière des Tourbières", "praticienne d'arts maudits", 14, 65, "5"),
        ],
        "objectives": [
            "lever la malédiction qui ronge lentement la contrée",
            "découvrir qui se cache derrière les disparitions et y mettre fin",
        ],
    },
    "enquete": {
        "label": "Mystère & enquête",
        "ambiance": "une enquête où chaque indice compte",
        "regions": ["la ville-port de Sombrequai", "la capitale de Valmont", "le quartier des orfèvres"],
        "starts": [
            ("La taverne du Héron Boiteux", "Un point de rendez-vous enfumé où un commanditaire attend, nerveux."),
            ("La demeure des Aubterre", "Un manoir cossu où vient d'être commis l'inexplicable."),
        ],
        "antagonists": [
            ("Maître Sélian Doré", "notable au-dessus de tout soupçon", 13, 45, "3"),
            ("l'Ombre", "criminel insaisissable", 15, 52, "4"),
        ],
        "objectives": [
            "démasquer le coupable et révéler la vérité derrière le crime",
            "retrouver la personne disparue avant qu'il ne soit trop tard",
        ],
    },
    "donjon": {
        "label": "Exploration & donjon",
        "ambiance": "une expédition périlleuse dans les profondeurs",
        "regions": ["les Pics Brisés", "la forêt de Vivebois", "les ruines de Khar-Dûm"],
        "starts": [
            ("L'entrée béante de Khar-Dûm", "Un portail de pierre runique s'ouvrant sur les ténèbres souterraines."),
            ("Le camp de base des pillards", "Un bivouac de fortune au seuil d'un complexe oublié."),
        ],
        "antagonists": [
            ("Ghul'mor le Dévoreur", "aberration tapie au cœur du donjon", 15, 82, "6"),
            ("le Roi-Gobelin Skarn", "chef de guerre des profondeurs", 14, 60, "4"),
        ],
        "objectives": [
            "atteindre le cœur du complexe et s'emparer du trésor légendaire",
            "sceller la menace qui sommeille au plus profond des ruines",
        ],
    },
    "intrigue": {
        "label": "Intrigue politique",
        "ambiance": "un jeu d'influence où les mots tranchent autant que l'acier",
        "regions": ["la cour de Lothaire", "la république marchande de Sérène", "le duché de Valmberg"],
        "starts": [
            ("Le grand bal du Doge", "Une nuit de faste et de masques où se trament mille complots."),
            ("La chancellerie de Sérène", "Des couloirs feutrés où se décide le sort du duché."),
        ],
        "antagonists": [
            ("le Chancelier Vidal", "maître manipulateur de la cour", 13, 48, "3"),
            ("la Baronne d'Estrac", "rivale ambitieuse et sans scrupules", 14, 50, "4"),
        ],
        "objectives": [
            "déjouer le complot qui menace de renverser l'ordre établi",
            "gagner la faveur du souverain tout en survivant aux rivalités de cour",
        ],
    },
    "horreur": {
        "label": "Horreur",
        "ambiance": "une descente angoissante où la peur est tangible",
        "regions": ["les marais de Lamente", "le village isolé de Creux-Noir", "la côte des Naufrageurs"],
        "starts": [
            ("L'auberge du Dernier Repos", "Le seul refuge éclairé à des lieues à la ronde, et la nuit tombe."),
            ("La chapelle abandonnée", "Un sanctuaire profané d'où monte une plainte sans source."),
        ],
        "antagonists": [
            ("l'Innommable de Creux-Noir", "horreur indicible", 14, 70, "5"),
            ("le Pasteur Galenne", "prophète d'un culte abject", 13, 44, "3"),
        ],
        "objectives": [
            "survivre à la nuit et comprendre la nature de l'horreur qui rôde",
            "briser le rituel avant qu'il ne s'achève",
        ],
    },
}

TONES = {
    "serieux": "Le ton est grave et immersif.",
    "heroique": "Le ton est épique et exalté.",
    "leger": "Le ton est léger, avec une pointe d'humour.",
}

# Objectif générique par acte (réutilise l'antagoniste/objectif global).
def _act_goals(skeleton: list[str], objective: str, antagonist: str) -> list[dict]:
    goals = {
        "L'accroche": "Planter le décor, présenter le commanditaire ou l'événement déclencheur, donner le premier indice.",
        "Les premiers indices": "Suivre les premières pistes, rencontrer des PNJ utiles, cerner l'enjeu.",
        "L'enquête": "Recueillir indices et témoignages, écarter les fausses pistes.",
        "La première épreuve": "Affronter un premier obstacle (combat, énigme ou choix moral).",
        "La révélation": "Dévoiler un retournement : l'ampleur réelle de la menace apparaît.",
        "La complication": "Tout se complique : un allié trahit, un piège se referme, le temps presse.",
        "La course contre la montre": "Agir vite avant que l'antagoniste n'accomplisse son dessein.",
        "La confrontation": f"Affronter {antagonist} et tenter d'accomplir : {objective}.",
        "La confrontation finale": f"Affrontement décisif contre {antagonist} : {objective}.",
        "Le dénouement": "Récolter les conséquences des choix, conclure l'aventure, distribuer les récompenses.",
    }
    return [{"title": t, "goal": goals.get(t, "Faire progresser l'intrigue.")} for t in skeleton]


def build_outline(genre: str, length: str, tone: str, pitch: Optional[str],
                  character_name: Optional[str]) -> dict:
    """Construit une ossature d'aventure solo déterministe (mais variée)."""
    g = GENRES.get(genre, GENRES["heroique"])
    n_acts = LENGTH_ACTS.get(length, 3)
    skeleton = ACT_SKELETONS.get(n_acts, ACT_SKELETONS[3])

    start_name, start_desc = random.choice(g["starts"])
    region = random.choice(g["regions"])
    ant_name, ant_occ, ant_ac, ant_hp, ant_cr = random.choice(g["antagonists"])
    objective = random.choice(g["objectives"])
    hero = character_name or "le héros"

    pitch_line = f" Souhait du joueur : {pitch.strip()}." if pitch and pitch.strip() else ""
    synopsis = (
        f"{TONES.get(tone, '')} Dans {region}, {hero} se trouve mêlé à {g['ambiance']}. "
        f"Tout commence à {start_name.lower()}. L'enjeu : {objective}. "
        f"L'antagoniste pressenti est {ant_name} ({ant_occ}).{pitch_line}"
    ).strip()

    title = f"{g['label']} — {region.split(' ', 1)[-1].capitalize()}"

    return {
        "title": title,
        "genre": g["label"],
        "synopsis": synopsis,
        "acts": _act_goals(skeleton, objective, ant_name),
        "start_location": {"name": start_name, "description": start_desc, "region": region},
        "antagonist": {"name": ant_name, "occupation": ant_occ, "armor_class": ant_ac,
                       "hit_points": ant_hp, "challenge_rating": ant_cr},
        "objective": objective,
        "tone": tone,
    }
