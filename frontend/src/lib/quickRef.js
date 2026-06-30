/** Reference rapide (SRD 5e, FR) — anti-seche hors-ligne pour La Table. */

export const CONDITIONS = [
  { name: "A terre", effect: "Deplacement en rampant uniquement. Desavantage a ses jets d'attaque. Attaques contre lui : avantage si l'attaquant est a 1,50 m ou moins, sinon desavantage." },
  { name: "Agrippe", effect: "Vitesse 0. Prend fin si l'agrippeur est neutralise ou si la creature est mise hors de portee." },
  { name: "Aveugle", effect: "Echoue tout jet exigeant la vue. Desavantage a ses attaques ; les attaques contre lui ont l'avantage." },
  { name: "Assourdi", effect: "Echoue tout jet exigeant l'ouie." },
  { name: "Charme", effect: "Ne peut attaquer le charmeur ni le cibler par une capacite nefaste. Le charmeur a l'avantage aux interactions sociales avec lui." },
  { name: "Effraye", effect: "Desavantage aux jets tant que la source est en vue. Ne peut volontairement s'en approcher." },
  { name: "Empoisonne", effect: "Desavantage aux jets d'attaque et aux jets de caracteristique." },
  { name: "Entrave", effect: "Vitesse 0. Desavantage a ses attaques et aux sauvegardes de Dexterite. Les attaques contre lui ont l'avantage." },
  { name: "Etourdi", effect: "Neutralise, ne peut bouger, parle avec hesitation. Echoue auto. les sauvegardes de For et Dex. Attaques contre lui : avantage." },
  { name: "Inconscient", effect: "Neutralise, lache ce qu'il tient, tombe a terre. Echoue auto. For/Dex. Attaques contre lui : avantage ; coup critique si l'attaquant est a 1,50 m." },
  { name: "Invisible", effect: "Impossible a voir sans moyen special. Avantage a ses attaques ; les attaques contre lui ont le desavantage." },
  { name: "Neutralise", effect: "Aucune action ni reaction possible." },
  { name: "Paralyse", effect: "Neutralise, ne peut bouger ni parler. Echoue auto. For/Dex. Attaques contre lui : avantage ; coup critique a 1,50 m." },
  { name: "Petrifie", effect: "Transforme en matiere solide. Neutralise, ne percoit plus. Resistance a tous les degats. Immunite poison et maladie." },
  { name: "Epuisement", effect: "Niv.1 desavantage aux tests ; 2 vitesse /2 ; 3 desavantage attaques et sauvegardes ; 4 PV max /2 ; 5 vitesse 0 ; 6 mort." },
];

export const ACTIONS = [
  { name: "Attaquer", effect: "Une attaque (plus selon les capacites comme Attaque supplementaire)." },
  { name: "Lancer un sort", effect: "Selon le temps d'incantation du sort (souvent 1 action)." },
  { name: "Foncer", effect: "Gagne un deplacement supplementaire egal a sa vitesse ce tour." },
  { name: "Se desengager", effect: "Son deplacement ne provoque pas d'attaque d'opportunite ce tour." },
  { name: "Esquiver", effect: "Les attaques contre lui ont le desavantage ; avantage aux sauvegardes de Dex (s'il voit l'attaquant)." },
  { name: "Aider", effect: "Donne l'avantage : a un allie attaquant une cible a 1,50 m, ou a un test de caracteristique." },
  { name: "Se cacher", effect: "Jet de Dexterite (Discretion) pour se dissimuler." },
  { name: "Preparer", effect: "Definit un declencheur et une reaction a executer quand il survient." },
  { name: "Chercher", effect: "Jet de Sagesse (Perception) ou d'Intelligence (Investigation)." },
  { name: "Utiliser un objet", effect: "Interagir avec ou activer un objet (2e interaction de l'objet)." },
];

export const COVER = [
  { name: "A demi", effect: "+2 a la CA et aux sauvegardes de Dexterite." },
  { name: "Aux trois quarts", effect: "+5 a la CA et aux sauvegardes de Dexterite." },
  { name: "Total", effect: "Ne peut etre cible directement par une attaque ou un sort." },
];

export const DCS = [
  ["Tres facile", "5"], ["Facile", "10"], ["Moyen", "15"],
  ["Difficile", "20"], ["Tres difficile", "25"], ["Quasi impossible", "30"],
];

export const RESTS = [
  { name: "Repos court (1 h)", effect: "Depenser des des de vie (1d + mod. CON chacun) pour recuperer des PV." },
  { name: "Repos long (8 h)", effect: "PV au maximum. Recupere la moitie des des de vie (min 1). Capacites journalieres rechargees." },
];
