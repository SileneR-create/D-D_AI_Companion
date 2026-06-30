# Évaluation de l'application — état actuel & axes d'amélioration

> Avis technique et produit sur D&D AI Companion, à un instant T. Honnête et constructif :
> l'objectif est de prioriser la suite, pas de minimiser le travail déjà fait (qui est conséquent).

## En bref

Une base **solide et cohérente** : architecture backend propre (FastAPI + services + schémas + MCP + RAG),
mécaniques D&D 2024 réelles (point-buy, sorts par classe, feuille calculée, historiques/sous-classes),
fonctionnement **100 % local** (respect de la vie privée), thème graphique soigné et homogène.

Les principaux chantiers : **fiabilité des serveurs MCP**, **accessibilité (RGAA)**, **outils orientés joueur**
(dés liés à la fiche, suivi de combat) et, à terme, le **multijoueur temps réel**.

---

## 1. UI / UX

**Points forts**
- Identité visuelle « Grimoire » forte et cohérente (parchemin/or/braise), composants réutilisés.
- Hiérarchie de l'accueil clarifiée (consultation vs création, séparateur).
- Feuille de personnage riche et lisible.

**À améliorer**
- **Lisibilité** : beaucoup de libellés en 8,5–11 px, majuscules très espacées, en `goldDim`/`mistDim` sur fond
  sombre → fatigants et parfois sous le seuil de contraste. Remonter les tailles minimales (≥ 12–14 px) et
  les contrastes.
- **Responsive** : largeurs fixes (rail 96 px, panneaux 270–330 px, modales) et `height: 100vh` ne tiennent pas
  sur mobile/petits écrans. Prévoir des points de rupture et un rail repliable.
- **Retours utilisateur** : certaines erreurs sont avalées (`catch {}`), confirmations via `window.confirm`.
  Ajouter un système de **toasts** (succès/erreur) et des états de chargement homogènes.
- **Styles inline** partout : difficile à maintenir/àthémer. Migrer vers des tokens de design + classes
  (CSS modules, Tailwind, ou variables CSS) faciliterait le thème clair/sombre et l'accessibilité.
- **Menus personnalisés** (sélecteur de campagne, ajouts rapides) : gérer le clic-extérieur, l'échap, le focus.

## 2. Accessibilité (RGAA / WCAG)

C'est l'axe le plus en retard aujourd'hui ; quelques chantiers concrets :
- **Contrastes** (RGAA 3.2/3.3) : vérifier or `#c9a24b` et gris sur fond sombre, surtout aux petites tailles.
  Viser un ratio ≥ 4,5:1 pour le texte courant.
- **Libellés de formulaire** (RGAA 11) : aujourd'hui beaucoup de champs n'ont qu'un `placeholder`. Ajouter de
  vrais `<label for>` associés (le placeholder n'est pas un libellé accessible).
- **Boutons-icônes** (RGAA 1/10) : ajouter `aria-label` partout où il n'y a qu'une icône.
- **Navigation clavier & focus** : assurer un `:focus-visible` net, un **piège de focus** dans les modales
  (`role="dialog"`, `aria-modal`), et la gestion des flèches/échap dans les menus déroulants.
- **Chat en streaming** : encapsuler la réponse en cours dans une zone `aria-live="polite"` pour que les
  lecteurs d'écran annoncent le texte qui arrive.
- **Mouvement** : respecter `prefers-reduced-motion` (overlay du dé, pulsations).
- **Sémantique** : `aria-hidden` sur les ornements décoratifs, hiérarchie de titres `h1>h2>h3` cohérente,
  `lang="fr"` sur la racine.

## 3. User flow

- **Onboarding** : un nouvel utilisateur arrive sur 7 portails sans guidage. Proposer un parcours premier
  lancement : *créer un personnage → forger une campagne → entrer à la Table*.
- **Rôles MD / Joueur** : la distinction structure tout mais n'est jamais expliquée à l'écran.
- **Modèle de campagne active** : une seule campagne active + limite « une campagne » côté MCP rendent l'état
  fragile (cf. bug de la campagne « fantôme »). Afficher clairement la campagne courante et son état, et
  fiabiliser la synchro app⇄MCP.
- **Liens perso ⇄ campagne** : le recrutement d'un personnage existant est peu visible.
- **Pannes** : quand le LLM ou un serveur MCP est indisponible, l'utilisateur doit le comprendre (bandeau
  d'état « Oracle / Outils » plutôt qu'un échec silencieux).

## 4. Outils pour un joueur de D&D

Aujourd'hui l'app est surtout taillée pour le **Maître du Jeu** (campagnes, PNJ, quêtes, forge, bibliothèque).
Côté **joueur**, les manques les plus utiles :
- **Dés liés à la fiche** : lancer une sauvegarde, une compétence ou une attaque **depuis la fiche** (avec le bon
  modificateur), plutôt qu'un lancer générique.
- **Suivi de combat** : PV en direct, **emplacements de sorts** consommés, **conditions** (empoisonné, à terre…),
  **initiative partagée** entre MD et joueurs.
- **Montée de niveau guidée** : aujourd'hui changer le niveau recalcule la maîtrise mais n'accompagne pas PV,
  améliorations de carac/dons, nouveaux sorts.
- **Inventaire avancé** : encombrement, et surtout **harmonisation** (max 3 objets) pour les objets magiques.
- **Assistant de règles pour le joueur** : le Grimoire (RAG) est pensé côté MD ; un accès joueur serait utile.
- **Cohérence des données** : la fiche suit **2024**, mais l'API de règles (dnd5eapi) est en **2014** — écart à
  signaler/gérer.
- **Multijoueur réel** : aujourd'hui chaque compte voit sa vue ; il n'y a pas de **table partagée en temps réel**
  (WebSockets, état de campagne synchronisé). C'est le plus gros chantier pour du jeu « à plusieurs » véritable.

---

## Priorisation suggérée

1. **Fiabiliser MCP/LLM** + indicateurs d'état clairs *(en cours)*.
2. **Passe d'accessibilité** (labels, contrastes, focus, `aria-live` du chat) → conformité RGAA.
3. **Dés intégrés à la fiche** (sauvegardes/compétences/attaques) — fort impact joueur, effort modéré.
4. **Suivi de combat** : initiative + PV + emplacements de sorts + conditions.
5. **Responsive / mobile**.
6. **Multijoueur temps réel** (chantier long terme).
