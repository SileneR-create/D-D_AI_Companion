/** Ecran d'accueil : le Seuil. Trois cartes de tarot — recto = pilier, verso = liens/outils. */
import { useState } from "react";
import { BookOpen, Swords, Drama, UserPlus, Sparkles, Castle, Sword, Hammer, ChevronRight, RotateCcw } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Crest } from "../components/atmosphere.jsx";
import { Filigree } from "../components/ornaments.jsx";

const CARDS = [
  {
    key: "grimoire", icon: BookOpen, numeral: "I", title: "Le Grimoire", tagline: "Le savoir",
    blurb: "Regles, sorts, monstres et oracle a portee de main.",
    links: [
      { view: "rules", icon: BookOpen, label: "Consulter les regles", sub: "Recherche SRD & oracle" },
    ],
  },
  {
    key: "md", icon: Swords, numeral: "II", title: "Le Maitre", tagline: "Mener une table",
    blurb: "Tes campagnes, ton monde et tous tes outils de jeu.",
    links: [
      { view: "gm", icon: Swords, label: "La Table", sub: "Tableau de bord de campagne" },
      { view: "antre", icon: Castle, label: "L'Antre", sub: "Persos, campagnes, PNJ, lieux" },
      { view: "arsenal", icon: Sword, label: "L'Arsenal", sub: "Objets & equipement" },
      { view: "forge", icon: Hammer, label: "La Forge", sub: "Armes & objets magiques" },
      { view: "create", icon: Sparkles, label: "Forger une campagne", sub: "Lancer une Session Zero" },
      { view: "character", icon: UserPlus, label: "Creer un personnage", sub: "Forger un heros" },
    ],
  },
  {
    key: "solo", icon: Drama, numeral: "III", title: "Le Solo", tagline: "Jouer seul",
    blurb: "Le narrateur tisse une aventure rien que pour toi.",
    links: [
      { view: "solo", icon: Drama, label: "Aventure en solo", sub: "Genre, duree, heros -> histoire" },
    ],
  },
];

export function Threshold({ setView }) {
  const [flipped, setFlipped] = useState(null);
  return (
    <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", alignItems: "center", padding: "38px 32px 56px" }}>
      <Crest size={92} />
      <div style={{ fontFamily: DISPLAY, fontSize: 11.5, letterSpacing: 7, textTransform: "uppercase", color: T.gold, marginTop: 16 }}>
        Compagnon des Royaumes
      </div>
      <h1 style={{ fontFamily: ORNATE, fontSize: 40, color: T.parch, margin: "8px 0 4px", letterSpacing: 1 }}>
        D&amp;D AI Companion
      </h1>
      <p style={{ fontFamily: BODY, fontSize: 16.5, fontStyle: "italic", color: T.mist, maxWidth: 520, textAlign: "center", marginBottom: 26 }}>
        Trois cartes, trois voies. Retourne-en une pour decouvrir ses outils.
      </p>

      <div style={{ display: "flex", gap: 26, flexWrap: "wrap", justifyContent: "center" }}>
        {CARDS.map((c) => (
          <TarotCard key={c.key} card={c} flipped={flipped === c.key}
            onFlip={() => setFlipped((f) => (f === c.key ? null : c.key))} setView={setView} />
        ))}
      </div>
    </div>
  );
}

function TarotCard({ card, flipped, onFlip, setView }) {
  const Icon = card.icon;
  return (
    <div style={{ width: 252, height: 412, perspective: 1500 }}>
      <div style={{
        position: "relative", width: "100%", height: "100%", transformStyle: "preserve-3d",
        transition: "transform .65s cubic-bezier(.2,.7,.2,1)", transform: flipped ? "rotateY(180deg)" : "none",
      }}>
        {/* RECTO */}
        <button onClick={onFlip} className="lift" style={{ ...face, ...recto, cursor: "pointer" }}>
          <span style={{ position: "absolute", top: 10, left: 10 }}><Filigree /></span>
          <span style={{ position: "absolute", top: 10, right: 10 }}><Filigree flip /></span>
          <span style={{ position: "absolute", bottom: 10, left: 10, transform: "rotate(180deg)" }}><Filigree /></span>
          <span style={{ position: "absolute", bottom: 10, right: 10, transform: "rotate(180deg)" }}><Filigree flip /></span>
          <div style={{ fontFamily: ORNATE, fontSize: 22, color: T.goldDim }}>{card.numeral}</div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14 }}>
            <div style={{ width: 84, height: 84, borderRadius: "50%", border: `1px solid ${T.goldDim}`, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(201,162,75,.06)" }}>
              <Icon size={40} strokeWidth={1.1} color={T.gold} />
            </div>
            <div style={{ fontFamily: ORNATE, fontSize: 27, color: T.parch }}>{card.title}</div>
            <div style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 3, textTransform: "uppercase", color: T.gold }}>{card.tagline}</div>
            <p style={{ fontFamily: BODY, fontSize: 14, fontStyle: "italic", color: T.mistDim, textAlign: "center", margin: "2px 6px 0", lineHeight: 1.5 }}>{card.blurb}</p>
          </div>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: DISPLAY, fontSize: 9, letterSpacing: 2, textTransform: "uppercase", color: T.gold }}>
            <RotateCcw size={12} /> Retourner la carte
          </div>
        </button>

        {/* VERSO */}
        <div style={{ ...face, ...verso, transform: "rotateY(180deg)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
            <Icon size={17} color={T.gold} strokeWidth={1.3} />
            <span style={{ fontFamily: ORNATE, fontSize: 19, color: T.parch }}>{card.title}</span>
          </div>
          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 7 }}>
            {card.links.map((l) => {
              const LIcon = l.icon;
              return (
                <button key={l.view} onClick={() => setView(l.view)} className="glow" style={linkRow}>
                  <LIcon size={16} color={T.gold} strokeWidth={1.4} />
                  <div style={{ flex: 1, textAlign: "left", minWidth: 0 }}>
                    <div style={{ fontFamily: BODY, fontSize: 14.5, color: T.parch }}>{l.label}</div>
                    <div style={{ fontFamily: BODY, fontSize: 11.5, color: T.mistDim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.sub}</div>
                  </div>
                  <ChevronRight size={14} color={T.gold} />
                </button>
              );
            })}
          </div>
          <button onClick={onFlip} style={{ marginTop: 10, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 6,
            background: "transparent", border: `1px solid ${T.line}`, borderRadius: 4, padding: "7px 0", cursor: "pointer",
            color: T.mistDim, fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase" }}>
            <RotateCcw size={11} /> Retourner
          </button>
        </div>
      </div>
    </div>
  );
}

const face = {
  position: "absolute", inset: 0, WebkitBackfaceVisibility: "hidden", backfaceVisibility: "hidden",
  borderRadius: 12, border: `1px solid ${T.goldDim}`, padding: "16px 16px", boxSizing: "border-box",
  display: "flex", flexDirection: "column", alignItems: "center", overflow: "hidden",
  boxShadow: "0 18px 44px rgba(0,0,0,.5)",
};
const recto = { background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, textAlign: "center" };
const verso = { background: `linear-gradient(180deg, ${T.panel}, ${T.ink})`, alignItems: "stretch" };
const linkRow = {
  display: "flex", alignItems: "center", gap: 10, padding: "9px 11px", borderRadius: 4, cursor: "pointer",
  background: "rgba(201,162,75,.05)", border: `1px solid ${T.line}`, color: T.mist,
};
