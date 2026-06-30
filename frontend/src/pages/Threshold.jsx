/** Ecran d'accueil : le Seuil. Consultation en portails uniformes, puis creations. */
import { BookOpen, Swords, UserPlus, Sparkles, Castle, Archive, Hammer, Sword, ChevronRight } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Crest } from "../components/atmosphere.jsx";
import { Filigree } from "../components/ornaments.jsx";

// Consultation / jeu : ce que l'on ouvre au quotidien.
const PORTALS = [
  { view: "rules", icon: BookOpen, title: "Le Grimoire", sub: "Consulter les regles" },
  { view: "gm", icon: Swords, title: "La Table du Maitre", sub: "Mener la campagne" },
  { view: "library", icon: Castle, title: "Mon antre", sub: "Mes persos & campagnes" },
  { view: "library2", icon: Archive, title: "Les Archives", sub: "PNJ & lieux" },
  { view: "arsenal", icon: Sword, title: "L'Arsenal", sub: "Objets, armes & armures" },
];

// Creation : forger de nouveaux elements (regroupes apres le trait).
const CREATIONS = [
  { view: "character", icon: UserPlus, title: "Creer un personnage", sub: "Forger un heros" },
  { view: "create", icon: Sparkles, title: "Forger une campagne", sub: "Lancer une Session Zero" },
  { view: "forge", icon: Hammer, title: "La Forge", sub: "Armes & objets magiques" },
];

export function Threshold({ setView }) {
  return (
    <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", alignItems: "center", padding: "48px 40px" }}>
      <Crest size={120} />
      <div style={{ fontFamily: DISPLAY, fontSize: 12, letterSpacing: 7, textTransform: "uppercase", color: T.gold, marginTop: 22 }}>
        Compagnon des Royaumes
      </div>
      <h1 style={{ fontFamily: ORNATE, fontSize: 46, color: T.parch, margin: "10px 0 4px", letterSpacing: 1 }}>
        D&amp;D AI Companion
      </h1>
      <p style={{ fontFamily: BODY, fontSize: 18, fontStyle: "italic", color: T.mist, maxWidth: 520, textAlign: "center", marginBottom: 36 }}>
        Un oracle pour les regles de la cinquieme edition, et une table de travail
        pour le Maitre qui faconne les mondes.
      </p>

      <div style={{ width: "100%", maxWidth: 780 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 18 }}>
          {PORTALS.map((p) => <Portal key={p.view} {...p} onClick={() => setView(p.view)} />)}
        </div>

        <Separator label="Forger" />

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 18 }}>
          {CREATIONS.map((p) => <Portal key={p.view} {...p} onClick={() => setView(p.view)} />)}
        </div>
      </div>
    </div>
  );
}

/** Long trait orne separant consultation et creation. */
function Separator({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 16, margin: "30px 4px 26px" }}>
      <span style={{ flex: 1, height: 1, background: `linear-gradient(90deg, transparent, ${T.line})` }} />
      <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 4, textTransform: "uppercase", color: T.gold }}>{label}</span>
      <span style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${T.line}, transparent)` }} />
    </div>
  );
}

function Portal({ icon: Icon, title, sub, onClick }) {
  return (
    <button onClick={onClick} className="lift" style={{
      position: "relative", padding: "26px 22px", cursor: "pointer", textAlign: "center",
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`,
      border: `1px solid ${T.line}`, borderRadius: 4, color: T.mist,
    }}>
      <span style={{ position: "absolute", top: 8, left: 8 }}><Filigree /></span>
      <span style={{ position: "absolute", top: 8, right: 8 }}><Filigree flip /></span>
      <Icon size={28} strokeWidth={1.2} color={T.gold} />
      <div style={{ fontFamily: ORNATE, fontSize: 20, color: T.parch, marginTop: 12 }}>{title}</div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim, marginTop: 4 }}>{sub}</div>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 14, fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: T.gold }}>
        Franchir <ChevronRight size={13} />
      </div>
    </button>
  );
}
