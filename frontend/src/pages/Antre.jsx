/** L'Antre : tout ce que l'utilisateur possède — campagnes, personnages, PNJ, lieux —
 *  réuni en une seule page avec des filtres à plat (fusion de Mon antre + Archives). */
import { useState } from "react";
import { T, DISPLAY } from "../theme.js";
import { useViewport } from "../hooks/useViewport.js";
import { ScreenTitle, Divider } from "../components/ornaments.jsx";
import { Library } from "./Library.jsx";
import { Bibliotheque } from "./Bibliotheque.jsx";

const FILTERS = [
  ["tout", "Tout"], ["campagnes", "Campagnes"], ["personnages", "Personnages"], ["npc", "PNJ"], ["location", "Lieux"],
];

export function Antre({ setView }) {
  const { isNarrow } = useViewport();
  const px = isNarrow ? 16 : 48;
  const [filter, setFilter] = useState("tout");
  const showLib = filter === "tout" || filter === "campagnes" || filter === "personnages";
  const showBib = filter === "tout" || filter === "npc" || filter === "location";
  const libOnly = filter === "campagnes" ? "campagnes" : filter === "personnages" ? "personnages" : undefined;
  const bibOnly = filter === "npc" ? "npc" : filter === "location" ? "location" : undefined;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Vos creations & votre monde" title="L'Antre" />
      <div style={{ padding: `0 ${px}px` }}><Divider /></div>

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 10, flexWrap: "wrap" }}>
        {FILTERS.map(([k, l]) => (
          <button key={k} onClick={() => setFilter(k)} style={{
            padding: "7px 16px", borderRadius: 3, cursor: "pointer",
            background: filter === k ? "rgba(201,162,75,.12)" : "transparent",
            border: `1px solid ${filter === k ? T.gold : T.line}`, color: filter === k ? T.gold : T.mistDim,
            fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase",
          }}>{l}</button>
        ))}
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: `8px ${px}px 32px`, maxWidth: 920, margin: "0 auto", width: "100%" }}>
        {showLib && <Library setView={setView} embedded only={libOnly} />}
        {showLib && showBib && <div style={{ height: 24 }} />}
        {showBib && <Bibliotheque embedded only={bibOnly} />}
      </div>
    </div>
  );
}
