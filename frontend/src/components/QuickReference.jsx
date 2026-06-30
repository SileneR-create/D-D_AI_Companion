/** Reference rapide (anti-seche) ouverte a La Table : instantanee, hors-ligne. */
import { useState } from "react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { CONDITIONS, ACTIONS, COVER, DCS, RESTS } from "../lib/quickRef.js";

const TABS = [
  ["conditions", "Conditions"],
  ["actions", "Actions"],
  ["cover", "Couvert & DD"],
  ["rest", "Repos"],
];

export function QuickReference() {
  const [tab, setTab] = useState("conditions");
  const [q, setQ] = useState("");
  const filter = (list) => {
    const s = q.trim().toLowerCase();
    return s ? list.filter((e) => e.name.toLowerCase().includes(s) || e.effect.toLowerCase().includes(s)) : list;
  };
  return (
    <div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
        {TABS.map(([k, l]) => (
          <button key={k} onClick={() => setTab(k)} style={{
            padding: "6px 13px", borderRadius: 3, cursor: "pointer",
            background: tab === k ? "rgba(201,162,75,.12)" : "transparent",
            border: `1px solid ${tab === k ? T.gold : T.line}`, color: tab === k ? T.gold : T.mistDim,
            fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase" }}>{l}</button>
        ))}
        <div style={{ flex: 1 }} />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filtrer…" style={{
          padding: "6px 10px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`,
          color: T.parch, fontFamily: BODY, fontSize: 13.5, outline: "none", width: 140 }} />
      </div>

      <div style={{ maxHeight: "60vh", overflowY: "auto", paddingRight: 4 }}>
        {tab === "conditions" && filter(CONDITIONS).map((e) => <Row key={e.name} {...e} />)}
        {tab === "actions" && filter(ACTIONS).map((e) => <Row key={e.name} {...e} />)}
        {tab === "cover" && (
          <>
            <Sub>Couvert</Sub>
            {filter(COVER).map((e) => <Row key={e.name} {...e} />)}
            <Sub>Degres de difficulte (DD)</Sub>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {DCS.map(([l, v]) => (
                <span key={l} style={{ fontFamily: BODY, fontSize: 14, color: T.mist, border: `1px solid ${T.line}`, borderRadius: 3, padding: "5px 10px" }}>
                  {l} <b style={{ color: T.gold }}>{v}</b>
                </span>
              ))}
            </div>
          </>
        )}
        {tab === "rest" && filter(RESTS).map((e) => <Row key={e.name} {...e} />)}
      </div>
    </div>
  );
}

function Row({ name, effect }) {
  return (
    <div style={{ padding: "8px 0", borderBottom: `1px solid ${T.line}` }}>
      <div style={{ fontFamily: ORNATE, fontSize: 15.5, color: T.parch }}>{name}</div>
      <div style={{ fontFamily: BODY, fontSize: 14, color: T.mist, lineHeight: 1.5, marginTop: 2 }}>{effect}</div>
    </div>
  );
}
const Sub = ({ children }) => <div style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim, margin: "10px 0 6px" }}>{children}</div>;
