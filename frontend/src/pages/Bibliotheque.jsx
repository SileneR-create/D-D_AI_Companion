/** Les Archives : PNJ et lieux crees, avec ajout et activation par campagne.
 *  (Les objets/tresors sont desormais dans L'Arsenal.) */
import { useEffect, useState } from "react";
import { Users, MapPin, Plus } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { ScreenTitle, Divider } from "../components/ornaments.jsx";
import { Modal } from "../components/Modal.jsx";
import { NpcForm, LocationWizard } from "../components/worldForms.jsx";
import { getLibrary, activateElement, deactivateElement } from "../api/library.js";

const FILTERS = [["all", "Tout"], ["npc", "PNJ"], ["location", "Lieux"]];

export function Bibliotheque({ embedded = false, only }) {
  const [lib, setLib] = useState({ campaigns: [], npcs: [], locations: [], items: [] });
  const [filter, setFilter] = useState("all");
  const [modal, setModal] = useState(null); // "npc" | "location" | null

  const refresh = () => getLibrary().then(setLib).catch(() => {});
  useEffect(() => { refresh(); }, []);
  const closeModal = () => { setModal(null); refresh(); };

  const toggle = async (element_type, element_id, campaign_id, active) => {
    try {
      if (active) await deactivateElement({ element_type, element_id, campaign_id });
      else await activateElement({ element_type, element_id, campaign_id });
      refresh();
    } catch { /* ignore */ }
  };

  const showNpc = only ? only === "npc" : (filter === "all" || filter === "npc");
  const showLoc = only ? only === "location" : (filter === "all" || filter === "location");

  const body = (
    <>
      {showNpc && (
        <Section icon={Users} title={`PNJ (${lib.npcs.length})`}>
          <AddButton label="Ajouter un PNJ" onClick={() => setModal("npc")} />
          {lib.npcs.map((n) => (
            <ElementCard key={n.id} el={n} campaigns={lib.campaigns}
              onToggle={(cid, active) => toggle("npc", n.id, cid, active)} />
          ))}
        </Section>
      )}
      {showLoc && (
        <Section icon={MapPin} title={`Lieux (${lib.locations.length})`}>
          <AddButton label="Ajouter un lieu" onClick={() => setModal("location")} />
          {lib.locations.map((l) => (
            <ElementCard key={l.id} el={l} campaigns={lib.campaigns}
              onToggle={(cid, active) => toggle("location", l.id, cid, active)} />
          ))}
        </Section>
      )}
    </>
  );

  const modals = (
    <>
      {/* Memes formulaires que La Table : sans campagne, l'element est cree dans les Archives. */}
      {modal === "npc" && (
        <Modal title="Nouveau PNJ" onClose={() => setModal(null)}>
          <NpcForm campaignId={null} onDone={closeModal} />
        </Modal>
      )}
      {modal === "location" && (
        <Modal title="Nouveau lieu" onClose={() => setModal(null)}>
          <LocationWizard campaignId={null} onDone={closeModal} />
        </Modal>
      )}
    </>
  );

  if (embedded) return (<>{body}{modals}</>);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="PNJ & lieux" title="Les Archives" />
      <div style={{ padding: "0 48px" }}><Divider /></div>

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

      <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px", maxWidth: 920, margin: "0 auto", width: "100%" }}>
        {body}
      </div>
      {modals}
    </div>
  );
}

/** Carte-bouton (pointilles) ouvrant la modale de creation. */
function AddButton({ label, onClick }) {
  return (
    <button onClick={onClick} style={{ ...card, cursor: "pointer", textAlign: "left",
      display: "flex", alignItems: "center", gap: 8, color: T.gold, fontFamily: DISPLAY, fontSize: 10,
      letterSpacing: 1.5, textTransform: "uppercase", border: `1px dashed ${T.goldDim}` }}>
      <Plus size={14} /> {label}
    </button>
  );
}

function ElementCard({ el, campaigns, onToggle }) {
  return (
    <div style={card}>
      <div style={{ fontFamily: ORNATE, fontSize: 16, color: T.parch }}>{el.name}</div>
      {el.summary && <Desc>{el.summary}</Desc>}
      {(el.is_adversary || el.armor_class != null || el.hit_points != null) && (
        <div style={{ display: "flex", gap: 10, marginTop: 6, flexWrap: "wrap", alignItems: "center", fontFamily: BODY, fontSize: 13, color: T.mist }}>
          {el.is_adversary && <span style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1, textTransform: "uppercase", color: "#d98a6a", border: "1px solid #5a3a30", borderRadius: 2, padding: "1px 6px" }}>Adversaire</span>}
          {el.armor_class != null && <span>CA <b style={{ color: T.parch }}>{el.armor_class}</b></span>}
          {el.hit_points != null && <span>PV <b style={{ color: T.parch }}>{el.hit_points}</b></span>}
          {el.challenge_rating && <span>FP <b style={{ color: T.parch }}>{el.challenge_rating}</b></span>}
        </div>
      )}
      <div style={{ marginTop: 8 }}>
        <span style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.mistDim }}>Actif dans :</span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
          {campaigns.length === 0 && <span style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13, color: T.mistDim }}>Aucune campagne.</span>}
          {campaigns.map((c) => {
            const active = el.active_in.includes(c.id);
            return (
              <button key={c.id} onClick={() => onToggle(c.id, active)} disabled={c.role !== "dm"} style={{
                padding: "5px 11px", borderRadius: 2, cursor: c.role === "dm" ? "pointer" : "not-allowed",
                background: active ? "rgba(201,162,75,.16)" : "transparent",
                border: `1px solid ${active ? T.gold : T.line}`, color: active ? T.goldBright : T.mistDim,
                fontFamily: BODY, fontSize: 13.5, opacity: c.role === "dm" ? 1 : 0.5,
              }} title={c.role === "dm" ? "" : "Seul le MD peut activer"}>
                {active ? "✦ " : ""}{c.name}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function Section({ icon: Icon, title, children }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "8px 0 12px" }}>
        <Icon size={15} color={T.gold} strokeWidth={1.4} />
        <span style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold }}>{title}</span>
        <span style={{ flex: 1, height: 1, background: T.line }} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>{children}</div>
    </div>
  );
}

const card = { padding: "14px 16px", borderRadius: 5, background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` };
const Desc = ({ children }) => <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: T.mistDim, marginTop: 3 }}>{children}</div>;
const tinp = { width: "100%", padding: "8px 10px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none", boxSizing: "border-box" };
const mini = { padding: "7px 14px", borderRadius: 3, background: "transparent", cursor: "pointer", fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" };
