/** Panneaux lateraux : parchemins, sceaux MCP, grimoire de campagne, choix du modele. */
import { Scroll, Gem, Shield, Moon, Sparkles } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { SectionLabel } from "./ornaments.jsx";
import { toolLabel } from "../lib/toolLabels.js";

export function ParchPanel({ title, children }) {
  return (
    <div style={{
      position: "relative", padding: "20px 18px",
      background: `linear-gradient(180deg, ${T.parch}, ${T.parch2})`,
      borderRadius: 3, color: T.inkOnParch,
      boxShadow: "inset 0 0 40px rgba(120,90,40,.25), 0 10px 30px rgba(0,0,0,.5)",
    }}>
      <div style={{ fontFamily: ORNATE, fontSize: 15, marginBottom: 12, color: "#4A3a22" }}>{title}</div>
      {children}
    </div>
  );
}

export function SourceLine({ name, meta, lit }) {
  return (
    <div style={{ padding: "9px 0", borderTop: "1px solid rgba(80,60,30,.25)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: BODY, fontSize: 16, fontWeight: 600, color: T.inkOnParch }}>
        <Scroll size={14} strokeWidth={1.5} color={lit ? T.ember : "#6b5836"} /> {name}
      </div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: "#6b5836", marginLeft: 22 }}>{meta}</div>
    </div>
  );
}

/** Liste des "sceaux" = outils MCP exposes par le serveur. */
export function SigilPanel({ sigils }) {
  return (
    <div style={{ padding: "16px 18px", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3 }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold, marginBottom: 10 }}>Sceaux du serveur</div>
      {sigils.map((s) => (
        <div key={s} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0", fontFamily: BODY, fontSize: 15, color: T.mist }}>
          <Gem size={11} color={T.goldDim} strokeWidth={1.4} /> {toolLabel(s)}
        </div>
      ))}
    </div>
  );
}

/** Selecteur de modele Ollama, stylise au theme. */
export function ModelSelect({ models, value, onChange }) {
  return (
    <div style={{ padding: "14px 18px", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3, marginTop: 18 }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold, marginBottom: 8 }}>Esprit invoque</div>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        style={{ width: "100%", background: T.ink, color: T.parch, border: `1px solid ${T.line}`,
          borderRadius: 3, padding: "8px 10px", fontFamily: BODY, fontSize: 15, outline: "none", cursor: "pointer" }}>
        {(models.length ? models : [value]).map((m) => <option key={m} value={m}>{m}</option>)}
      </select>
    </div>
  );
}

/* ---------------- Grimoire de campagne (cote La Table) ------------------- */

/**
 * Grimoire de campagne, alimente par GET /api/gamemaster/state.
 * @param {object} campaign - CampaignState (active, companions, quests, counts, omen...).
 */
export function Ledger({ campaign }) {
  if (!campaign?.active) {
    return (
      <div style={{ padding: "18px 16px", background: T.panel, border: `1px dashed ${T.line}`, borderRadius: 3,
        fontFamily: BODY, fontStyle: "italic", fontSize: 15.5, color: T.mistDim, textAlign: "center" }}>
        <Sparkles size={18} color={T.goldDim} strokeWidth={1.3} />
        <div style={{ marginTop: 8 }}>
          Aucune campagne active. Demandez au Maitre d'ouvrir une Session Zero,
          et ce grimoire se remplira de vos compagnons et de vos quetes.
        </div>
      </div>
    );
  }

  const c = campaign.counts || {};
  const stats = [
    [c.companions ?? campaign.companions.length, "Compagnons"],
    [c.npcs ?? 0, "Ames rencontrees"],
    [c.quests ?? campaign.quests.length, "Quetes"],
    [c.sessions ?? 0, "Veillees"],
  ];

  return (
    <>
      <div style={{ fontFamily: ORNATE, fontSize: 18, color: T.parch, marginBottom: 4 }}>{campaign.name}</div>
      {campaign.description && (
        <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14.5, color: T.mistDim, marginBottom: 14 }}>
          {campaign.description}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 18 }}>
        {stats.map(([n, l]) => (
          <div key={l} style={{ padding: "12px 10px", textAlign: "center", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3 }}>
            <div style={{ fontFamily: ORNATE, fontSize: 22, color: T.gold }}>{n}</div>
            <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.mistDim }}>{l}</div>
          </div>
        ))}
      </div>

      <SectionLabel icon={Shield} text="Compagnons" />
      {campaign.companions.length === 0
        ? <Empty text="Nul aventurier n'a encore rejoint la table." />
        : campaign.companions.map((h) => <HeroCard key={h.name} name={h.name} detail={h.detail} level={h.level} />)}

      <SectionLabel icon={Scroll} text="Serments & Quetes" />
      {campaign.quests.length === 0
        ? <Empty text="Aucun serment n'a encore ete prononce." />
        : campaign.quests.map((q) => <QuestRow key={q.title} name={q.title} status={q.status} />)}

      {campaign.omen && (
        <>
          <SectionLabel icon={Moon} text="Presages" />
          <div style={{ padding: "12px 14px", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3, fontFamily: BODY, fontStyle: "italic", fontSize: 15.5, color: T.mist }}>
            « {campaign.omen} »
          </div>
        </>
      )}
    </>
  );
}

function Empty({ text }) {
  return (
    <div style={{ padding: "10px 12px", fontFamily: BODY, fontStyle: "italic", fontSize: 14.5, color: T.mistDim }}>
      {text}
    </div>
  );
}

const ACTIVE_STATUSES = ["active", "en cours", "in_progress", "ongoing"];

function HeroCard({ name, detail, level }) {
  return (
    <div className="lift" style={{
      position: "relative", padding: "14px 16px", marginBottom: 10, borderRadius: 3,
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontFamily: ORNATE, fontSize: 17, color: T.parch }}>{name}</span>
        {level != null && <span style={{ fontFamily: DISPLAY, fontSize: 11, color: T.gold }}>Niv. {level}</span>}
      </div>
      {detail && <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim, marginTop: 2 }}>{detail}</div>}
    </div>
  );
}

function QuestRow({ name, status }) {
  const lit = ACTIVE_STATUSES.includes((status || "").toLowerCase());
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "11px 14px", marginBottom: 8, background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3 }}>
      <span style={{ fontFamily: BODY, fontSize: 16.5, color: T.parch }}>{name}</span>
      {status && <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", padding: "3px 9px", borderRadius: 2,
        color: lit ? T.goldBright : T.mistDim, border: `1px solid ${lit ? T.goldDim : T.line}` }}>{status}</span>}
    </div>
  );
}
