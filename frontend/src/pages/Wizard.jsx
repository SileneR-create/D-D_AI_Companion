/**
 * Wizard de creation guidee (MD) : 1) type d'aventure -> 2) personnages -> 3) lancement.
 * Cree la campagne (createur = MD) puis ses personnages (rattaches a la campagne).
 */
import { useEffect, useState } from "react";
import { Scroll, Users, Flame, Check, ChevronRight } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { ScreenTitle, Divider } from "../components/ornaments.jsx";
import { CharacterForm } from "../components/CharacterForm.jsx";
import { Qcm } from "../components/Qcm.jsx";
import { createCampaign, recruitCharacter } from "../api/campaigns.js";
import { createCharacter, listCharacters } from "../api/characters.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";

const STEPS = [
  { id: 1, label: "Aventure", icon: Scroll },
  { id: 2, label: "Personnages", icon: Users },
  { id: 3, label: "Lancement", icon: Flame },
];
const THEMES = ["Heroique", "Sombre & gothique", "Mystere & enquete", "Exploration", "Politique & intrigue", "Horreur"];

export function Wizard({ setView }) {
  const { setActive } = useActiveCampaign();
  const [step, setStep] = useState(1);
  const [campaign, setCampaign] = useState(null);
  const [characters, setCharacters] = useState([]);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Forge de campagne" title="Session Zero" />
      <div style={{ padding: "0 48px" }}><Stepper step={step} /></div>

      <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px" }}>
        {step === 1 && <StepAdventure onDone={(c) => { setCampaign(c); setActive(c); setStep(2); }} />}
        {step === 2 && campaign && (
          <div style={{ maxWidth: 760, margin: "0 auto" }}>
            {characters.length > 0 && (
              <div style={{ marginBottom: 16, padding: "12px 16px", borderRadius: 5, background: T.panel, border: `1px solid ${T.line}` }}>
                <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: T.gold, marginBottom: 6 }}>Compagnie ({characters.length})</div>
                {characters.map((c, i) => (
                  <span key={i} style={{ fontFamily: BODY, fontSize: 15, color: T.parch, marginRight: 14 }}>
                    {c.name} <span style={{ color: T.mistDim }}>({c.race} {c.character_class})</span>
                  </span>
                ))}
              </div>
            )}
            <Recruit campaignId={campaign.id} onRecruited={(c) => setCharacters((x) => [...x, c])} taken={characters} />
            <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: T.mistDim, textAlign: "center", margin: "10px 0" }}>— ou cree un nouveau heros —</div>
            <CharacterForm campaignId={campaign.id} submitLabel="Ajouter ce personnage"
              key={characters.length}
              onSave={async (p) => { const c = await createCharacter(p); setCharacters((x) => [...x, c]); }} />
            <div style={{ textAlign: "center", marginTop: 18 }}>
              <button onClick={() => setStep(3)} className="glow" style={ghost}>
                Poursuivre <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
        {step === 3 && campaign && (
          <div style={{ maxWidth: 640, margin: "0 auto", padding: "24px 26px", borderRadius: 6,
            background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` }}>
            <div style={{ fontFamily: ORNATE, fontSize: 20, color: T.parch, marginBottom: 10 }}>Tout est pret</div>
            <Divider />
            <p style={{ fontFamily: BODY, fontSize: 17, color: T.mist, lineHeight: 1.7 }}>
              La campagne <strong style={{ color: T.parch, fontFamily: ORNATE }}>{campaign.name}</strong> est forgee,
              avec <strong style={{ color: T.gold }}>{characters.length}</strong> heros. Le Maitre du Jeu vous attend a la Table.
            </p>
            <button onClick={() => setView("gm")} className="glow" style={{ ...ghost, marginTop: 8 }}>
              <Flame size={14} /> Entrer a la Table
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

const CAMP_STEPS = [
  { key: "name", q: "Quel est le nom de la campagne ?", type: "text", placeholder: "Les Ombres de Neverwinter" },
  { key: "type", q: "Quel type de campagne ?", type: "single", options: ["Heroique", "Sombre & gothique", "Mystere & enquete", "Exploration", "Politique & intrigue", "Horreur", "Comique"] },
  { key: "players", q: "Combien de joueurs a la creation ?", type: "single", options: ["Solo", "2", "3", "4", "5", "6+"] },
  { key: "duration", q: "Quelle duree prevue ?", type: "single", options: ["One-shot", "Courte", "Longue campagne", "Indeterminee"] },
  { key: "tone", q: "Quel ton dominant ?", type: "single", options: ["Serieux", "Epique", "Leger", "Tragique", "Mature"] },
  { key: "univers", q: "Quel univers / cadre ?", type: "single", options: ["Medieval-fantastique", "Dark fantasy", "Cites & intrigues", "Nature sauvage", "Donjons", "Autre"] },
  { key: "pitch", q: "Le pitch en une phrase ? (facultatif)", type: "text", optional: true, textarea: true, placeholder: "Une menace ancienne s'eveille..." },
];

function composeCampaign(a) {
  const s = [];
  s.push(`Campagne ${(a.type || "").toLowerCase()}${a.duration ? ` (${a.duration.toLowerCase()})` : ""}, pour ${a.players || "?"} joueur(s).`);
  const ud = [a.tone ? `ton ${a.tone.toLowerCase()}` : null, a.univers ? `univers ${a.univers.toLowerCase()}` : null].filter(Boolean).join(", ");
  if (ud) s.push(ud.charAt(0).toUpperCase() + ud.slice(1) + ".");
  if (a.pitch) s.push(a.pitch);
  return s.join(" ");
}

function StepAdventure({ onDone }) {
  return (
    <div style={{ maxWidth: 620, margin: "0 auto", padding: "24px 26px", borderRadius: 6,
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` }}>
      <Qcm steps={CAMP_STEPS} compose={composeCampaign} createLabel="Creer la campagne"
        onCreate={async (a) => { onDone(await createCampaign({ name: a.name, description: composeCampaign(a), setting: a.type })); }} />
    </div>
  );
}

function Recruit({ campaignId, onRecruited, taken }) {
  const [chars, setChars] = useState([]);
  const [pick, setPick] = useState("");
  const [msg, setMsg] = useState(null);
  const takenIds = new Set(taken.map((c) => c.id));
  const refresh = () => listCharacters().then((all) => setChars(all.filter((c) => !c.campaign_id && !takenIds.has(c.id))));
  useEffect(() => { refresh(); }, [taken.length]);
  if (chars.length === 0) return null;
  const recruit = async () => {
    if (!pick) return;
    const c = chars.find((x) => String(x.id) === pick);
    try { await recruitCharacter(campaignId, Number(pick)); onRecruited({ ...c, campaign_id: campaignId }); setPick(""); setMsg(`${c?.name} rejoint la compagnie.`); }
    catch (e) { setMsg(e.message); }
  };
  return (
    <div style={{ marginBottom: 14, padding: "12px 16px", borderRadius: 5, background: T.ink, border: `1px solid ${T.goldDim}` }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: T.gold, marginBottom: 8 }}>Recruter un personnage existant</div>
      <div style={{ display: "flex", gap: 8 }}>
        <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ flex: 1, padding: "9px 12px", borderRadius: 3, background: T.panel, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 15, outline: "none", cursor: "pointer" }}>
          <option value="">Choisir un de tes heros…</option>
          {chars.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.race} {c.character_class} niv. {c.class_level})</option>)}
        </select>
        <button onClick={recruit} className="glow" style={ghost}>Recruter</button>
      </div>
      {msg && <div style={{ fontFamily: BODY, fontSize: 13, color: T.gold, marginTop: 6 }}>{msg}</div>}
    </div>
  );
}

function L({ label, children }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 2, textTransform: "uppercase", color: T.mistDim }}>{label}</label>
      {children}
    </div>
  );
}

function Stepper({ step }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 14, margin: "8px 0 18px" }}>
      {STEPS.map(({ id, label, icon: Icon }, i) => {
        const done = step > id, on = step === id;
        const color = done ? T.goldBright : on ? T.gold : T.mistDim;
        return (
          <div key={id} style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 30, height: 30, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
                border: `1px solid ${color}`, color, background: on ? "rgba(201,162,75,.12)" : "transparent" }}>
                {done ? <Check size={15} /> : <Icon size={15} />}
              </div>
              <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color }}>{label}</span>
            </div>
            {i < STEPS.length - 1 && <span style={{ width: 40, height: 1, background: T.line }} />}
          </div>
        );
      })}
    </div>
  );
}

const ghost = { display: "inline-flex", alignItems: "center", gap: 8, padding: "11px 22px", borderRadius: 3,
  background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
  fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2, textTransform: "uppercase" };
