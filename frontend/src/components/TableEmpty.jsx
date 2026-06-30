/** Ecran d'actions quand aucune campagne n'est active (au lieu du chat). */
import { useEffect, useState } from "react";
import { UserPlus, Scroll, MapPin, Sparkles, ChevronRight } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Crest } from "./atmosphere.jsx";
import { listCampaigns, activateCampaign } from "../api/campaigns.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";

export function TableEmpty({ setView }) {
  const { setActive } = useActiveCampaign();
  const [campaigns, setCampaigns] = useState([]);
  const [pick, setPick] = useState("");
  const [hint, setHint] = useState(null);

  useEffect(() => { listCampaigns().then(setCampaigns).catch(() => {}); }, []);

  const enter = async () => {
    if (!pick) return;
    try { const s = await activateCampaign(Number(pick)); setActive(s); }
    catch { const c = campaigns.find((x) => String(x.id) === pick); if (c) setActive(c); }
  };
  // Quete / Lieu necessitent une campagne active : on en active une si choisie.
  const needCampaign = async () => {
    if (pick) return enter();
    setHint("Choisissez ou creez d'abord une campagne pour gerer quetes et lieux.");
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 40, gap: 8 }}>
      <Crest size={92} />
      <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 5, textTransform: "uppercase", color: T.gold, marginTop: 10 }}>La Table</div>
      <h1 style={{ fontFamily: ORNATE, fontSize: 30, color: T.parch, margin: "4px 0 2px" }}>Que veux-tu entreprendre ?</h1>
      <p style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 16, color: T.mistDim, marginBottom: 18 }}>
        Aucune campagne active. Choisis une action ou reprends une campagne existante.
      </p>

      {campaigns.length > 0 && (
        <div style={{ display: "flex", gap: 8, marginBottom: 22 }}>
          <select value={pick} onChange={(e) => setPick(e.target.value)} style={{
            padding: "9px 12px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`,
            color: T.parch, fontFamily: BODY, fontSize: 15, outline: "none", cursor: "pointer", minWidth: 240 }}>
            <option value="">Reprendre une campagne…</option>
            {campaigns.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.role === "dm" ? "Maitre" : "Joueur"})</option>)}
          </select>
          <button onClick={enter} className="glow" style={cta}>Entrer <ChevronRight size={13} /></button>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, maxWidth: 560 }}>
        <Action icon={Sparkles} title="Lancer une Session Zero" sub="Forger une nouvelle campagne" onClick={() => setView("create")} />
        <Action icon={UserPlus} title="Creer un personnage" sub="Une fiche de heros" onClick={() => setView("character")} />
        <Action icon={Scroll} title="Creer une quete" sub="Necessite une campagne" onClick={needCampaign} />
        <Action icon={MapPin} title="Creer un lieu" sub="Necessite une campagne" onClick={needCampaign} />
      </div>
      {hint && <div style={{ fontFamily: BODY, fontSize: 14, color: T.emberBright, marginTop: 14 }}>{hint}</div>}
    </div>
  );
}

function Action({ icon: Icon, title, sub, onClick }) {
  return (
    <button onClick={onClick} className="lift" style={{
      width: 270, padding: "20px 22px", textAlign: "left", cursor: "pointer", borderRadius: 5,
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}`, color: T.mist,
    }}>
      <Icon size={22} color={T.gold} strokeWidth={1.3} />
      <div style={{ fontFamily: ORNATE, fontSize: 17, color: T.parch, marginTop: 8 }}>{title}</div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14, color: T.mistDim }}>{sub}</div>
    </button>
  );
}

const cta = { display: "inline-flex", alignItems: "center", gap: 6, padding: "9px 18px", borderRadius: 3,
  background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
  fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2, textTransform: "uppercase" };
