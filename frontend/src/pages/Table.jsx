/** "La Table" — tableau de bord de la campagne en cours : PJ, PNJ/lieux actifs,
 *  quetes, attributions + outils de creation. (Le conteur IA vit desormais en Solo.) */
import { useEffect, useState } from "react";
import { Users, Scroll, MapPin, ChevronDown, BookOpenCheck, UserPlus, Shield, PanelRightOpen, PanelRightClose } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { useViewport } from "../hooks/useViewport.js";
import { useCampaign } from "../hooks/useCampaign.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";
import { listCampaigns, activateCampaign } from "../api/campaigns.js";
import { activateElement, deactivateElement } from "../api/library.js";
import { listCharacters } from "../api/characters.js";
import { Divider } from "../components/ornaments.jsx";
import { Ledger } from "../components/panels.jsx";
import { CampaignTools } from "../components/CampaignTools.jsx";
import { TableEmpty } from "../components/TableEmpty.jsx";
import { Modal } from "../components/Modal.jsx";
import { NpcForm, QuestForm, LocationWizard } from "../components/worldForms.jsx";
import { QuickReference } from "../components/QuickReference.jsx";

export function Table({ setView }) {
  const { active, setActive } = useActiveCampaign();
  const role = active?.role || "dm";
  const { campaign, refresh } = useCampaign();
  const { isNarrow } = useViewport();

  const [modal, setModal] = useState(null);
  const [showLedger, setShowLedger] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);
  const [resuming, setResuming] = useState(!active?.id);

  // Sans campagne active : reprend automatiquement la plus recente.
  useEffect(() => {
    if (active?.id) { setResuming(false); return undefined; }
    let alive = true;
    listCampaigns().then(async (list) => {
      if (!alive) return;
      if (list.length) {
        try { const sm = await activateCampaign(list[0].id); if (alive) setActive(sm); }
        catch { if (alive) setActive(list[0]); }
      }
    }).finally(() => { if (alive) setResuming(false); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const afterCreate = () => { setModal(null); setReloadKey((k) => k + 1); refresh(); };
  const afterChange = () => { setReloadKey((k) => k + 1); refresh(); };

  if (!active?.id) {
    if (resuming) return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: DISPLAY, letterSpacing: 3, textTransform: "uppercase", color: T.mistDim }}>
        Reprise de la derniere campagne...
      </div>
    );
    return <TableEmpty setView={setView} />;
  }
  const isDm = role === "dm";

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <CampaignHeader active={active} setActive={setActive} />

      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: `0 ${isNarrow ? 16 : 48}px`, marginBottom: 6, flexWrap: "wrap" }}>
        {isDm && <>
          <ActionBtn icon={Users} label="Creer un PNJ" onClick={() => setModal("npc")} />
          <ActionBtn icon={Scroll} label="Creer une quete" onClick={() => setModal("quest")} />
          <ActionBtn icon={MapPin} label="Creer un lieu" onClick={() => setModal("location")} />
          <ActionBtn icon={UserPlus} label="Ajouter un PJ" onClick={() => setModal("addpj")} />
        </>}
        <div style={{ flex: 1 }} />
        <ActionBtn icon={showLedger ? PanelRightClose : PanelRightOpen} label={showLedger ? "Masquer" : "Grimoire"} onClick={() => setShowLedger((v) => !v)} subtle />
        <ActionBtn icon={BookOpenCheck} label="Reference" onClick={() => setModal("ref")} subtle />
      </div>
      <div style={{ padding: `0 ${isNarrow ? 16 : 48}px` }}><Divider label={isDm ? "Le Maitre du Jeu" : "A la table"} /></div>

      <div style={{ flex: 1, overflowY: "auto", padding: `14px ${isNarrow ? 16 : 48}px 32px` }}>
        <div style={{ display: "grid", gridTemplateColumns: (!isNarrow && showLedger) ? "minmax(0, 1fr) 330px" : "1fr", gap: 26, maxWidth: 1120, margin: "0 auto", alignItems: "start" }}>
          <div>
            <Players campaignId={active.id} reloadKey={reloadKey} />
            <div style={{ height: 18 }} />
            <CampaignTools campaignId={active.id} role={role} reloadKey={reloadKey} onChanged={afterChange} />
          </div>
          {showLedger && (
            <aside style={{ position: isNarrow ? "static" : "sticky", top: 0 }}>
              <Ledger campaign={campaign} />
            </aside>
          )}
        </div>
      </div>

      {modal === "npc" && <Modal title="Creer un PNJ" onClose={() => setModal(null)}><NpcForm campaignId={active.id} onDone={afterCreate} /></Modal>}
      {modal === "quest" && <Modal title="Lancer une quete" onClose={() => setModal(null)}><QuestForm campaignId={active.id} onDone={afterCreate} /></Modal>}
      {modal === "location" && <Modal title="Creer un lieu" onClose={() => setModal(null)} width={600}><LocationWizard campaignId={active.id} onDone={afterCreate} /></Modal>}
      {modal === "ref" && <Modal title="Reference rapide" onClose={() => setModal(null)} width={640}><QuickReference /></Modal>}
      {modal === "addpj" && <Modal title="Personnages de l'aventure" onClose={() => setModal(null)} width={520}><AddPlayers campaignId={active.id} onChanged={afterChange} /></Modal>}
    </div>
  );
}

/** Personnages joueurs participant a la campagne en cours. */
function Players({ campaignId, reloadKey }) {
  const [chars, setChars] = useState([]);
  useEffect(() => {
    listCharacters().then((cs) => setChars(cs.filter((c) =>
      c.campaign_id === campaignId || (c.campaigns || []).includes(campaignId)))).catch(() => {});
  }, [campaignId, reloadKey]);
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.line}`, borderRadius: 4, padding: "12px 14px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <Shield size={15} color={T.gold} strokeWidth={1.4} />
        <span style={{ fontFamily: DISPLAY, fontSize: 10.5, letterSpacing: 2, textTransform: "uppercase", color: T.gold }}>
          Personnages de l'aventure ({chars.length})
        </span>
      </div>
      {chars.length === 0 ? (
        <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: T.mistDim }}>
          Aucun personnage. Utilise « Ajouter un PJ » pour en faire entrer.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 8 }}>
          {chars.map((c) => (
            <div key={c.id} style={{ padding: "8px 11px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}` }}>
              <div style={{ fontFamily: ORNATE, fontSize: 15.5, color: T.parch }}>{c.name}</div>
              <div style={{ fontFamily: BODY, fontSize: 12.5, color: T.mistDim }}>{c.race} {c.character_class} · niv. {c.class_level}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CampaignHeader({ active, setActive }) {
  const [open, setOpen] = useState(false);
  const [campaigns, setCampaigns] = useState([]);
  useEffect(() => { if (open && campaigns.length === 0) listCampaigns().then(setCampaigns).catch(() => {}); }, [open]);
  const choose = async (c) => {
    setOpen(false);
    if (c.id === active?.id) return;
    try { const s = await activateCampaign(c.id); setActive(s); } catch { setActive(c); }
  };
  return (
    <div style={{ textAlign: "center", padding: "12px 0 2px", position: "relative" }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 5, textTransform: "uppercase", color: T.gold }}>La Table du Maitre</div>
      <button onClick={() => setOpen((v) => !v)} style={{
        display: "inline-flex", alignItems: "center", gap: 8, margin: "4px auto 0", padding: "3px 12px",
        background: "transparent", border: `1px solid ${T.line}`, borderRadius: 4, cursor: "pointer",
        fontFamily: ORNATE, fontSize: 24, color: T.parch }}>
        {active?.name || "Campagne"} <ChevronDown size={18} color={T.gold} />
      </button>
      {open && (
        <div style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", top: "100%", zIndex: 30,
          minWidth: 260, background: T.panel2, border: `1px solid ${T.line}`, borderRadius: 5, padding: 6,
          boxShadow: "0 14px 34px rgba(0,0,0,.5)" }}>
          {campaigns.length === 0 ? <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14, color: T.mistDim, padding: 8 }}>Chargement…</div>
            : campaigns.map((c) => (
              <button key={c.id} onClick={() => choose(c)} style={{
                width: "100%", textAlign: "left", padding: "9px 12px", borderRadius: 3, cursor: "pointer",
                background: c.id === active?.id ? "rgba(201,162,75,.12)" : "transparent", border: "none",
                color: c.id === active?.id ? T.goldBright : T.mist, fontFamily: BODY, fontSize: 15.5 }}>
                {c.id === active?.id ? "✦ " : ""}{c.name} <span style={{ color: T.mistDim, fontSize: 12 }}>({c.role === "dm" ? "Maitre" : "Joueur"})</span>
              </button>
            ))}
        </div>
      )}
    </div>
  );
}

function AddPlayers({ campaignId, onChanged }) {
  const [chars, setChars] = useState([]);
  const [msg, setMsg] = useState(null);
  const load = () => listCharacters().then(setChars).catch(() => {});
  useEffect(() => { load(); }, []);
  const toggle = async (c, inIt) => {
    setMsg(null);
    try {
      if (inIt) await deactivateElement({ element_type: "character", element_id: c.id, campaign_id: campaignId });
      else await activateElement({ element_type: "character", element_id: c.id, campaign_id: campaignId });
      setMsg(c.name + (inIt ? " a quitte l'aventure." : " rejoint l'aventure."));
      load(); onChanged && onChanged();
    } catch (e) { setMsg(e.message); }
  };
  if (chars.length === 0) return <div style={{ fontFamily: BODY, fontStyle: "italic", color: T.mistDim }}>Aucun personnage. Cree-en un d'abord.</div>;
  return (
    <div>
      <div style={{ fontFamily: BODY, fontSize: 14, color: T.mistDim, marginBottom: 8 }}>Fais entrer ou sortir tes personnages de cette campagne.</div>
      {chars.map((c) => {
        const inIt = (c.campaigns || []).includes(campaignId);
        return (
          <div key={c.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 0", borderBottom: `1px solid ${T.line}` }}>
            <span style={{ fontFamily: BODY, fontSize: 15, color: T.parch }}>{c.name} <span style={{ color: T.mistDim, fontSize: 13 }}>({c.race} {c.character_class} niv. {c.class_level})</span></span>
            <button onClick={() => toggle(c, inIt)} style={{ padding: "6px 13px", borderRadius: 3, cursor: "pointer",
              background: inIt ? "rgba(201,162,75,.12)" : "transparent", border: `1px solid ${inIt ? T.gold : T.line}`,
              color: inIt ? T.gold : T.mistDim, fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase" }}>
              {inIt ? "Retirer" : "Ajouter"}
            </button>
          </div>
        );
      })}
      {msg && <div style={{ fontFamily: BODY, fontSize: 13, color: T.gold, marginTop: 8 }}>{msg}</div>}
    </div>
  );
}

function ActionBtn({ icon: Icon, label, onClick, subtle }) {
  return (
    <button onClick={onClick} className="glow" style={{
      display: "inline-flex", alignItems: "center", gap: 7, padding: "8px 14px", borderRadius: 3, cursor: "pointer",
      background: "transparent", border: `1px solid ${subtle ? T.line : T.gold}`, color: subtle ? T.mistDim : T.gold,
      fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase",
    }}>
      <Icon size={14} /> {label}
    </button>
  );
}
