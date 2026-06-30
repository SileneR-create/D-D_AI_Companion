/** Ecran "La Table" — chat MD (historique persistant) + actions (modales) + sidebar. */
import { useEffect, useState } from "react";
import { Users, Scroll, MapPin, PanelRightOpen, PanelRightClose, ChevronDown, BookOpenCheck, UserPlus } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { useModels } from "../hooks/useModels.js";
import { useCampaign } from "../hooks/useCampaign.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";
import { listMessages, listCampaigns, activateCampaign } from "../api/campaigns.js";
import { activateElement, deactivateElement } from "../api/library.js";
import { listCharacters } from "../api/characters.js";
import { DOMAINS } from "../api";
import { Divider, ScreenTitle } from "../components/ornaments.jsx";
import { Ledger, ModelSelect } from "../components/panels.jsx";
import { CampaignTools } from "../components/CampaignTools.jsx";
import { TableEmpty } from "../components/TableEmpty.jsx";
import { TableChat } from "../components/TableChat.jsx";
import { Modal } from "../components/Modal.jsx";
import { NpcForm, QuestForm, LocationWizard } from "../components/worldForms.jsx";
import { QuickReference } from "../components/QuickReference.jsx";

const WELCOME = [{
  role: "assistant",
  content: "Bienvenue a la Table. Utilise les boutons ci-dessus pour creer un PNJ, "
    + "une quete ou un lieu, ou demande-moi de lancer un combat, les des, ou de consulter l'etat de la partie.",
}];

export function Table({ setView }) {
  const { active, setActive } = useActiveCampaign();
  const role = active?.role || "dm";
  const models = useModels(DOMAINS.GAMEMASTER);
  const [model, setModel] = useState("mistral:7b-instruct");
  const { campaign, refresh } = useCampaign();

  const [modal, setModal] = useState(null);
  const [sidebar, setSidebar] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);
  const [history, setHistory] = useState(null);     // null = chargement
  const [resuming, setResuming] = useState(!active?.id);  // recherche de la derniere campagne

  // Au clic sur La Table sans campagne active : reprend automatiquement la plus
  // recente (la "derniere en jeu"). Si aucune campagne, on tombera sur TableEmpty.
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

  // Charge l'historique de conversation a chaque changement de campagne.
  useEffect(() => {
    if (!active?.id) { setHistory(null); return; }
    setHistory(null);
    listMessages(active.id).then((msgs) => setHistory(msgs.length ? msgs : WELCOME)).catch(() => setHistory(WELCOME));
  }, [active?.id]);

  const afterCreate = () => { setModal(null); setReloadKey((k) => k + 1); refresh(); };
  const afterTurn = () => { refresh(); setReloadKey((k) => k + 1); };

  if (!active?.id) {
    if (resuming) return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: DISPLAY, letterSpacing: 3, textTransform: "uppercase", color: T.mistDim }}>
        Reprise de la derniere campagne...
      </div>
    );
    return <TableEmpty setView={setView} />;
  }
  const isDm = role === "dm";

  const sidebarNode = sidebar ? (
    <aside style={{ width: 330, flexShrink: 0, overflowY: "auto" }}>
      <Ledger campaign={campaign} />
      <div style={{ height: 16 }} />
      <CampaignTools campaignId={active.id} role={role} reloadKey={reloadKey} onChanged={refresh} />
      <ModelSelect models={models} value={model} onChange={setModel} />
    </aside>
  ) : null;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <CampaignHeader active={active} setActive={setActive} />

      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 48px", marginBottom: 6, flexWrap: "wrap" }}>
        {isDm && <>
          <ActionBtn icon={Users} label="Creer un PNJ" onClick={() => setModal("npc")} />
          <ActionBtn icon={Scroll} label="Creer une quete" onClick={() => setModal("quest")} />
          <ActionBtn icon={MapPin} label="Creer un lieu" onClick={() => setModal("location")} />
          <ActionBtn icon={UserPlus} label="Ajouter un PJ" onClick={() => setModal("addpj")} />
        </>}
        <div style={{ flex: 1 }} />
        <ActionBtn icon={BookOpenCheck} label="Reference" onClick={() => setModal("ref")} subtle />
        <ActionBtn icon={sidebar ? PanelRightClose : PanelRightOpen} label={sidebar ? "Masquer" : "Grimoire"} onClick={() => setSidebar((v) => !v)} subtle />
      </div>
      <div style={{ padding: "0 48px" }}><Divider label={isDm ? "Le Maitre du Jeu" : "A la table"} /></div>

      {history === null ? (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: DISPLAY, letterSpacing: 3, textTransform: "uppercase", color: T.mistDim }}>
          Chargement de l'historique...
        </div>
      ) : (
        <TableChat key={active.id} campaignId={active.id} role={role} model={model} initial={history} sidebar={sidebarNode} onTurn={afterTurn} />
      )}

      {modal === "npc" && <Modal title="Creer un PNJ" onClose={() => setModal(null)}><NpcForm campaignId={active.id} onDone={afterCreate} /></Modal>}
      {modal === "quest" && <Modal title="Lancer une quete" onClose={() => setModal(null)}><QuestForm campaignId={active.id} onDone={afterCreate} /></Modal>}
      {modal === "location" && <Modal title="Creer un lieu" onClose={() => setModal(null)} width={600}><LocationWizard campaignId={active.id} onDone={afterCreate} /></Modal>}
      {modal === "ref" && <Modal title="Reference rapide" onClose={() => setModal(null)} width={640}><QuickReference /></Modal>}
      {modal === "addpj" && <Modal title="Personnages de l'aventure" onClose={() => setModal(null)} width={520}><AddPlayers campaignId={active.id} onChanged={afterTurn} /></Modal>}
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
      setMsg(c.name + (inIt ? " a quitte l" + AP + "aventure." : " rejoint l" + AP + "aventure."));
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
