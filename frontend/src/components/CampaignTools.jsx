/**
 * Panneau de gestion de campagne (sidebar de La Table) : quetes, PNJ, lieux.
 * On peut SELECTIONNER des elements de la bibliotheque (activer dans la campagne
 * courante) et les retirer (desactiver, reversible). La suppression definitive
 * se fait dans la Bibliotheque.
 */
import { useEffect, useState } from "react";
import { Scroll, Users, MapPin, UserPlus, Check, AlertTriangle, X, Plus } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { SectionLabel } from "./ornaments.jsx";
import { GiveItem } from "./worldForms.jsx";
import { listQuests, updateQuest, deleteQuest, listNpcs, listLocations, recruitCharacter } from "../api/campaigns.js";
import { listCharacters } from "../api/characters.js";
import { getLibrary, activateElement, deactivateElement } from "../api/library.js";
import { listItems, assignItem } from "../api/forge.js";
import { Gem } from "lucide-react";

export function CampaignTools({ campaignId, role, reloadKey, onChanged }) {
  const [quests, setQuests] = useState([]);
  const [npcs, setNpcs] = useState([]);
  const [locs, setLocs] = useState([]);
  const [lib, setLib] = useState({ npcs: [], locations: [] });
  const [confirm, setConfirm] = useState(null);
  const [giveTo, setGiveTo] = useState(null);

  const refresh = () => {
    listQuests(campaignId).then(setQuests).catch(() => {});
    listNpcs(campaignId).then(setNpcs).catch(() => {});
    listLocations(campaignId).then(setLocs).catch(() => {});
    getLibrary().then(setLib).catch(() => {});
  };
  useEffect(() => { refresh(); }, [campaignId, reloadKey]);

  const validate = async (q) => { await updateQuest(campaignId, q.id, { objective_done: !q.objective_done, status: !q.objective_done ? "completed" : "active" }); refresh(); onChanged?.(); };
  const delQuest = async (id) => { await deleteQuest(campaignId, id); setConfirm(null); refresh(); onChanged?.(); };
  const addElem = async (type, id) => { try { await activateElement({ element_type: type, element_id: id, campaign_id: campaignId }); refresh(); onChanged?.(); } catch { /* */ } };
  const removeElem = async (type, id) => { try { await deactivateElement({ element_type: type, element_id: id, campaign_id: campaignId }); refresh(); onChanged?.(); } catch { /* */ } };
  const isDm = role === "dm";

  const npcsLibre = lib.npcs.filter((n) => !n.active_in.includes(campaignId));
  const locsLibre = lib.locations.filter((l) => !l.active_in.includes(campaignId));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Box icon={Scroll} title={`Quetes (${quests.length})`}>
        {quests.length === 0 ? <Empty>Aucune quete.</Empty> : quests.map((q) => (
          <Card key={q.id}>
            <Head title={q.title} onDelete={isDm ? () => setConfirm(`q:${q.id}`) : null} />
            <Sub>{q.kind} · {q.status}</Sub>
            <Line>Objectif : {q.objective} {q.objective_done && <span style={{ color: T.goldBright }}>✦</span>}</Line>
            {q.enemies?.length > 0 && <Italic>Ennemis : {q.enemies.join(", ")}</Italic>}
            {isDm && <>
              <button onClick={() => validate(q)} style={{ ...mini, marginTop: 6 }}><Check size={11} /> {q.objective_done ? "Annuler" : "Valider l'objectif"}</button>
              <button onClick={() => setGiveTo(giveTo === `quest:${q.id}` ? null : `quest:${q.id}`)} style={{ ...mini, marginTop: 6, marginLeft: 6 }}>Donner un objet</button>
              {giveTo === `quest:${q.id}` && <GiveItem targetType="quest" targetId={q.id} onDone={() => setGiveTo(null)} />}
            </>}
            {confirm === `q:${q.id}` && <Confirm label="Supprimer cette quete ?" onYes={() => delQuest(q.id)} onNo={() => setConfirm(null)} />}
          </Card>
        ))}
      </Box>

      {isDm && (
        <Box icon={Users} title={`PNJ actifs (${npcs.length})`}>
          <LibPicker items={npcsLibre} placeholder="Activer un PNJ de la bibliotheque…" onAdd={(id) => addElem("npc", id)} />
          {npcs.length === 0 ? <Empty>Aucun PNJ actif. Selectionne-en un ci-dessus.</Empty> : npcs.map((n) => (
            <Card key={n.id}>
              <Head title={n.name} ornate onRemove={() => removeElem("npc", n.id)} />
              <Sub>{[n.occupation, n.attitude].filter(Boolean).join(" · ")}</Sub>
              {n.appearance && <Italic>{n.appearance}</Italic>}
              <button onClick={() => setGiveTo(giveTo === `npc:${n.id}` ? null : `npc:${n.id}`)} style={{ ...mini, marginTop: 6 }}>Donner un objet</button>
              {giveTo === `npc:${n.id}` && <GiveItem targetType="npc" targetId={n.id} onDone={() => setGiveTo(null)} />}
            </Card>
          ))}
        </Box>
      )}

      {isDm && (
        <Box icon={MapPin} title={`Lieux actifs (${locs.length})`}>
          <LibPicker items={locsLibre} placeholder="Activer un lieu de la bibliotheque…" onAdd={(id) => addElem("location", id)} />
          {locs.length === 0 ? <Empty>Aucun lieu actif.</Empty> : locs.map((l) => (
            <Card key={l.id}>
              <Head title={l.name} ornate onRemove={() => removeElem("location", l.id)} />
              {l.description && <Italic>{l.description}</Italic>}
            </Card>
          ))}
        </Box>
      )}

      {isDm && <AssignBox campaignId={campaignId} npcs={npcs} quests={quests} />}
      <RecruitPanel campaignId={campaignId} onChanged={onChanged} />
    </div>
  );
}

function LibPicker({ items, placeholder, onAdd }) {
  const [pick, setPick] = useState("");
  if (items.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
      <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ ...inp, flex: 1, cursor: "pointer" }}>
        <option value="">{placeholder}</option>
        {items.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}
      </select>
      <button onClick={() => { if (pick) { onAdd(Number(pick)); setPick(""); } }} className="glow" style={mini}><Plus size={11} /> Activer</button>
    </div>
  );
}

function AssignBox({ campaignId, npcs, quests }) {
  const [items, setItems] = useState([]);
  const [chars, setChars] = useState([]);
  const [item, setItem] = useState("");
  const [type, setType] = useState("character");
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState(null);
  useEffect(() => {
    listItems().then(setItems).catch(() => {});
    listCharacters().then((cs) => setChars(cs.filter((c) => c.campaign_id === campaignId))).catch(() => {});
  }, [campaignId]);
  const targets = type === "character" ? chars : type === "npc" ? npcs : quests;
  const give = async () => {
    setMsg(null);
    if (!item || !target) return setMsg("Choisis un objet et une cible.");
    try { await assignItem({ item_id: Number(item), target_type: type, target_id: Number(target) }); setMsg("Objet attribue."); }
    catch (e) { setMsg(e.message); }
  };
  return (
    <Box icon={Gem} title="Attribuer un objet">
      {items.length === 0 ? <Empty>Aucun objet dans la Forge.</Empty> : (
        <>
          <select value={item} onChange={(e) => setItem(e.target.value)} style={{ ...inp, width: "100%", cursor: "pointer" }}>
            <option value="">Objet…</option>
            {items.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}
          </select>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <select value={type} onChange={(e) => { setType(e.target.value); setTarget(""); }} style={{ ...inp, width: 110, cursor: "pointer" }}>
              <option value="character">Perso</option>
              <option value="npc">PNJ</option>
              <option value="quest">Quete</option>
            </select>
            <select value={target} onChange={(e) => setTarget(e.target.value)} style={{ ...inp, flex: 1, cursor: "pointer" }}>
              <option value="">Cible…</option>
              {targets.map((t) => <option key={t.id} value={t.id}>{t.name || t.title}</option>)}
            </select>
            <button onClick={give} className="glow" style={mini}>Attribuer</button>
          </div>
          {msg && <div style={{ fontFamily: BODY, fontSize: 12.5, color: T.gold, marginTop: 8 }}>{msg}</div>}
        </>
      )}
    </Box>
  );
}

function RecruitPanel({ campaignId, onChanged }) {
  const [available, setAvailable] = useState([]);
  const [pick, setPick] = useState("");
  const [msg, setMsg] = useState(null);
  const refresh = () => listCharacters().then((cs) => setAvailable(cs.filter((c) => !c.campaign_id))).catch(() => {});
  useEffect(() => { refresh(); }, [campaignId]);
  const recruit = async () => {
    if (!pick) return;
    try { const c = available.find((x) => String(x.id) === pick); await recruitCharacter(campaignId, Number(pick)); setMsg(`${c?.name} rejoint la table.`); setPick(""); refresh(); onChanged?.(); }
    catch (e) { setMsg(e.message); }
  };
  return (
    <Box icon={UserPlus} title="Recruter un personnage">
      {available.length === 0 ? <Empty>Aucun personnage hors campagne.</Empty> : (
        <div style={{ display: "flex", gap: 8 }}>
          <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ ...inp, flex: 1, cursor: "pointer" }}>
            <option value="">Choisir…</option>
            {available.map((c) => <option key={c.id} value={c.id}>{c.name} (Niv. {c.class_level})</option>)}
          </select>
          <button onClick={recruit} className="glow" style={mini}>Recruter</button>
        </div>
      )}
      {msg && <div style={{ fontFamily: BODY, fontSize: 12.5, color: T.gold, marginTop: 8 }}>{msg}</div>}
    </Box>
  );
}

/* helpers */
function Box({ icon, title, children }) {
  return <div style={{ background: T.panel, border: `1px solid ${T.line}`, borderRadius: 4, padding: "12px 14px" }}><SectionLabel icon={icon} text={title} />{children}</div>;
}
function Card({ children }) { return <div style={{ padding: "9px 10px", marginBottom: 6, borderRadius: 3, background: T.ink, border: `1px solid ${T.line}` }}>{children}</div>; }
function Head({ title, ornate, onDelete, onRemove }) {
  return <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
    <span style={{ fontFamily: ornate ? ORNATE : BODY, fontSize: 15, color: T.parch }}>{title}</span>
    {onDelete && <button onClick={onDelete} title="Supprimer" style={{ background: "none", border: "none", cursor: "pointer", color: T.mistDim, display: "flex" }}><AlertTriangle size={13} /></button>}
    {onRemove && <button onClick={onRemove} title="Retirer de la campagne" style={{ background: "none", border: "none", cursor: "pointer", color: T.mistDim, display: "flex" }}><X size={15} /></button>}
  </div>;
}
const Sub = ({ children }) => <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1, textTransform: "uppercase", color: T.goldDim }}>{children}</div>;
const Line = ({ children }) => <div style={{ fontFamily: BODY, fontSize: 13.5, color: T.mist, marginTop: 4 }}>{children}</div>;
const Italic = ({ children }) => <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 12.5, color: T.mistDim, marginTop: 2 }}>{children}</div>;
const Empty = ({ children }) => <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13, color: T.mistDim }}>{children}</div>;
function Confirm({ label, onYes, onNo }) {
  return <div style={{ marginTop: 8, padding: "8px 10px", borderRadius: 3, background: "rgba(166,64,46,.12)", border: `1px solid ${T.ember}` }}>
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: BODY, fontSize: 13, color: T.parch }}><AlertTriangle size={13} color={T.emberBright} /> {label}</div>
    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
      <button onClick={onYes} style={{ ...mini, borderColor: T.ember, color: T.emberBright }}>Supprimer</button>
      <button onClick={onNo} style={mini}>Annuler</button>
    </div>
  </div>;
}
const inp = { padding: "8px 10px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none", boxSizing: "border-box" };
const mini = { display: "inline-flex", alignItems: "center", gap: 5, padding: "6px 11px", borderRadius: 3, background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1, textTransform: "uppercase" };
