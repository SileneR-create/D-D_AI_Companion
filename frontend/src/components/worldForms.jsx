/** Formulaires/assistants de creation du monde (modales) + don d'objet. */
import { useEffect, useMemo, useState } from "react";
import { Plus, Gem, ChevronRight, ChevronLeft, Check } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { createNpc, createLocation, createQuest, listNpcs } from "../api/campaigns.js";
import { createLibraryNpc, createLibraryLocation } from "../api/library.js";
import { listItems, assignItem } from "../api/forge.js";
import { getMonsters, getMonsterDetail } from "../api/arsenal.js";
import { Qcm } from "./Qcm.jsx";

const KINDS = ["Principale", "Secondaire", "Annexe"];
const ATTITUDES = ["Amical", "Neutre", "Hostile", "Inconnu"];

export function NpcForm({ campaignId, onDone }) {
  const [f, setF] = useState({ name: "", appearance: "", race: "", occupation: "", attitude: ATTITUDES[1],
    is_adversary: false, armor_class: "", hit_points: "", challenge_rating: "", monster_ref: "" });
  const [err, setErr] = useState(null); const [busy, setBusy] = useState(false);
  const set = (k, v) => setF((s) => ({ ...s, [k]: v }));

  // Recherche de monstre SRD pour pre-remplir CA/PV/FP.
  const [monsters, setMonsters] = useState([]);
  const [mq, setMq] = useState("");
  useEffect(() => { if (f.is_adversary && monsters.length === 0) getMonsters().then(setMonsters); }, [f.is_adversary]);
  const matches = mq.trim().length >= 2
    ? monsters.filter((m) => m.name.toLowerCase().includes(mq.toLowerCase())).slice(0, 8) : [];
  const pickMonster = async (m) => {
    setMq(m.name);
    const d = await getMonsterDetail(m.index);
    if (d) setF((s) => ({ ...s, monster_ref: d.name, armor_class: d.armor_class ?? "", hit_points: d.hit_points ?? "", challenge_rating: d.challenge_rating ?? "",
      attitude: "Hostile" }));
  };

  const submit = async () => {
    if (!f.name.trim()) return setErr("Nom requis.");
    setBusy(true); setErr(null);
    const payload = {
      name: f.name, appearance: f.appearance || null, race: f.race || null, occupation: f.occupation || null, attitude: f.attitude,
      is_adversary: f.is_adversary,
      armor_class: f.armor_class === "" ? null : Number(f.armor_class),
      hit_points: f.hit_points === "" ? null : Number(f.hit_points),
      challenge_rating: f.challenge_rating || null,
      monster_ref: f.monster_ref || null,
    };
    try {
      // Avec campagne -> PNJ rattache + actif ; sans campagne (Archives) -> PNJ de bibliotheque.
      if (campaignId) await createNpc(campaignId, payload);
      else await createLibraryNpc(payload);
      onDone();
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <>
      <Field label="Nom"><input style={inp} value={f.name} onChange={(e) => set("name", e.target.value)} /></Field>
      <Field label="Description physique"><textarea style={{ ...inp, resize: "vertical" }} rows={3} value={f.appearance} onChange={(e) => set("appearance", e.target.value)} /></Field>
      <div style={{ display: "flex", gap: 10 }}>
        <Field label="Race"><input style={inp} value={f.race} onChange={(e) => set("race", e.target.value)} /></Field>
        <Field label="Metier"><input style={inp} value={f.occupation} onChange={(e) => set("occupation", e.target.value)} /></Field>
        <Field label="Attitude"><select style={{ ...inp, cursor: "pointer" }} value={f.attitude} onChange={(e) => set("attitude", e.target.value)}>{ATTITUDES.map((a) => <option key={a}>{a}</option>)}</select></Field>
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 8, margin: "10px 0 4px", cursor: "pointer", fontFamily: BODY, fontSize: 15, color: T.mist }}>
        <input type="checkbox" checked={f.is_adversary} onChange={(e) => set("is_adversary", e.target.checked)} />
        Adversaire des heros (combat)
      </label>
      {f.is_adversary && (
        <div style={{ padding: "10px 12px", borderRadius: 4, background: T.ink, border: `1px solid ${T.goldDim}` }}>
          <Field label="Associer un monstre (SRD) — pre-remplit CA/PV/FP">
            <input style={inp} value={mq} onChange={(e) => setMq(e.target.value)} placeholder="ex: goblin, dragon…" />
          </Field>
          {matches.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 6 }}>
              {matches.map((m) => (
                <button key={m.index} onClick={() => pickMonster(m)} style={{ padding: "5px 10px", borderRadius: 3, cursor: "pointer",
                  background: "transparent", border: `1px solid ${T.line}`, color: T.mist, fontFamily: BODY, fontSize: 13 }}>{m.name}</button>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 10 }}>
            <Field label="Classe d'armure"><input type="number" style={inp} value={f.armor_class} onChange={(e) => set("armor_class", e.target.value)} /></Field>
            <Field label="Points de vie"><input type="number" style={inp} value={f.hit_points} onChange={(e) => set("hit_points", e.target.value)} /></Field>
            <Field label="FP"><input style={inp} value={f.challenge_rating} onChange={(e) => set("challenge_rating", e.target.value)} /></Field>
          </div>
        </div>
      )}

      <Submit onClick={submit} busy={busy} label="Creer le PNJ" />
      {err && <Err>{err}</Err>}
    </>
  );
}

export function QuestForm({ campaignId, onDone }) {
  const [npcs, setNpcs] = useState([]);
  const [f, setF] = useState({ title: "", kind: KINDS[0], objective: "", giver: "", enemies: "", description: "" });
  const [err, setErr] = useState(null); const [busy, setBusy] = useState(false);
  useEffect(() => { listNpcs(campaignId).then(setNpcs).catch(() => {}); }, [campaignId]);
  const set = (k, v) => setF((s) => ({ ...s, [k]: v }));
  const submit = async () => {
    if (!f.title.trim()) return setErr("Titre requis.");
    if (!f.objective.trim()) return setErr("Un objectif est obligatoire.");
    setBusy(true); setErr(null);
    try {
      await createQuest(campaignId, { title: f.title, kind: f.kind, objective: f.objective, giver: f.giver || null,
        description: f.description || null, enemies: f.enemies.split(",").map((e) => e.trim()).filter(Boolean) });
      onDone();
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <>
      <Field label="Titre"><input style={inp} value={f.title} onChange={(e) => set("title", e.target.value)} /></Field>
      <Field label="Objectif (obligatoire)"><input style={inp} value={f.objective} onChange={(e) => set("objective", e.target.value)} /></Field>
      <div style={{ display: "flex", gap: 10 }}>
        <Field label="Type"><select style={{ ...inp, cursor: "pointer" }} value={f.kind} onChange={(e) => set("kind", e.target.value)}>{KINDS.map((k) => <option key={k}>{k}</option>)}</select></Field>
        <Field label="Donneur (PNJ)"><select style={{ ...inp, cursor: "pointer" }} value={f.giver} onChange={(e) => set("giver", e.target.value)}><option value="">—</option>{npcs.map((n) => <option key={n.id} value={n.name}>{n.name}</option>)}</select></Field>
      </div>
      <Field label="Ennemis (separes par des virgules)"><input style={inp} value={f.enemies} onChange={(e) => set("enemies", e.target.value)} /></Field>
      <Field label="Description"><textarea style={{ ...inp, resize: "vertical" }} rows={2} value={f.description} onChange={(e) => set("description", e.target.value)} /></Field>
      <Submit onClick={submit} busy={busy} label="Lancer la quete" />
      {err && <Err>{err}</Err>}
    </>
  );
}

/* ----------------- Assistant de creation de lieu (QCM) ------------------- */
const LOC_STEPS = [
  { key: "name", q: "Quel est le nom du lieu ?", type: "text", placeholder: "Ex: La Taverne du Dragon Ivre" },
  { key: "inout", q: "Interieur ou exterieur ?", type: "single", options: ["Interieur", "Exterieur"] },
  { key: "env", q: "Quel type d'environnement ?", type: "single", options: ["Ville", "Village", "Campagne", "Foret", "Desert", "Montagne", "Marais", "Souterrain", "Cote / Mer"] },
  { key: "pop", q: "Le lieu est-il peuple ?", type: "single", options: ["Tres peuple", "Peu peuple", "Desert / abandonne"] },
  { key: "people", q: "Quelle population l'habite ?", type: "multi", options: ["Humains", "Elfes", "Nains", "Halfelins", "Marchands", "Gardes / soldats", "Hors-la-loi", "Religieux", "Creatures", "Nobles"], skipIf: (a) => a.pop === "Desert / abandonne" },
  { key: "temp", q: "Quelle temperature ?", type: "single", options: ["Glacial", "Froid", "Tempere", "Chaud", "Torride"] },
  { key: "humid", q: "Quel taux d'humidite ?", type: "single", options: ["Sec", "Normal", "Humide", "Detrempe"] },
  { key: "smell", q: "Quelles odeurs flottent dans l'air ?", type: "multi", options: ["Bois & fumee", "Epices & cuisine", "Moisi & humidite", "Sang & metal", "Fleurs & herbe", "Pourriture", "Air pur", "Encens", "Mer & sel", "Renferme"] },
  { key: "mood", q: "Quelle est l'ambiance generale ?", type: "single", options: ["Paisible", "Animee", "Tendue", "Hostile", "Mysterieuse", "Lugubre"] },
  { key: "notes", q: "Un detail particulier ? (facultatif)", type: "text", optional: true, textarea: true, placeholder: "Une fontaine asséchée au centre..." },
];

function composeDescription(a) {
  const s = [];
  const loc = [a.inout ? a.inout.toLowerCase() : null, a.env ? `de type ${a.env.toLowerCase()}` : null].filter(Boolean).join(" ");
  s.push(`${a.name} est un lieu ${loc}.`.replace(/\s+/g, " "));
  if (a.pop === "Desert / abandonne") s.push("Il est desert et abandonne.");
  else if (a.pop) s.push(`${a.pop}${a.people?.length ? `, on y croise ${a.people.join(", ").toLowerCase()}` : ""}.`);
  const clim = [a.temp?.toLowerCase(), a.humid?.toLowerCase()].filter(Boolean).join(" et ");
  if (clim) s.push(`Le climat y est ${clim}.`);
  if (a.smell?.length) s.push(`On y respire ${a.smell.join(", ").toLowerCase()}.`);
  if (a.mood) s.push(`L'ambiance generale est ${a.mood.toLowerCase()}.`);
  if (a.notes) s.push(a.notes);
  return s.join(" ");
}

export function LocationWizard({ campaignId, onDone }) {
  return (
    <Qcm steps={LOC_STEPS} compose={composeDescription} createLabel="Creer le lieu"
      onCreate={async (a) => {
        const body = { name: a.name, description: composeDescription(a) };
        if (campaignId) await createLocation(campaignId, body);
        else await createLibraryLocation(body);
        onDone();
      }} />
  );
}

/** Don d'un objet de la forge a une cible (perso/PNJ/quete). */
export function GiveItem({ targetType, targetId, onDone }) {
  const [items, setItems] = useState([]);
  const [pick, setPick] = useState("");
  const [msg, setMsg] = useState(null);
  useEffect(() => { listItems().then(setItems).catch(() => {}); }, []);
  const give = async () => {
    if (!pick) return setMsg("Choisissez un objet.");
    try { await assignItem({ item_id: Number(pick), target_type: targetType, target_id: targetId }); setMsg("Objet attribue."); onDone?.(); }
    catch (e) { setMsg(e.message); }
  };
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
      <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ ...inp, flex: 1, cursor: "pointer" }}>
        <option value="">Donner un objet…</option>
        {items.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}
      </select>
      <button onClick={give} className="glow" style={miniBtn}><Gem size={12} /> Donner</button>
      {msg && <span style={{ fontFamily: BODY, fontSize: 12.5, color: T.gold, alignSelf: "center" }}>{msg}</span>}
    </div>
  );
}

/* helpers */
function Field({ label, children }) {
  return <div style={{ flex: 1, marginBottom: 12 }}><label style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 2, textTransform: "uppercase", color: T.mistDim }}>{label}</label>{children}</div>;
}
function Submit({ onClick, busy, label }) {
  return <button onClick={onClick} disabled={busy} className="glow" style={{ ...goldBtn, marginTop: 6, opacity: busy ? 0.6 : 1 }}><Plus size={14} /> {label}</button>;
}
const Err = ({ children }) => <div style={{ fontFamily: BODY, fontSize: 14, color: T.emberBright, marginTop: 8 }}>{children}</div>;
const inp = { width: "100%", marginTop: 5, padding: "10px 12px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 16, outline: "none", boxSizing: "border-box" };
const goldBtn = { display: "inline-flex", alignItems: "center", gap: 8, padding: "10px 18px", borderRadius: 3, background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase" };
const ghostBtn = { display: "inline-flex", alignItems: "center", gap: 6, padding: "10px 16px", borderRadius: 3, background: "transparent", border: `1px solid ${T.line}`, color: T.mistDim, cursor: "pointer", fontFamily: DISPLAY, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase" };
const miniBtn = { display: "inline-flex", alignItems: "center", gap: 6, padding: "9px 14px", borderRadius: 3, background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1, textTransform: "uppercase" };
const choice = { padding: "12px 14px", textAlign: "left", borderRadius: 4, cursor: "pointer", background: T.panel, border: `1px solid ${T.line}`, color: T.mist, fontFamily: BODY, fontSize: 15.5 };
const choiceOn = { background: "rgba(201,162,75,.12)", border: `1px solid ${T.gold}`, color: T.parch };
