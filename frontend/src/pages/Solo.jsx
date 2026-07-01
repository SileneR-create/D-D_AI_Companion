/** Jeu en solo : liste des aventures, assistant de création, et jeu (narrateur LLM). */
import { useEffect, useMemo, useRef, useState } from "react";
import { Sparkles, Play, ChevronLeft, ScrollText, UserPlus, Dices } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Divider } from "../components/ornaments.jsx";
import { Modal } from "../components/Modal.jsx";
import { startSolo, listSoloAdventures } from "../api/solo.js";
import { listCharacters } from "../api/characters.js";
import { listMessages } from "../api/campaigns.js";
import { useModels } from "../hooks/useModels.js";
import { useViewport } from "../hooks/useViewport.js";
import { computeSheet, ABILITY_KEYS, ABILITY_FR } from "../lib/sheet.js";
import { DOMAINS } from "../api";
import { TableChat } from "../components/TableChat.jsx";

const GENRES = [
  ["heroique", "Heroic fantasy", "Hauts faits & espoir"],
  ["sombre", "Dark fantasy", "L'espoir se paie cher"],
  ["enquete", "Mystère & enquête", "Indices & déductions"],
  ["donjon", "Exploration & donjon", "Profondeurs & trésors"],
  ["intrigue", "Intrigue politique", "Complots de cour"],
  ["horreur", "Horreur", "Peur & survie"],
];
const LENGTHS = [["courte", "Courte", "~3 scènes"], ["moyenne", "Moyenne", "~5 scènes"], ["longue", "Longue", "~8 scènes"]];
const TONES = [["serieux", "Sérieux"], ["heroique", "Héroïque"], ["leger", "Léger"]];

const WELCOME_SOLO = [{ role: "assistant",
  content: "Ton aventure est prête. Dis-moi ce que fait ton personnage, ou demande-moi de planter le décor du premier acte." }];

export function Solo({ setView }) {
  const { isNarrow } = useViewport();
  const px = isNarrow ? 16 : 48;
  const [mode, setMode] = useState("menu");       // menu | setup | play
  const [adventures, setAdventures] = useState([]);
  const [playing, setPlaying] = useState(null);
  const [history, setHistory] = useState(null);

  const refresh = () => listSoloAdventures().then(setAdventures).catch(() => {});
  useEffect(() => { refresh(); }, []);

  const open = async (adv) => {
    setPlaying(adv); setHistory(null); setMode("play");
    try { const m = await listMessages(adv.id); setHistory(m.length ? m : WELCOME_SOLO); }
    catch { setHistory(WELCOME_SOLO); }
  };

  if (mode === "play" && playing) {
    return <SoloPlay adv={playing} history={history} onBack={() => { setMode("menu"); setPlaying(null); refresh(); }} />;
  }
  if (mode === "setup") {
    return <SoloSetup setView={setView} onCancel={() => setMode("menu")} onStarted={(camp) => open(camp)} />;
  }

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: `40px ${px}px`, maxWidth: 920, margin: "0 auto", width: "100%" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 5, textTransform: "uppercase", color: T.gold }}>Jouer seul</div>
        <h1 style={{ fontFamily: ORNATE, fontSize: 36, color: T.parch, margin: "6px 0 2px" }}>Le Jeu en solo</h1>
        <p style={{ fontFamily: BODY, fontSize: 16, fontStyle: "italic", color: T.mist, maxWidth: 540, margin: "0 auto 12px" }}>
          Choisis un genre, une durée, ton héros — et le narrateur tisse l'aventure pour toi.
        </p>
      </div>
      <Divider />

      <button onClick={() => setMode("setup")} className="lift" style={{
        width: "100%", display: "flex", alignItems: "center", gap: 14, padding: "20px 22px", marginBottom: 18, cursor: "pointer",
        background: "linear-gradient(180deg, rgba(201,162,75,.10), rgba(201,162,75,.03))", border: `1px solid ${T.gold}`, borderRadius: 6, color: T.parch }}>
        <Sparkles size={26} color={T.gold} strokeWidth={1.3} />
        <div style={{ textAlign: "left" }}>
          <div style={{ fontFamily: ORNATE, fontSize: 21, color: T.parch }}>Nouvelle aventure</div>
          <div style={{ fontFamily: BODY, fontSize: 14.5, color: T.mistDim }}>Générer une histoire sur mesure</div>
        </div>
      </button>

      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold, margin: "6px 0 12px" }}>
        Mes aventures ({adventures.length})
      </div>
      {adventures.length === 0 ? (
        <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim }}>
          Aucune aventure pour l'instant. Lance ta première ci-dessus.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14 }}>
          {adventures.map((a) => (
            <div key={a.id} style={card}>
              <div style={{ fontFamily: ORNATE, fontSize: 18, color: T.parch }}>{a.name}</div>
              <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.gold, margin: "4px 0 6px" }}>
                {a.genre || "Aventure"} · {a.acts} actes{a.character_name ? ` · ${a.character_name}` : ""}
              </div>
              {a.description && <div style={{ fontFamily: BODY, fontSize: 13.5, fontStyle: "italic", color: T.mistDim, lineHeight: 1.5, maxHeight: 76, overflow: "hidden" }}>{a.description}</div>}
              <button onClick={() => open(a)} className="glow" style={{ ...goldBtn, marginTop: 12 }}>
                <Play size={13} /> Reprendre
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SoloPlay({ adv, history, onBack }) {
  const models = useModels(DOMAINS.GAMEMASTER);
  const [model, setModel] = useState("mistral:7b-instruct");
  const [char, setChar] = useState(null);
  const [dice, setDice] = useState(false);
  const api = useRef({ send: null, messages: [] });
  useEffect(() => {
    listCharacters().then((cs) => setChar(cs.find((c) => c.campaign_id === adv.id || (c.campaigns || []).includes(adv.id)) || null)).catch(() => {});
  }, [adv.id]);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div style={{ textAlign: "center", padding: "12px 48px 2px", position: "relative" }}>
        <button onClick={onBack} style={{ position: "absolute", left: 48, top: 12, display: "inline-flex", alignItems: "center", gap: 5,
          background: "transparent", border: `1px solid ${T.line}`, borderRadius: 4, padding: "6px 11px", cursor: "pointer",
          color: T.mistDim, fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase" }}>
          <ChevronLeft size={13} /> Mes aventures
        </button>
        {/* Selecteur de modele discret : choisir l'esprit conteur (mistral-nemo conseille). */}
        <select value={model} onChange={(e) => setModel(e.target.value)} title="Modele du narrateur"
          style={{ position: "absolute", right: 48, top: 12, maxWidth: 190, background: T.ink, color: T.mistDim,
            border: `1px solid ${T.line}`, borderRadius: 4, padding: "6px 9px", fontFamily: BODY, fontSize: 12.5,
            outline: "none", cursor: "pointer" }}>
          {(models.length ? models : [model]).map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 5, textTransform: "uppercase", color: T.gold }}>Aventure en solo</div>
        <div style={{ fontFamily: ORNATE, fontSize: 24, color: T.parch }}>{adv.name}</div>
        <div>
          <button onClick={() => setDice(true)} className="glow" style={{ marginTop: 6, display: "inline-flex", alignItems: "center", gap: 6,
            background: "transparent", border: `1px solid ${T.gold}`, borderRadius: 4, padding: "6px 13px", cursor: "pointer",
            color: T.gold, fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase" }}>
            <Dices size={13} /> Lancer un dé
          </button>
        </div>
      </div>
      <div style={{ padding: "0 48px" }}><Divider label="Le récit" /></div>
      {history === null ? (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: DISPLAY, letterSpacing: 3, textTransform: "uppercase", color: T.mistDim }}>
          Ouverture du récit...
        </div>
      ) : (
        <TableChat key={adv.id} campaignId={adv.id} role="dm" model={model} initial={history} sidebar={null}
          onReady={(a) => { api.current = a; }} />
      )}
      {dice && (char
        ? <SoloDice character={char} api={api} onClose={() => setDice(false)} />
        : <Modal title="Lancer les dés" onClose={() => setDice(false)}>
            <div style={{ fontFamily: BODY, fontStyle: "italic", color: T.mistDim }}>Aucun personnage rattaché à cette aventure.</div>
          </Modal>)}
    </div>
  );
}

/** Pop-up de jet : d20 + modificateur du personnage (competence ou caracteristique),
 *  avec pre-selection du test detecte dans le dernier message du narrateur. */
function SoloDice({ character, api, onClose }) {
  const sheet = useMemo(() => computeSheet(character), [character]);
  const ABILITY_FULL = { strength: "force", dexterity: "dexterite", constitution: "constitution",
    intelligence: "intelligence", wisdom: "sagesse", charisma: "charisme" };
  const norm = (s) => (s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

  const options = [
    ...sheet.skills.map((s) => ({ key: "sk:" + s.key, label: s.label, value: s.value })),
    ...ABILITY_KEYS.map((k) => ({ key: "ab:" + k, label: `Test de ${ABILITY_FR[k]}`, value: sheet.mods[k] })),
  ];
  const detected = (() => {
    const msgs = api.current?.messages || [];
    let last = "";
    for (let i = msgs.length - 1; i >= 0; i--) { if (msgs[i].role === "assistant") { last = msgs[i].content; break; } }
    const t = norm(last);
    for (const s of sheet.skills) if (t && t.includes(norm(s.label))) return "sk:" + s.key;
    for (const k of ABILITY_KEYS) if (t && t.includes(ABILITY_FULL[k])) return "ab:" + k;
    return options[0] && options[0].key;
  })();
  const [sel, setSel] = useState(detected);
  const [result, setResult] = useState(null);
  const current = options.find((o) => o.key === sel) || options[0];
  const fmt = (v) => (v >= 0 ? `+${v}` : `${v}`);

  const roll = () => {
    const d = 1 + Math.floor(Math.random() * 20);
    setResult({ d, mod: current.value, total: d + current.value, label: current.label, crit: d === 20, fail: d === 1 });
  };
  const transmit = () => {
    if (!result) return;
    const extra = result.crit ? " (reussite critique !)" : result.fail ? " (echec critique !)" : "";
    api.current?.send?.(`🎲 ${result.label} : d20 (${result.d}) ${fmt(result.mod)} = ${result.total}${extra}`);
    onClose();
  };

  return (
    <Modal title={`Lancer les dés — ${character.name}`} onClose={onClose} width={540}>
      <div style={{ fontFamily: BODY, fontSize: 14, color: T.mistDim, marginBottom: 10 }}>
        Choisis le test demandé par le narrateur : le d20 s'ajoute au modificateur de {character.name}.
      </div>
      <div style={{ maxHeight: 260, overflowY: "auto", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 6 }}>
        {options.map((o) => {
          const on = o.key === sel;
          return (
            <button key={o.key} onClick={() => { setSel(o.key); setResult(null); }} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 11px", borderRadius: 4, cursor: "pointer",
              background: on ? "rgba(201,162,75,.16)" : "transparent", border: `1px solid ${on ? T.gold : T.line}`,
              color: on ? T.parch : T.mist, fontFamily: BODY, fontSize: 13.5 }}>
              <span>{o.label}</span><span style={{ color: T.gold }}>{fmt(o.value)}</span>
            </button>
          );
        })}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 14, flexWrap: "wrap" }}>
        <button onClick={roll} className="glow" style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "10px 18px", borderRadius: 4,
          background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
          fontFamily: DISPLAY, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase" }}>
          <Dices size={15} /> Lancer le d20
        </button>
        {result && (
          <div style={{ fontFamily: BODY, fontSize: 15, color: T.parch }}>
            d20 <b>({result.d})</b> {fmt(result.mod)} = <b style={{ color: T.goldBright, fontSize: 20 }}>{result.total}</b>
            {result.crit && <span style={{ color: "#7bb37b", marginLeft: 8 }}>critique !</span>}
            {result.fail && <span style={{ color: T.emberBright, marginLeft: 8 }}>échec critique</span>}
          </div>
        )}
      </div>
      {result && (
        <button onClick={transmit} className="glow" style={{ marginTop: 14, display: "inline-flex", alignItems: "center", gap: 7, padding: "9px 16px", borderRadius: 4,
          background: "rgba(201,162,75,.10)", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
          fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" }}>
          Transmettre au récit
        </button>
      )}
    </Modal>
  );
}

function SoloSetup({ setView, onCancel, onStarted }) {
  const { isNarrow } = useViewport();
  const px = isNarrow ? 16 : 48;
  const [genre, setGenre] = useState("heroique");
  const [length, setLength] = useState("courte");
  const [tone, setTone] = useState("serieux");
  const [pitch, setPitch] = useState("");
  const [chars, setChars] = useState([]);
  const [charId, setCharId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => { listCharacters().then((cs) => { setChars(cs); if (cs[0]) setCharId(cs[0].id); }).catch(() => {}); }, []);

  const launch = async () => {
    if (!charId) return setErr("Choisis un personnage (ou crée-en un).");
    setBusy(true); setErr(null);
    try {
      const camp = await startSolo({ genre, length, tone, pitch: pitch || null, character_id: charId });
      onStarted(camp);
    } catch (e) { setErr(e.message); setBusy(false); }
  };

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: `32px ${px}px`, maxWidth: 760, margin: "0 auto", width: "100%" }}>
      <button onClick={onCancel} style={{ display: "inline-flex", alignItems: "center", gap: 5, background: "transparent",
        border: `1px solid ${T.line}`, borderRadius: 4, padding: "6px 11px", cursor: "pointer", color: T.mistDim,
        fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 14 }}>
        <ChevronLeft size={13} /> Retour
      </button>
      <h1 style={{ fontFamily: ORNATE, fontSize: 30, color: T.parch, margin: "0 0 4px" }}>Forger une aventure solo</h1>
      <Divider />

      <FieldLabel icon={ScrollText} text="Genre" />
      <ChipGrid options={GENRES} value={genre} onPick={setGenre} cols={3} withSub />

      <FieldLabel text="Durée" />
      <ChipGrid options={LENGTHS} value={length} onPick={setLength} cols={3} withSub />

      <FieldLabel text="Ton" />
      <ChipGrid options={TONES} value={tone} onPick={setTone} cols={3} />

      <FieldLabel text="Une envie particulière ? (facultatif)" />
      <textarea value={pitch} onChange={(e) => setPitch(e.target.value)} rows={2}
        placeholder="ex: je veux sauver mon village natal, retrouver un frère disparu..."
        style={{ width: "100%", padding: "10px 12px", borderRadius: 4, background: T.ink, border: `1px solid ${T.line}`,
          color: T.parch, fontFamily: BODY, fontSize: 15, outline: "none", resize: "vertical", boxSizing: "border-box" }} />

      <FieldLabel icon={UserPlus} text="Ton personnage" />
      {chars.length === 0 ? (
        <div style={{ padding: "14px 16px", borderRadius: 5, border: `1px dashed ${T.goldDim}`, background: T.panel }}>
          <div style={{ fontFamily: BODY, fontSize: 14.5, color: T.mistDim, marginBottom: 8 }}>Tu n'as pas encore de personnage.</div>
          <button onClick={() => setView("character")} className="glow" style={goldBtn}><UserPlus size={13} /> Créer un personnage</button>
        </div>
      ) : (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 8 }}>
            {chars.map((c) => {
              const on = c.id === charId;
              return (
                <button key={c.id} onClick={() => setCharId(c.id)} style={{
                  textAlign: "left", padding: "11px 13px", borderRadius: 4, cursor: "pointer",
                  background: on ? "rgba(201,162,75,.14)" : "transparent", border: `1px solid ${on ? T.gold : T.line}`,
                  color: on ? T.parch : T.mist }}>
                  <div style={{ fontFamily: BODY, fontSize: 15.5 }}>{on ? "✦ " : ""}{c.name}</div>
                  <div style={{ fontFamily: BODY, fontSize: 12.5, color: T.mistDim }}>{c.race} {c.character_class} niv. {c.class_level}</div>
                </button>
              );
            })}
          </div>
          <button onClick={() => setView("character")} style={{ marginTop: 8, background: "transparent", border: "none",
            color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase" }}>
            + Créer un autre personnage
          </button>
        </>
      )}

      <div style={{ marginTop: 22, display: "flex", alignItems: "center", gap: 14 }}>
        <button onClick={launch} disabled={busy || !charId} className="glow" style={{ ...goldBtn, padding: "12px 22px", fontSize: 12, opacity: busy || !charId ? 0.55 : 1 }}>
          <Sparkles size={15} /> {busy ? "Génération..." : "Lancer l'aventure"}
        </button>
        {err && <span style={{ fontFamily: BODY, fontSize: 14, color: T.emberBright }}>{err}</span>}
      </div>
    </div>
  );
}

function FieldLabel({ icon: Icon, text }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7, margin: "20px 0 9px" }}>
      {Icon && <Icon size={14} color={T.gold} strokeWidth={1.4} />}
      <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold }}>{text}</span>
    </div>
  );
}

function ChipGrid({ options, value, onPick, cols = 3, withSub }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 8 }}>
      {options.map(([val, label, sub]) => {
        const on = val === value;
        return (
          <button key={val} onClick={() => onPick(val)} style={{
            textAlign: "left", padding: "11px 13px", borderRadius: 4, cursor: "pointer",
            background: on ? "rgba(201,162,75,.14)" : "transparent", border: `1px solid ${on ? T.gold : T.line}`,
            color: on ? T.parch : T.mist }}>
            <div style={{ fontFamily: BODY, fontSize: 14.5 }}>{on ? "✦ " : ""}{label}</div>
            {withSub && sub && <div style={{ fontFamily: BODY, fontSize: 12, color: T.mistDim }}>{sub}</div>}
          </button>
        );
      })}
    </div>
  );
}

const card = { padding: "14px 16px", borderRadius: 6, background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` };
const goldBtn = { display: "inline-flex", alignItems: "center", gap: 7, padding: "9px 16px", borderRadius: 3, background: "transparent",
  border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" };
