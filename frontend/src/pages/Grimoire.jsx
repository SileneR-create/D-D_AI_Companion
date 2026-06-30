/** Ecran "Le Grimoire" — assistant Regles (D&D 5e + RAG), branche sur l'API. */
import { useEffect, useRef, useState } from "react";
import { Upload, ScrollText, Globe } from "lucide-react";
import { DOMAINS } from "../api";
import { T, DISPLAY, BODY } from "../theme.js";
import { useChatStream } from "../hooks/useChatStream.js";
import { useSources } from "../hooks/useSources.js";
import { crawlSite } from "../api/client.js";
import { Divider, ScreenTitle } from "../components/ornaments.jsx";
import { Conversation, Inkwell } from "../components/chat.jsx";
import { Modal } from "../components/Modal.jsx";
import { getReferenceIndex, getReferenceDetail } from "../api/rules.js";
import { ParchPanel, SourceLine, SigilPanel } from "../components/panels.jsx";

const WELCOME = [{
  role: "assistant",
  content: "Bienvenue, chercheur. Pose ta question sur les regles de la cinquieme edition : "
    + "mecaniques, sorts, creation de personnage... je consulte le manuel et les sources.",
}];

// Sceaux = outils MCP du serveur de regles (cf. server_config.json).
const SIGILS = [
  "search_all_categories",
  "filter_spells_by_level",
  "find_monsters_by_challenge_rating",
  "generate_treasure_hoard",
];

export function Grimoire() {
  const model = "mistral:7b-instruct";   // modele par defaut (selecteur retire du Grimoire)
  const [input, setInput] = useState("");
  const { messages, draft, streaming, toolName, send } = useChatStream(DOMAINS.RULES, { model, initial: WELCOME });
  const { sources, busy, error, upload, refresh } = useSources();

  const onSend = () => { send(input); setInput(""); };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Consultation des regles" title="Le Grimoire" />
      <div style={{ padding: "0 48px" }}><Divider label="Cinquieme Edition" /></div>
      <RulesSearch />

      <div style={{ flex: 1, display: "flex", gap: 28, padding: "4px 48px 14px", minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <Conversation messages={messages} draft={draft} streaming={streaming}
            toolName={toolName} source="basic-rules-fr.pdf" />
          <Inkwell value={input} setValue={setInput} onSend={onSend} disabled={streaming}
            placeholder="Formulez votre question a l'Oracle..." />
        </div>

        <aside style={{ width: 270, flexShrink: 0, overflowY: "auto" }}>
          <ParchPanel title="Sources convoquees">
            {sources.length === 0
              ? <SourceLine name="Aucune source indexee" meta="le manuel se charge..." />
              : sources.map((s, i) => (
                  <SourceLine key={s.name} name={s.name}
                    meta={`${s.chunks} fragments indexes`} lit={i === 0} />
                ))}
            <Uploader onUpload={upload} busy={busy} error={error} />
            <SiteCrawler onDone={refresh} />
          </ParchPanel>
          <div style={{ height: 18 }} />
          <SigilPanel sigils={SIGILS} />
        </aside>
      </div>
    </div>
  );
}

const TYPE_LABEL = { spell: "Sort", monster: "Monstre", equipment: "Equipement", magic: "Objet magique", condition: "Condition" };

/** Recherche instantanee dans les regles (SRD), sans LLM. */
function RulesSearch() {
  const [index, setIndex] = useState([]);
  const [q, setQ] = useState("");
  const [detail, setDetail] = useState(null);
  useEffect(() => { getReferenceIndex().then((d) => setIndex(d.items || [])); }, []);
  const results = q.trim().length >= 2
    ? index.filter((i) => i.name.toLowerCase().includes(q.toLowerCase())).slice(0, 30) : [];
  const open = async (it) => {
    setDetail({ loading: true, name: it.name });
    setDetail(await getReferenceDetail(it.type, it.index) || { name: it.name, desc: "Indisponible (hors ligne)." });
  };
  return (
    <div style={{ padding: "0 48px 6px", position: "relative", zIndex: 20 }}>
      <input value={q} onChange={(e) => setQ(e.target.value)}
        placeholder="Recherche rapide : sort, monstre, equipement, condition..."
        style={{ width: "100%", padding: "10px 14px", borderRadius: 4, background: T.ink, border: `1px solid ${T.line}`,
          color: T.parch, fontFamily: BODY, fontSize: 15, outline: "none", boxSizing: "border-box" }} />
      {results.length > 0 && (
        <div style={{ position: "absolute", left: 48, right: 48, top: "100%", maxHeight: 320, overflowY: "auto",
          background: T.panel2, border: `1px solid ${T.line}`, borderRadius: 5, boxShadow: "0 16px 40px rgba(0,0,0,.55)", padding: 6 }}>
          {results.map((it) => (
            <button key={`${it.type}.${it.index}`} onClick={() => open(it)} style={{
              width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8,
              padding: "8px 12px", borderRadius: 3, cursor: "pointer", background: "transparent", border: "none",
              color: T.mist, fontFamily: BODY, fontSize: 14.5, textAlign: "left" }}>
              <span>{it.name}</span>
              <span style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1, textTransform: "uppercase", color: T.goldDim }}>{TYPE_LABEL[it.type] || it.type}</span>
            </button>
          ))}
        </div>
      )}
      {detail && (
        <Modal title={detail.name || "Regle"} width={560} onClose={() => setDetail(null)}>
          {detail.loading ? <div style={{ fontFamily: BODY, color: T.mistDim, fontStyle: "italic" }}>Chargement...</div> : (
            <div>
              {(detail.meta || []).length > 0 && (
                <div style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 0.5, color: T.goldDim, marginBottom: 8 }}>
                  {detail.meta.join("  ·  ")}
                </div>
              )}
              <p style={{ fontFamily: BODY, fontSize: 15, color: T.mist, lineHeight: 1.65, whiteSpace: "pre-wrap", margin: 0 }}>{detail.desc}</p>
            </div>
          )}
        </Modal>
      )}
    </div>
  );
}

/** Bouton d'ajout de grimoire (upload PDF / txt / md), stylise sur parchemin. */
function Uploader({ onUpload, busy, error }) {
  const ref = useRef(null);
  const pick = (e) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    e.target.value = "";
  };
  return (
    <div style={{ marginTop: 14, borderTop: "1px solid rgba(80,60,30,.25)", paddingTop: 12 }}>
      <input ref={ref} type="file" accept=".pdf,.txt,.md" onChange={pick} style={{ display: "none" }} />
      <button onClick={() => ref.current?.click()} disabled={busy} style={{
        width: "100%", display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
        cursor: busy ? "wait" : "pointer", padding: "9px 12px", borderRadius: 3,
        background: "rgba(74,58,34,.10)", border: `1px solid ${T.goldDim}`, color: "#4A3a22",
        fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase", opacity: busy ? 0.6 : 1,
      }}>
        {busy ? <ScrollText size={14} /> : <Upload size={14} />}
        {busy ? "Inscription..." : "Ajouter un grimoire"}
      </button>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 12.5, color: "#6b5836", marginTop: 6, textAlign: "center" }}>
        PDF, txt ou md
      </div>
      {error && <div style={{ fontFamily: BODY, fontSize: 13, color: T.ember, marginTop: 4, textAlign: "center" }}>{error}</div>}
    </div>
  );
}

/** Aspiration d'un site web entier vers le RAG. */
function SiteCrawler({ onDone }) {
  const ref = useRef(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const go = async () => {
    const url = ref.current?.value?.trim();
    if (!url) return;
    setBusy(true); setMsg(null);
    try { const srcs = await crawlSite(url); setMsg(`Indexe (${srcs.length} source(s)).`); onDone?.(); ref.current.value = ""; }
    catch (e) { setMsg(e.message); } finally { setBusy(false); }
  };
  return (
    <div style={{ marginTop: 10, borderTop: "1px solid rgba(80,60,30,.25)", paddingTop: 10 }}>
      <div style={{ display: "flex", gap: 6 }}>
        <input ref={ref} placeholder="https://un-wiki-dnd..." style={{
          flex: 1, padding: "8px 10px", borderRadius: 3, background: "rgba(74,58,34,.10)",
          border: `1px solid ${T.goldDim}`, color: "#4A3a22", fontFamily: BODY, fontSize: 14, outline: "none", boxSizing: "border-box" }} />
        <button onClick={go} disabled={busy} style={{
          display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 12px", borderRadius: 3,
          cursor: busy ? "wait" : "pointer", background: "rgba(74,58,34,.10)", border: `1px solid ${T.goldDim}`,
          color: "#4A3a22", fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase", opacity: busy ? 0.6 : 1 }}>
          <Globe size={13} /> {busy ? "Aspiration..." : "Aspirer le site"}
        </button>
      </div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 12, color: "#6b5836", marginTop: 5 }}>
        Parcourt tout le site (meme domaine). Peut prendre du temps.
      </div>
      {msg && <div style={{ fontFamily: BODY, fontSize: 12.5, color: T.ember, marginTop: 4 }}>{msg}</div>}
    </div>
  );
}
