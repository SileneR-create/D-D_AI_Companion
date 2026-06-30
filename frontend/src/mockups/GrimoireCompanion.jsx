import { useState } from "react";
import {
  Feather, Swords, Scroll, Send, Compass, Flame, BookOpen,
  Shield, Sparkles, Gem, Moon, ChevronRight
} from "lucide-react";

/* ============================================================ THEME ===== */
const T = {
  void: "#0D0A12",
  ink: "#141019",
  panel: "#1B1422",
  panel2: "#221A2B",
  gold: "#C9A24B",
  goldBright: "#E7C972",
  goldDim: "#7E6531",
  ember: "#A6402E",
  emberBright: "#C7592F",
  parch: "#E8DCC0",
  parch2: "#DBCBA4",
  inkOnParch: "#2A2017",
  mist: "#C7BCA0",
  mistDim: "#8E826A",
  line: "#3A2F44",
};
const DISPLAY = "'Cinzel', 'Trajan Pro', Georgia, serif";
const ORNATE = "'Cinzel Decorative', 'Cinzel', Georgia, serif";
const BODY = "'EB Garamond', 'Garamond', 'Times New Roman', serif";

/* ============================================================ ROOT ====== */
export default function GrimoireCompanion() {
  const [view, setView] = useState("rules");
  return (
    <div style={{ position: "relative", minHeight: "100vh", background: T.void, color: T.mist, fontFamily: BODY, overflow: "hidden" }}>
      <FontImport />
      <Atmosphere />
      <div style={{ position: "relative", display: "flex", minHeight: "100vh" }}>
        <Rail view={view} setView={setView} />
        <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          {view === "home" && <Threshold setView={setView} />}
          {view === "rules" && <Grimoire />}
          {view === "gm" && <Table />}
        </main>
      </div>
    </div>
  );
}

function FontImport() {
  return (
    <style>{`@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Cinzel+Decorative:wght@700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');
      *::selection{ background:${T.gold}; color:${T.void}; }
      .lift{ transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
      .lift:hover{ transform: translateY(-4px); border-color:${T.gold}!important; box-shadow:0 16px 40px rgba(0,0,0,.55); }
      .glow:hover{ color:${T.goldBright}!important; }
      input::placeholder{ color:${T.mistDim}; font-style:italic; }
    `}</style>
  );
}

/* ===================================================== ATMOSPHERE ======= */
function Atmosphere() {
  const noise =
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='200' height='200' filter='url(%23n)' opacity='0.5'/></svg>\")";
  return (
    <>
      <div style={{ position: "absolute", inset: 0, background:
        `radial-gradient(120% 80% at 50% -10%, #241a30 0%, ${T.void} 60%)` }} />
      <div style={{ position: "absolute", inset: 0, background:
        "radial-gradient(100% 100% at 50% 120%, rgba(166,64,46,.18), transparent 55%)" }} />
      <div style={{ position: "absolute", inset: 0, backgroundImage: noise, opacity: 0.04, mixBlendMode: "overlay" }} />
      {/* constellations */}
      <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0.5 }}>
        {[[120,90],[300,60],[480,140],[760,80],[920,180],[1180,70],[1040,260],[200,300]].map(([x,y],i)=>(
          <circle key={i} cx={x} cy={y} r={i%3===0?1.6:1} fill={T.gold} opacity={0.35} />
        ))}
      </svg>
    </>
  );
}

/* ========================================================== RAIL ======== */
function Rail({ view, setView }) {
  const items = [
    { id: "home", label: "Seuil", icon: Compass },
    { id: "rules", label: "Grimoire", icon: BookOpen },
    { id: "gm", label: "La Table", icon: Swords },
  ];
  return (
    <aside style={{
      width: 96, display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "space-between", padding: "26px 0",
      background: `linear-gradient(180deg, ${T.ink}, ${T.void})`,
      borderRight: `1px solid ${T.line}`,
    }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 30 }}>
        <Crest size={52} />
        <nav style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {items.map(({ id, label, icon: Icon }) => {
            const on = view === id;
            return (
              <button key={id} onClick={() => setView(id)} title={label}
                style={{
                  width: 60, height: 60, borderRadius: 14, cursor: "pointer",
                  display: "flex", flexDirection: "column", alignItems: "center",
                  justifyContent: "center", gap: 3,
                  background: on ? "rgba(201,162,75,.12)" : "transparent",
                  border: `1px solid ${on ? T.gold : "transparent"}`,
                  color: on ? T.goldBright : T.mistDim,
                }}>
                <Icon size={19} strokeWidth={1.4} />
                <span style={{ fontFamily: DISPLAY, fontSize: 8, letterSpacing: 1.5, textTransform: "uppercase" }}>{label}</span>
              </button>
            );
          })}
        </nav>
      </div>
      <Flame size={18} strokeWidth={1.3} color={T.ember} />
    </aside>
  );
}

/* ========================================================= CREST ======== */
function Crest({ size = 120 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" fill="none">
      <defs>
        <radialGradient id="cg" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stopColor="#2A1E12" />
          <stop offset="100%" stopColor="#120D08" />
        </radialGradient>
      </defs>
      <circle cx="60" cy="60" r="56" fill="url(#cg)" stroke={T.gold} strokeWidth="1.4" />
      <circle cx="60" cy="60" r="48" stroke={T.goldDim} strokeWidth="0.8" />
      {Array.from({ length: 36 }).map((_, i) => {
        const a = (i / 36) * Math.PI * 2;
        const r1 = 48, r2 = i % 3 === 0 ? 44 : 46;
        return <line key={i} x1={60 + Math.cos(a) * r1} y1={60 + Math.sin(a) * r1}
          x2={60 + Math.cos(a) * r2} y2={60 + Math.sin(a) * r2} stroke={T.goldDim} strokeWidth="0.7" />;
      })}
      {/* dragon coiled */}
      <g stroke={T.gold} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <path d="M60 32 C40 34 30 52 38 70 C44 84 64 88 76 78" />
        <path d="M76 78 C86 70 86 56 78 50 C72 46 64 48 62 56" />
        {/* head */}
        <path d="M60 32 C57 28 58 24 63 23 C66 22 70 24 70 28 L66 30" fill={T.gold} />
        {/* wing */}
        <path d="M52 50 L40 40 L46 54 L38 52 L48 62" />
        {/* tail flick */}
        <path d="M62 56 C58 60 58 66 63 67" />
      </g>
      <circle cx="64.5" cy="27" r="1.1" fill={T.emberBright} />
    </svg>
  );
}

/* ====================================================== ORNAMENTS ======= */
function Filigree({ flip }) {
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none"
      style={{ transform: flip ? "scaleX(-1)" : "none", opacity: .8 }}>
      <path d="M2 2 H18 M2 2 V18 M2 2 C18 6 22 10 26 26 C30 30 36 30 40 26"
        stroke={T.gold} strokeWidth="1" fill="none" strokeLinecap="round" />
      <circle cx="40" cy="26" r="2" fill={T.gold} />
    </svg>
  );
}

function Divider({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14, margin: "4px 0 18px" }}>
      <span style={{ flex: 1, height: 1, background: `linear-gradient(90deg, transparent, ${T.goldDim})` }} />
      <Gem size={12} color={T.gold} strokeWidth={1.3} />
      {label && <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 3, textTransform: "uppercase", color: T.gold }}>{label}</span>}
      <Gem size={12} color={T.gold} strokeWidth={1.3} />
      <span style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${T.goldDim}, transparent)` }} />
    </div>
  );
}

function ScreenTitle({ eyebrow, title }) {
  return (
    <div style={{ textAlign: "center", padding: "26px 0 6px" }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 5, textTransform: "uppercase", color: T.gold }}>{eyebrow}</div>
      <h1 style={{ fontFamily: ORNATE, fontSize: 30, color: T.parch, margin: "6px 0 0", fontWeight: 700, letterSpacing: 1 }}>{title}</h1>
    </div>
  );
}

/* ======================================================= THRESHOLD ====== */
function Threshold({ setView }) {
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 40 }}>
      <Crest size={132} />
      <div style={{ fontFamily: DISPLAY, fontSize: 12, letterSpacing: 7, textTransform: "uppercase", color: T.gold, marginTop: 26 }}>
        Compagnon des Royaumes
      </div>
      <h1 style={{ fontFamily: ORNATE, fontSize: 52, color: T.parch, margin: "10px 0 4px", letterSpacing: 1 }}>
        D&amp;D AI Companion
      </h1>
      <p style={{ fontFamily: BODY, fontSize: 19, fontStyle: "italic", color: T.mist, maxWidth: 520, textAlign: "center" }}>
        Un oracle pour les règles de la cinquième édition, et une table de travail
        pour le Maître qui façonne les mondes.
      </p>

      <div style={{ display: "flex", gap: 28, marginTop: 44 }}>
        <Portal icon={BookOpen} title="Le Grimoire" sub="Consulter les règles" onClick={() => setView("rules")} />
        <Portal icon={Swords} title="La Table du Maître" sub="Mener la campagne" onClick={() => setView("gm")} />
      </div>

      <div style={{ display: "flex", gap: 30, marginTop: 48 }}>
        {["Savoir des règles", "Forge de personnages", "Trame des quêtes", "Conduite des combats"].map((t) => (
          <span key={t} style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.mistDim }}>{t}</span>
        ))}
      </div>
    </div>
  );
}

function Portal({ icon: Icon, title, sub, onClick }) {
  return (
    <button onClick={onClick} className="lift" style={{
      position: "relative", width: 300, padding: "34px 28px", cursor: "pointer", textAlign: "center",
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`,
      border: `1px solid ${T.line}`, borderRadius: 4, color: T.mist,
    }}>
      <span style={{ position: "absolute", top: 8, left: 8 }}><Filigree /></span>
      <span style={{ position: "absolute", top: 8, right: 8 }}><Filigree flip /></span>
      <Icon size={30} strokeWidth={1.2} color={T.gold} />
      <div style={{ fontFamily: ORNATE, fontSize: 22, color: T.parch, marginTop: 14 }}>{title}</div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 16, color: T.mistDim, marginTop: 4 }}>{sub}</div>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 18, fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: T.gold }}>
        Franchir <ChevronRight size={13} />
      </div>
    </button>
  );
}

/* ======================================================== GRIMOIRE ====== */
function Grimoire() {
  const [msgs, setMsgs] = useState([
    { who: "seeker", text: "Comment se résout une attaque d'opportunité ?" },
    {
      who: "oracle",
      text: "Lorsqu'une créature hostile quitte volontairement votre allonge, elle s'expose. Vous dépensez votre réaction pour porter une unique attaque de mêlée contre elle. Votre action demeure intacte ; mais vous ne disposez que d'une réaction par round.",
      source: "basic-rules-fr.pdf — folio 74",
      sigil: "search_all_categories",
    },
  ]);
  const [input, setInput] = useState("");
  const send = () => {
    if (!input.trim()) return;
    setMsgs((m) => [...m, { who: "seeker", text: input },
      { who: "oracle", pending: true, source: "basic-rules-fr.pdf", sigil: "search_all_categories" }]);
    setInput("");
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Consultation des règles" title="Le Grimoire" />
      <div style={{ padding: "0 48px" }}><Divider label="Cinquième Édition" /></div>

      <div style={{ flex: 1, display: "flex", gap: 28, padding: "4px 48px 0", minHeight: 0 }}>
        {/* Conversation */}
        <div style={{ flex: 1, overflowY: "auto", paddingRight: 6, minWidth: 0 }}>
          {msgs.map((m, i) => <Exchange key={i} m={m} />)}
        </div>

        {/* Marge / parchemin */}
        <aside style={{ width: 270, flexShrink: 0 }}>
          <ParchPanel title="Sources convoquées">
            <SourceLine name="basic-rules-fr.pdf" meta="Règles de base — actif" lit />
            <SourceLine name="Manuel des Monstres" meta="312 fragments indexés" />
          </ParchPanel>
          <div style={{ height: 18 }} />
          <SigilPanel />
        </aside>
      </div>

      <Inkwell value={input} setValue={setInput} onSend={send} placeholder="Formulez votre question à l'Oracle..." />
    </div>
  );
}

function Exchange({ m }) {
  const oracle = m.who === "oracle";
  return (
    <div style={{ marginBottom: 26 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        {oracle ? <Sparkles size={14} color={T.gold} strokeWidth={1.4} /> : <Feather size={14} color={T.mistDim} strokeWidth={1.4} />}
        <span style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: oracle ? T.gold : T.mistDim }}>
          {oracle ? "L'Oracle" : "Le Chercheur"}
        </span>
        <span style={{ flex: 1, height: 1, background: T.line }} />
      </div>
      <p style={{
        fontFamily: BODY, fontSize: 18, lineHeight: 1.7, margin: 0,
        color: oracle ? T.parch : T.mist, fontStyle: oracle ? "normal" : "italic",
        paddingLeft: 24, borderLeft: `2px solid ${oracle ? T.goldDim : T.line}`,
      }}>
        {m.pending ? <span style={{ color: T.mistDim }}>L'Oracle consulte les pages anciennes...</span> : m.text}
      </p>
      {oracle && !m.pending && (
        <div style={{ display: "flex", gap: 16, marginTop: 10, paddingLeft: 26 }}>
          <Tag icon={BookOpen} text={m.source} />
          <Tag icon={Gem} text={m.sigil} accent />
        </div>
      )}
    </div>
  );
}

function Tag({ icon: Icon, text, accent }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase", color: accent ? T.gold : T.mistDim }}>
      <Icon size={12} strokeWidth={1.4} /> {text}
    </span>
  );
}

function ParchPanel({ title, children }) {
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

function SourceLine({ name, meta, lit }) {
  return (
    <div style={{ padding: "9px 0", borderTop: "1px solid rgba(80,60,30,.25)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: BODY, fontSize: 16, fontWeight: 600, color: T.inkOnParch }}>
        <Scroll size={14} strokeWidth={1.5} color={lit ? T.ember : "#6b5836"} /> {name}
      </div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: "#6b5836", marginLeft: 22 }}>{meta}</div>
    </div>
  );
}

function SigilPanel() {
  const sigils = ["search_all_categories", "filter_spells_by_level", "find_monsters_by_challenge_rating", "generate_treasure_hoard"];
  return (
    <div style={{ padding: "16px 18px", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3 }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold, marginBottom: 10 }}>Sceaux du serveur</div>
      {sigils.map((s) => (
        <div key={s} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0", fontFamily: BODY, fontSize: 15, color: T.mist }}>
          <Gem size={11} color={T.goldDim} strokeWidth={1.4} /> {s}
        </div>
      ))}
    </div>
  );
}

/* ========================================================= LA TABLE ===== */
function Table() {
  const [msgs, setMsgs] = useState([
    { who: "oracle", text: "Que la Session Zéro commence. Quel nom portera votre campagne ?" },
    { who: "seeker", text: "Les Ombres de Néverwinter." },
    { who: "oracle", text: "Un nom qui appelle la brume. En une phrase, quelle en est l'âme, le thème qui la hantera ?" },
  ]);
  const [input, setInput] = useState("");
  const send = () => { if (!input.trim()) return; setMsgs((m)=>[...m,{who:"seeker",text:input}]); setInput(""); };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Les Ombres de Néverwinter" title="La Table du Maître" />
      <div style={{ padding: "0 48px" }}><Divider label="Campagne active" /></div>

      <div style={{ flex: 1, display: "flex", gap: 28, padding: "4px 48px 0", minHeight: 0 }}>
        <div style={{ flex: 1, overflowY: "auto", minWidth: 0 }}>
          {msgs.map((m, i) => <Exchange key={i} m={m} />)}
        </div>

        <aside style={{ width: 320, flexShrink: 0, overflowY: "auto" }}>
          <Ledger />
        </aside>
      </div>

      <Inkwell value={input} setValue={setInput} onSend={send} placeholder="Répondez au Maître du Jeu..." />
    </div>
  );
}

function Ledger() {
  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 18 }}>
        {[["II","Compagnons"],["IV","Ames rencontrées"],["II","Serments en cours"],["VII","Veillées"]].map(([n,l])=>(
          <div key={l} style={{ padding: "12px 10px", textAlign: "center", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3 }}>
            <div style={{ fontFamily: ORNATE, fontSize: 22, color: T.gold }}>{n}</div>
            <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.mistDim }}>{l}</div>
          </div>
        ))}
      </div>

      <SectionLabel icon={Shield} text="Compagnons" />
      <HeroCard name="Lyra Vent-d'Acier" role="Rôdeuse — Demi-elfe" level="III" hp={24} hpMax={24} />
      <HeroCard name="Grom Poing-de-Fer" role="Barbare — Demi-orc" level="III" hp={28} hpMax={31} />

      <SectionLabel icon={Scroll} text="Serments & Quêtes" />
      <QuestRow name="La caravane disparue" status="En cours" lit />
      <QuestRow name="Le sceau brisé" status="Scellée" />

      <SectionLabel icon={Moon} text="Présages" />
      <div style={{ padding: "12px 14px", background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3, fontFamily: BODY, fontStyle: "italic", fontSize: 15.5, color: T.mist }}>
        « Une lune rousse se lève sur les remparts. Les morts ne dorment plus. »
      </div>
    </>
  );
}

function SectionLabel({ icon: Icon, text }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "20px 0 10px" }}>
      <Icon size={14} color={T.gold} strokeWidth={1.4} />
      <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold }}>{text}</span>
      <span style={{ flex: 1, height: 1, background: T.line }} />
    </div>
  );
}

function HeroCard({ name, role, level, hp, hpMax }) {
  const pct = Math.round((hp / hpMax) * 100);
  return (
    <div className="lift" style={{
      position: "relative", padding: "14px 16px", marginBottom: 10, borderRadius: 3,
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontFamily: ORNATE, fontSize: 17, color: T.parch }}>{name}</span>
        <span style={{ fontFamily: DISPLAY, fontSize: 11, color: T.gold }}>Niv. {level}</span>
      </div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim, marginTop: 2 }}>{role}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
        <div style={{ flex: 1, height: 5, borderRadius: 4, background: "#2a2030", overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: `linear-gradient(90deg, ${T.ember}, ${T.emberBright})` }} />
        </div>
        <span style={{ fontFamily: DISPLAY, fontSize: 11, color: T.mist }}>{hp}/{hpMax} PV</span>
      </div>
    </div>
  );
}

function QuestRow({ name, status, lit }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "11px 14px", marginBottom: 8, background: T.panel, border: `1px solid ${T.line}`, borderRadius: 3 }}>
      <span style={{ fontFamily: BODY, fontSize: 16.5, color: T.parch }}>{name}</span>
      <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", padding: "3px 9px", borderRadius: 2,
        color: lit ? T.goldBright : T.mistDim, border: `1px solid ${lit ? T.goldDim : T.line}` }}>{status}</span>
    </div>
  );
}

/* ========================================================= INKWELL ====== */
function Inkwell({ value, setValue, onSend, placeholder }) {
  return (
    <div style={{ padding: "18px 48px 26px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 18px", borderRadius: 4,
        background: T.ink, border: `1px solid ${T.line}`, boxShadow: "inset 0 1px 0 rgba(201,162,75,.08)" }}>
        <Feather size={18} color={T.gold} strokeWidth={1.3} />
        <input value={value} onChange={(e)=>setValue(e.target.value)} onKeyDown={(e)=>e.key==="Enter"&&onSend()}
          placeholder={placeholder}
          style={{ flex: 1, background: "transparent", border: "none", outline: "none",
            fontFamily: BODY, fontSize: 18, color: T.parch }} />
        <button onClick={onSend} className="glow" style={{
          display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer",
          padding: "8px 18px", borderRadius: 3, background: "transparent",
          border: `1px solid ${T.gold}`, color: T.gold,
          fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2, textTransform: "uppercase" }}>
          Invoquer <Send size={13} />
        </button>
      </div>
    </div>
  );
}
