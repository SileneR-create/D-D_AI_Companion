import { useState } from "react";
import {
  Home, Dices, ScrollText, Send, ThumbsUp, ThumbsDown, FileText,
  Upload, Sparkles, Swords, Map, Users, BookOpen, Plus, Circle,
  ChevronDown, Wand2, Skull, Scroll, Flame
} from "lucide-react";

// ===== Charte graphique (reprise de .streamlit/config.toml) =====
const C = {
  bg: "#F7F3E9",          // backgroundColor
  panel: "#EDE5D8",       // secondaryBackgroundColor
  brown: "#7A5C3E",       // primaryColor
  brownDark: "#5C4530",
  text: "#2B2B2B",        // textColor
  muted: "#8A7A66",
  gold: "#B8860B",
  burgundy: "#8B2E2E",
  line: "#D8CBB6",
  card: "#FBF8F1",
};

export default function DnDCompanionApp() {
  const [page, setPage] = useState("rules");

  return (
    <div
      className="w-full min-h-screen flex"
      style={{ background: C.bg, color: C.text, fontFamily: "ui-sans-serif, system-ui, sans-serif" }}
    >
      <Sidebar page={page} setPage={setPage} />
      <main className="flex-1 flex flex-col" style={{ minWidth: 0 }}>
        {page === "home" && <HomeScreen setPage={setPage} />}
        {page === "rules" && <RulesScreen />}
        {page === "gm" && <GamemasterScreen />}
      </main>
    </div>
  );
}

/* ---------------------------------------------------------------- Sidebar */
function Sidebar({ page, setPage }) {
  const nav = [
    { id: "home", label: "Accueil", icon: Home },
    { id: "rules", label: "Règles D&D", icon: ScrollText },
    { id: "gm", label: "Gamemaster", icon: Dices },
  ];
  return (
    <aside
      className="w-64 flex flex-col justify-between p-4"
      style={{ background: C.brownDark, color: "#F1E9DA" }}
    >
      <div>
        <div className="flex items-center gap-2 px-2 py-3 mb-4">
          <span className="text-2xl">🐉</span>
          <div className="leading-tight">
            <div className="font-bold text-base">D&D AI Companion</div>
            <div className="text-xs" style={{ color: "#C9B79A" }}>5e Édition · MCP + RAG</div>
          </div>
        </div>

        <nav className="space-y-1">
          {nav.map(({ id, label, icon: Icon }) => {
            const active = page === id;
            return (
              <button
                key={id}
                onClick={() => setPage(id)}
                className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors"
                style={{
                  background: active ? C.gold : "transparent",
                  color: active ? "#2B2B2B" : "#E8DDC8",
                  fontWeight: active ? 600 : 400,
                }}
              >
                <Icon size={18} />
                {label}
              </button>
            );
          })}
        </nav>

        <div className="mt-6 px-2">
          <div className="text-xs uppercase tracking-wide mb-2" style={{ color: "#C9B79A" }}>
            Modèle Ollama
          </div>
          <div
            className="flex items-center justify-between px-3 py-2 rounded-lg text-sm"
            style={{ background: "#6B5238" }}
          >
            <span>mistral:7b-instruct</span>
            <ChevronDown size={15} />
          </div>
        </div>
      </div>

      <div className="px-2 space-y-2 text-xs">
        <StatusRow label="Serveur MCP : DnD" ok />
        <StatusRow label="Gamemaster MCP" ok />
        <StatusRow label="Document RAG chargé" ok />
      </div>
    </aside>
  );
}

function StatusRow({ label, ok }) {
  return (
    <div className="flex items-center gap-2">
      <Circle size={9} fill={ok ? "#5FA86B" : "#B85C5C"} color={ok ? "#5FA86B" : "#B85C5C"} />
      <span style={{ color: "#D8CBB6" }}>{label}</span>
    </div>
  );
}

/* ------------------------------------------------------------------- Home */
function HomeScreen({ setPage }) {
  return (
    <div className="p-10 overflow-y-auto">
      <div
        className="rounded-2xl p-8 mb-8"
        style={{ background: `linear-gradient(135deg, ${C.brown}, ${C.brownDark})`, color: "#F6EFDF" }}
      >
        <div className="text-3xl font-bold mb-2">🧙‍♂️ Bienvenue, Maître du Jeu</div>
        <p className="max-w-xl" style={{ color: "#E3D7BE" }}>
          Votre assistant intelligent pour Donjons & Dragons 5e : réponses aux règles,
          gestion de campagne, génération d'aventures et de personnages.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        <HubCard
          icon={ScrollText} title="Règles D&D"
          desc="Posez vos questions sur les règles 5e. Réponses sourcées via RAG + base SRD."
          cta="Ouvrir les règles" onClick={() => setPage("rules")}
        />
        <HubCard
          icon={Dices} title="Gamemaster"
          desc="Créez et pilotez votre campagne : personnages, PNJ, quêtes, combats."
          cta="Ouvrir le Gamemaster" onClick={() => setPage("gm")}
        />
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { i: BookOpen, t: "Règles précises" },
          { i: Map, t: "Génération d'aventures" },
          { i: Users, t: "Création de personnages" },
          { i: Swords, t: "Gestion des combats" },
        ].map(({ i: Icon, t }) => (
          <div key={t} className="rounded-xl p-4 flex flex-col items-center text-center gap-2"
            style={{ background: C.card, border: `1px solid ${C.line}` }}>
            <Icon size={22} color={C.brown} />
            <span className="text-sm" style={{ color: C.text }}>{t}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HubCard({ icon: Icon, title, desc, cta, onClick }) {
  return (
    <div className="rounded-2xl p-6 flex flex-col" style={{ background: C.card, border: `1px solid ${C.line}` }}>
      <div className="flex items-center gap-3 mb-3">
        <div className="rounded-xl p-2" style={{ background: C.panel }}><Icon size={22} color={C.brown} /></div>
        <div className="text-lg font-semibold">{title}</div>
      </div>
      <p className="text-sm mb-5" style={{ color: C.muted }}>{desc}</p>
      <button onClick={onClick}
        className="mt-auto self-start px-4 py-2 rounded-lg text-sm font-medium"
        style={{ background: C.brown, color: "#F6EFDF" }}>
        {cta}
      </button>
    </div>
  );
}

/* ----------------------------------------------------------------- Rules */
function RulesScreen() {
  const [msgs, setMsgs] = useState([
    { role: "user", content: "Comment fonctionne un jet d'attaque d'opportunité ?" },
    {
      role: "assistant",
      content:
        "Une attaque d'opportunité se déclenche quand une créature hostile sort de votre allonge en se déplaçant volontairement. Vous utilisez votre réaction pour effectuer une seule attaque de mêlée contre elle. Cela ne consomme pas votre action, mais une seule réaction par round.",
      source: "basic-rules-fr.pdf · p. 74",
      tool: "DnD.search_all_categories",
    },
  ]);
  const [input, setInput] = useState("");

  const send = () => {
    if (!input.trim()) return;
    setMsgs((m) => [
      ...m,
      { role: "user", content: input },
      { role: "assistant", content: "…", source: "basic-rules-fr.pdf", tool: "DnD.search_all_categories", pending: true },
    ]);
    setInput("");
  };

  return (
    <div className="flex-1 flex" style={{ minHeight: 0 }}>
      {/* Colonne chat */}
      <div className="flex-1 flex flex-col" style={{ minWidth: 0 }}>
        <Header icon={ScrollText} title="Règles D&D"
          subtitle="Règles de la 5e édition · enrichies par vos documents" />

        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-5" style={{ background: C.bg }}>
          {msgs.map((m, i) => <Bubble key={i} m={m} />)}
        </div>

        <Composer value={input} setValue={setInput} onSend={send}
          placeholder="Posez une question sur les règles…" />
      </div>

      {/* Panneau documents RAG */}
      <aside className="w-72 p-5 border-l overflow-y-auto"
        style={{ background: C.panel, borderColor: C.line }}>
        <div className="text-sm font-semibold mb-3 flex items-center gap-2">
          <FileText size={16} color={C.brown} /> Documents (RAG)
        </div>
        <DocItem name="basic-rules-fr.pdf" meta="Règles de base · chargé" active />
        <DocItem name="monster-manual.pdf" meta="Importé · 312 extraits" />

        <button className="w-full mt-3 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm"
          style={{ border: `1.5px dashed ${C.brown}`, color: C.brown }}>
          <Upload size={15} /> Importer un PDF
        </button>

        <div className="text-xs uppercase tracking-wide mt-6 mb-2" style={{ color: C.muted }}>
          Outils MCP actifs
        </div>
        {["search_all_categories", "filter_spells_by_level", "find_monsters_by_challenge_rating", "generate_treasure_hoard"]
          .map((t) => (
            <div key={t} className="flex items-center gap-2 text-xs py-1" style={{ color: C.text }}>
              <Wand2 size={13} color={C.gold} /> {t}
            </div>
          ))}
      </aside>
    </div>
  );
}

function DocItem({ name, meta, active }) {
  return (
    <div className="rounded-lg p-3 mb-2"
      style={{ background: C.card, border: `1px solid ${active ? C.gold : C.line}` }}>
      <div className="text-sm font-medium flex items-center gap-2">
        <FileText size={14} color={C.brown} /> {name}
      </div>
      <div className="text-xs mt-1" style={{ color: C.muted }}>{meta}</div>
    </div>
  );
}

/* ------------------------------------------------------------ Gamemaster */
function GamemasterScreen() {
  const [msgs, setMsgs] = useState([
    { role: "assistant", content: "Commençons votre Session Zéro. Quel est le nom de votre campagne ?" },
    { role: "user", content: "Les Ombres de Néverwinter" },
    { role: "assistant", content: "Excellent. En une phrase, quel est le thème ou l'ambiance de cette campagne ?" },
  ]);
  const [input, setInput] = useState("");
  const send = () => {
    if (!input.trim()) return;
    setMsgs((m) => [...m, { role: "user", content: input }]);
    setInput("");
  };

  return (
    <div className="flex-1 flex" style={{ minHeight: 0 }}>
      <div className="flex-1 flex flex-col" style={{ minWidth: 0 }}>
        <Header icon={Dices} title="Gamemaster"
          subtitle="Campagne active : Les Ombres de Néverwinter" />
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-5" style={{ background: C.bg }}>
          {msgs.map((m, i) => <Bubble key={i} m={m} />)}
        </div>
        <Composer value={input} setValue={setInput} onSend={send}
          placeholder="Répondez au Maître du Jeu…" />
      </div>

      {/* Tableau de bord campagne */}
      <aside className="w-80 p-5 border-l overflow-y-auto" style={{ background: C.panel, borderColor: C.line }}>
        <CampaignCard />
        <PanelList title="Personnages" icon={Users} items={[
          { name: "Lyra Vent-d'Acier", meta: "Rôdeuse Demi-elfe · Niv. 3", hp: "24/24" },
          { name: "Grom Poing-de-Fer", meta: "Barbare Demi-orc · Niv. 3", hp: "31/31" },
        ]} />
        <PanelList title="PNJ" icon={Skull} items={[
          { name: "Maître Eldrin", meta: "Mage de la guilde" },
          { name: "Sombre Néra", meta: "Antagoniste · cachée" },
        ]} />
        <PanelList title="Quêtes" icon={Scroll} items={[
          { name: "La caravane disparue", meta: "En cours" },
          { name: "Le sceau brisé", meta: "Non commencée" },
        ]} />
        <button className="w-full mt-2 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium"
          style={{ background: C.brown, color: "#F6EFDF" }}>
          <Plus size={15} /> Ajouter un élément
        </button>
      </aside>
    </div>
  );
}

function CampaignCard() {
  return (
    <div className="rounded-xl p-4 mb-5"
      style={{ background: `linear-gradient(135deg, ${C.burgundy}, ${C.brownDark})`, color: "#F6EFDF" }}>
      <div className="flex items-center gap-2 text-xs mb-1" style={{ color: "#EBD9C2" }}>
        <Flame size={13} /> CAMPAGNE ACTIVE
      </div>
      <div className="text-lg font-bold leading-tight">Les Ombres de Néverwinter</div>
      <div className="flex gap-4 mt-3 text-xs">
        <Stat n="2" l="Joueurs" />
        <Stat n="4" l="PNJ" />
        <Stat n="2" l="Quêtes" />
        <Stat n="7" l="Séances" />
      </div>
    </div>
  );
}
function Stat({ n, l }) {
  return (
    <div className="text-center">
      <div className="text-lg font-bold leading-none">{n}</div>
      <div style={{ color: "#E3D2B8" }}>{l}</div>
    </div>
  );
}

function PanelList({ title, icon: Icon, items }) {
  return (
    <div className="mb-5">
      <div className="text-xs uppercase tracking-wide mb-2 flex items-center gap-2" style={{ color: C.muted }}>
        <Icon size={14} color={C.brown} /> {title}
      </div>
      <div className="space-y-2">
        {items.map((it) => (
          <div key={it.name} className="rounded-lg p-3 flex items-center justify-between"
            style={{ background: C.card, border: `1px solid ${C.line}` }}>
            <div>
              <div className="text-sm font-medium">{it.name}</div>
              <div className="text-xs" style={{ color: C.muted }}>{it.meta}</div>
            </div>
            {it.hp && (
              <span className="text-xs px-2 py-1 rounded-md"
                style={{ background: C.panel, color: C.burgundy, fontWeight: 600 }}>
                {it.hp} PV
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* --------------------------------------------------------- Shared pieces */
function Header({ icon: Icon, title, subtitle }) {
  return (
    <div className="px-8 py-4 border-b flex items-center gap-3"
      style={{ background: C.card, borderColor: C.line }}>
      <div className="rounded-xl p-2" style={{ background: C.panel }}><Icon size={20} color={C.brown} /></div>
      <div>
        <div className="text-lg font-bold leading-tight">{title}</div>
        <div className="text-xs" style={{ color: C.muted }}>{subtitle}</div>
      </div>
    </div>
  );
}

function Bubble({ m }) {
  const isUser = m.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className="max-w-[78%]">
        <div className="rounded-2xl px-4 py-3 text-sm leading-relaxed"
          style={{
            background: isUser ? C.brown : C.card,
            color: isUser ? "#F6EFDF" : C.text,
            border: isUser ? "none" : `1px solid ${C.line}`,
            borderTopRightRadius: isUser ? 4 : 16,
            borderTopLeftRadius: isUser ? 16 : 4,
          }}>
          {m.pending ? <span style={{ color: C.muted }}>⏳ Génération en cours…</span> : m.content}
        </div>

        {!isUser && !m.pending && (
          <div className="flex items-center gap-3 mt-1.5 px-1">
            {m.source && (
              <span className="text-xs flex items-center gap-1" style={{ color: C.muted }}>
                <BookOpen size={12} /> {m.source}
              </span>
            )}
            {m.tool && (
              <span className="text-xs flex items-center gap-1" style={{ color: C.gold }}>
                <Wand2 size={12} /> {m.tool}
              </span>
            )}
            <div className="flex items-center gap-1 ml-auto">
              <button className="p-1 rounded hover:opacity-70"><ThumbsUp size={13} color={C.muted} /></button>
              <button className="p-1 rounded hover:opacity-70"><ThumbsDown size={13} color={C.muted} /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Composer({ value, setValue, onSend, placeholder }) {
  return (
    <div className="px-8 py-4 border-t" style={{ background: C.card, borderColor: C.line }}>
      <div className="flex items-center gap-2 rounded-xl px-3 py-2"
        style={{ background: C.bg, border: `1px solid ${C.line}` }}>
        <Sparkles size={17} color={C.muted} />
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSend()}
          placeholder={placeholder}
          className="flex-1 bg-transparent outline-none text-sm"
          style={{ color: C.text }}
        />
        <button onClick={onSend}
          className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium"
          style={{ background: C.brown, color: "#F6EFDF" }}>
          <Send size={14} /> Envoyer
        </button>
      </div>
    </div>
  );
}
