/** L'Arsenal : catalogue d'objets D&D (SRD) + mes objets forges, attribuables aux PNJ/quetes. */
import { useEffect, useMemo, useState } from "react";
import { Sword, Gem, Coins, ChevronRight, UserPlus, Hammer } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { ScreenTitle, Divider } from "../components/ornaments.jsx";
import { Modal } from "../components/Modal.jsx";
import { getCatalog, getItemDetail, assignArsenal } from "../api/arsenal.js";
import { listItems, fmtCoins } from "../api/forge.js";
import { listCharacters } from "../api/characters.js";
import { getLibrary } from "../api/library.js";
import { listQuests } from "../api/campaigns.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";

function typeFromCategory(kind, category) {
  if (kind === "magic") return "magic";
  const c = (category || "").toLowerCase();
  if (c.includes("weapon") || c.includes("arme")) return "weapon";
  if (c.includes("armor") || c.includes("armure") || c.includes("shield") || c.includes("bouclier")) return "armor";
  return "misc";
}

export function Arsenal({ setView }) {
  const [tab, setTab] = useState("catalog");
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Armes, armures & objets" title="L'Arsenal" />
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 8 }}>
        <Tab on={tab === "catalog"} onClick={() => setTab("catalog")} icon={Sword} label="Catalogue D&D" />
        <Tab on={tab === "mine"} onClick={() => setTab("mine")} icon={Gem} label="Mes objets & tresors" />
      </div>
      <div style={{ padding: "0 48px" }}><Divider /></div>
      {tab === "catalog" ? <Catalog /> : <Mine setView={setView} />}
    </div>
  );
}

/* ----------------------------- Catalogue SRD ----------------------------- */
function Catalog() {
  const [cat, setCat] = useState({ available: false, categories: [] });
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => { getCatalog().then(setCat).finally(() => setLoading(false)); }, []);
  const cats = cat.categories || [];
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return cats.filter((c) => filter === "all" || c.key === filter)
      .map((c) => ({ ...c, items: c.items.filter((i) => !q || i.name.toLowerCase().includes(q)) }))
      .filter((c) => c.items.length > 0);
  }, [cats, filter, query]);

  const open = async (kind, index) => {
    setDetail({ loading: true });
    const d = await getItemDetail(kind, index);
    setDetail(d ? { ...d, kind } : { name: index, desc: "Indisponible.", kind });
  };

  if (loading) return <Note>Consultation du catalogue…</Note>;
  if (!cat.available) return <Note>Catalogue indisponible (le backend n'a pas pu joindre l'API D&D 5e). Réessaie quand la connexion est rétablie.</Note>;

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px", maxWidth: 980, margin: "0 auto", width: "100%" }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
        <Chip on={filter === "all"} onClick={() => setFilter("all")}>Tout</Chip>
        {cats.map((c) => <Chip key={c.key} on={filter === c.key} onClick={() => setFilter(c.key)}>{c.label} ({c.items.length})</Chip>)}
        <div style={{ flex: 1 }} />
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Rechercher…" style={search} />
      </div>

      {visible.map((c) => (
        <div key={c.key} style={{ marginBottom: 22 }}>
          <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold, margin: "6px 0 10px" }}>{c.label}</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 8 }}>
            {c.items.map((i) => (
              <button key={i.index} onClick={() => open(c.kind, i.index)} className="lift" style={itemBtn}>
                <span style={{ fontFamily: BODY, fontSize: 15, color: T.parch }}>{i.name}</span>
                <ChevronRight size={14} color={T.goldDim} />
              </button>
            ))}
          </div>
        </div>
      ))}

      {detail && (
        <Modal title={detail.name || "Objet"} width={560} onClose={() => setDetail(null)}>
          {detail.loading ? <Note>Chargement…</Note> : (
            <>
              <Detail d={detail} />
              <AssignBox item={{ name: detail.name, item_type: typeFromCategory(detail.kind, detail.category), value: detail.cost || null, description: (detail.desc || "").slice(0, 800) || null }} />
            </>
          )}
        </Modal>
      )}
    </div>
  );
}

function Detail({ d }) {
  const line = (label, val) => val ? (
    <div style={{ display: "flex", gap: 8, fontFamily: BODY, fontSize: 14.5, color: T.mist, padding: "3px 0" }}>
      <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1, textTransform: "uppercase", color: T.goldDim, minWidth: 100 }}>{label}</span>
      <span>{val}</span>
    </div>
  ) : null;
  return (
    <div>
      {line("Catégorie", d.category)}
      {line("Rareté", d.rarity)}
      {line("Coût", d.cost)}
      {line("Poids", d.weight)}
      {line("Dégâts", d.damage)}
      {line("Classe d'armure", d.armor_class)}
      {line("Propriétés", (d.properties || []).join(", "))}
      {d.desc && <p style={{ fontFamily: BODY, fontSize: 15, color: T.mist, lineHeight: 1.6, marginTop: 10, whiteSpace: "pre-wrap" }}>{d.desc}</p>}
    </div>
  );
}

/* ----------------------------- Mes objets -------------------------------- */
function Mine({ setView }) {
  const [items, setItems] = useState(null);
  const [assign, setAssign] = useState(null);
  useEffect(() => { listItems().then(setItems); }, []);
  if (items === null) return <Note>Chargement…</Note>;
  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px", maxWidth: 980, margin: "0 auto", width: "100%" }}>
      <button onClick={() => setView && setView("forge")} style={{ ...miniGhost, marginBottom: 14 }}><Hammer size={13} /> Forger un objet / tresor</button>
      {items.length === 0 ? (
        <Note>Aucun objet forgé. Utilise « Forger un objet / trésor » ci-dessus.</Note>
      ) : (
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
        {items.map((i) => (
          <div key={i.id} style={card}>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              {i.item_type === "treasure" ? <Coins size={14} color={T.gold} /> : <Gem size={14} color={T.gold} />}
              <span style={{ fontFamily: ORNATE, fontSize: 16, color: T.parch }}>{i.name}</span>
            </div>
            <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1, textTransform: "uppercase", color: T.goldDim, marginTop: 3 }}>
              {[i.item_type, i.rarity, fmtCoins(i) || i.value].filter(Boolean).join(" · ")}
            </div>
            {i.description && <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: T.mistDim, marginTop: 4 }}>{i.description}</div>}
            <button onClick={() => setAssign(i)} style={{ ...miniGhost, marginTop: 8 }}><UserPlus size={12} /> Attribuer</button>
          </div>
        ))}
      </div>
      )}
      {assign && (
        <Modal title={`Attribuer — ${assign.name}`} width={460} onClose={() => setAssign(null)}>
          <AssignBox item={{ name: assign.name, item_type: assign.item_type, value: assign.value || null, description: assign.description || null }} />
        </Modal>
      )}
    </div>
  );
}

/* ------------------------- Attribution PNJ/Quete ------------------------- */
function AssignBox({ item }) {
  const { active } = useActiveCampaign();
  const [type, setType] = useState("npc");
  const [npcs, setNpcs] = useState([]);
  const [quests, setQuests] = useState([]);
  const [chars, setChars] = useState([]);
  const [target, setTarget] = useState("");
  const [msg, setMsg] = useState(null);

  useEffect(() => { getLibrary().then((l) => setNpcs(l.npcs || [])); listCharacters().then(setChars); }, []);
  useEffect(() => { if (active?.id) listQuests(active.id).then(setQuests); else setQuests([]); }, [active?.id]);

  const list = type === "npc" ? npcs : type === "character" ? chars : quests;
  const submit = async () => {
    setMsg(null);
    if (!target) { setMsg("Choisis une cible."); return; }
    try {
      await assignArsenal({ name: item.name, item_type: item.item_type || "misc", value: item.value || null, description: item.description || null, target_type: type, target_id: Number(target) });
      setMsg("Attribué ✓"); setTarget("");
    } catch (e) { setMsg(e.message); }
  };

  return (
    <div style={{ marginTop: 14, paddingTop: 12, borderTop: `1px solid ${T.line}` }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim, marginBottom: 6 }}>Attribuer à</div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <select value={type} onChange={(e) => { setType(e.target.value); setTarget(""); }} style={{ ...sel, width: 130 }}>
          <option value="npc">PNJ</option>
          <option value="character">Personnage</option>
          <option value="quest">Quête</option>
        </select>
        <select value={target} onChange={(e) => setTarget(e.target.value)} style={{ ...sel, flex: 1, minWidth: 160 }}>
          <option value="">{type === "quest" && !active?.id ? "Active une campagne…" : "Cible…"}</option>
          {list.map((t) => <option key={t.id} value={t.id}>{t.name || t.title}</option>)}
        </select>
        <button onClick={submit} className="glow" style={miniGhost}><UserPlus size={12} /> Attribuer</button>
      </div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 12.5, color: T.mistDim, marginTop: 6 }}>
        Un même objet peut être attribué à plusieurs PNJ ou personnages.
      </div>
      {msg && <div style={{ fontFamily: BODY, fontSize: 13, color: T.gold, marginTop: 6 }}>{msg}</div>}
    </div>
  );
}

/* -------------------------------- UI bits -------------------------------- */
function Tab({ on, onClick, icon: Icon, label }) {
  return (
    <button onClick={onClick} style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "8px 16px", borderRadius: 3, cursor: "pointer",
      background: on ? "rgba(201,162,75,.12)" : "transparent", border: `1px solid ${on ? T.gold : T.line}`, color: on ? T.gold : T.mistDim,
      fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" }}>
      <Icon size={13} /> {label}
    </button>
  );
}
function Chip({ on, onClick, children }) {
  return (
    <button onClick={onClick} style={{ padding: "6px 13px", borderRadius: 3, cursor: "pointer",
      background: on ? "rgba(201,162,75,.12)" : "transparent", border: `1px solid ${on ? T.gold : T.line}`, color: on ? T.gold : T.mistDim,
      fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase" }}>{children}</button>
  );
}
const Note = ({ children }) => <div style={{ fontFamily: BODY, fontSize: 15, fontStyle: "italic", color: T.mistDim, padding: "16px 48px", maxWidth: 640, lineHeight: 1.6 }}>{children}</div>;
const itemBtn = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, padding: "11px 13px", borderRadius: 4, cursor: "pointer",
  background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}`, textAlign: "left" };
const card = { padding: "14px 16px", borderRadius: 5, background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` };
const search = { padding: "8px 12px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none", minWidth: 200 };
const sel = { padding: "8px 10px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none", cursor: "pointer", boxSizing: "border-box" };
const miniGhost = { display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 13px", borderRadius: 3, background: "transparent",
  border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" };
