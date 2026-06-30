/** La Forge : bibliotheque d'objets reutilisables + attribution a PJ/PNJ/quete. */
import { useEffect, useState } from "react";
import { Hammer, Trash2, Gem } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { ScreenTitle, Divider } from "../components/ornaments.jsx";
import { createItem, deleteItem, listItems, assignItem, fmtCoins } from "../api/forge.js";
import { listCharacters } from "../api/characters.js";
import { listNpcs, listQuests } from "../api/campaigns.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";

const TYPES = [["weapon", "Arme"], ["armor", "Armure"], ["magic", "Objet magique"], ["consumable", "Consommable"], ["treasure", "Tresor"], ["misc", "Divers"]];
const RARITIES = ["", "Commun", "Peu commun", "Rare", "Tres rare", "Legendaire"];

export function Forge() {
  const { active } = useActiveCampaign();
  const [items, setItems] = useState([]);
  const [f, setF] = useState({ name: "", item_type: "weapon", rarity: "", value: "", gp: "", sp: "", cp: "", description: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // Cibles d'attribution (campagne active)
  const [targets, setTargets] = useState({ character: [], npc: [], quest: [] });
  const [assign, setAssign] = useState({ item: "", type: "character", target: "" });
  const [msg, setMsg] = useState(null);

  const refreshItems = () => listItems().then(setItems).catch(() => {});
  useEffect(() => { refreshItems(); }, []);
  const loadTargets = () => {
    if (!active?.id) { setTargets({ character: [], npc: [], quest: [] }); return; }
    Promise.all([listCharacters(), listNpcs(active.id), listQuests(active.id)]).then(([chs, npcs, quests]) => {
      setTargets({ character: chs.filter((c) => c.campaign_id === active.id), npc: npcs, quest: quests });
    }).catch(() => {});
  };
  useEffect(() => { loadTargets(); }, [active]);
  // Recharge les cibles a chaque changement de type (PNJ/quete crees entre-temps).
  useEffect(() => { loadTargets(); }, [assign.type]);

  const set = (k, v) => setF((s) => ({ ...s, [k]: v }));

  const add = async () => {
    if (!f.name.trim()) return setError("Nom requis.");
    setBusy(true); setError(null);
    try {
      await createItem({
        ...f, rarity: f.rarity || null, value: f.value || null, description: f.description || null,
        gp: Number(f.gp) || 0, sp: Number(f.sp) || 0, cp: Number(f.cp) || 0,
      });
      setF({ name: "", item_type: "weapon", rarity: "", value: "", gp: "", sp: "", cp: "", description: "" }); refreshItems();
    }
    catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const remove = async (id) => { await deleteItem(id); refreshItems(); };

  const targetList = targets[assign.type] || [];
  const doAssign = async () => {
    setMsg(null);
    if (!assign.item || !assign.target) return setMsg("Choisissez un objet et une cible.");
    try {
      await assignItem({ item_id: Number(assign.item), target_type: assign.type, target_id: Number(assign.target) });
      const it = items.find((x) => String(x.id) === assign.item);
      setMsg(`${it?.name} attribue.`);
    } catch (e) { setMsg(e.message); }
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Atelier de l'artisan" title="La Forge" />
      <div style={{ padding: "0 48px" }}><Divider label="Armes & objets magiques" /></div>

      <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px", display: "flex", gap: 24, flexWrap: "wrap" }}>
        {/* Creation */}
        <div style={{ flex: "1 1 340px", maxWidth: 460 }}>
          <Panel title="Forger un objet">
            <input value={f.name} onChange={(e) => set("name", e.target.value)} placeholder="Nom de l'objet" style={inp} />
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <select value={f.item_type} onChange={(e) => set("item_type", e.target.value)} style={{ ...inp, marginTop: 0, cursor: "pointer" }}>
                {TYPES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
              <select value={f.rarity} onChange={(e) => set("rarity", e.target.value)} style={{ ...inp, marginTop: 0, cursor: "pointer" }}>
                {RARITIES.map((r) => <option key={r} value={r}>{r || "Rarete…"}</option>)}
              </select>
            </div>
            <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim, margin: "10px 0 4px" }}>Valeur</div>
            <div style={{ display: "flex", gap: 8 }}>
              <input type="number" min="0" value={f.gp} onChange={(e) => set("gp", e.target.value)} placeholder="po" style={{ ...inp, marginTop: 0 }} />
              <input type="number" min="0" value={f.sp} onChange={(e) => set("sp", e.target.value)} placeholder="pa" style={{ ...inp, marginTop: 0 }} />
              <input type="number" min="0" value={f.cp} onChange={(e) => set("cp", e.target.value)} placeholder="pc" style={{ ...inp, marginTop: 0 }} />
            </div>
            <textarea value={f.description} onChange={(e) => set("description", e.target.value)} rows={3} placeholder="Effet / description" style={{ ...inp, marginTop: 8, resize: "vertical" }} />
            {error && <Err>{error}</Err>}
            <button onClick={add} disabled={busy} className="glow" style={{ ...ghost, marginTop: 10, opacity: busy ? 0.6 : 1 }}><Hammer size={13} /> Forger</button>
          </Panel>

          {/* Attribution */}
          <div style={{ height: 16 }} />
          <Panel title="Attribuer un objet">
            {!active?.id ? (
              <Empty>Activez une campagne pour attribuer a un PJ, PNJ ou quete.</Empty>
            ) : (
              <>
                <select value={assign.item} onChange={(e) => setAssign((s) => ({ ...s, item: e.target.value }))} style={{ ...inp, cursor: "pointer" }}>
                  <option value="">Objet…</option>
                  {items.map((i) => <option key={i.id} value={i.id}>{i.name}</option>)}
                </select>
                <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                  <select value={assign.type} onChange={(e) => setAssign((s) => ({ ...s, type: e.target.value, target: "" }))} style={{ ...inp, marginTop: 0, width: 130, cursor: "pointer" }}>
                    <option value="character">Personnage</option>
                    <option value="npc">PNJ</option>
                    <option value="quest">Quete</option>
                  </select>
                  <select value={assign.target} onChange={(e) => setAssign((s) => ({ ...s, target: e.target.value }))} style={{ ...inp, marginTop: 0, flex: 1, cursor: "pointer" }}>
                    <option value="">Cible…</option>
                    {targetList.map((t) => <option key={t.id} value={t.id}>{t.name || t.title}</option>)}
                  </select>
                </div>
                <button onClick={doAssign} className="glow" style={{ ...ghost, marginTop: 10 }}><Gem size={13} /> Attribuer</button>
                {msg && <div style={{ fontFamily: BODY, fontSize: 13, color: T.gold, marginTop: 8 }}>{msg}</div>}
              </>
            )}
          </Panel>
        </div>

        {/* Bibliotheque */}
        <div style={{ flex: "1 1 340px" }}>
          <Panel title={`Bibliotheque (${items.length})`}>
            {items.length === 0 ? <Empty>Aucun objet forge.</Empty> : items.map((i) => (
              <div key={i.id} style={{ padding: "10px 12px", marginBottom: 8, borderRadius: 3, background: T.ink, border: `1px solid ${T.line}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontFamily: ORNATE, fontSize: 16, color: T.parch }}>{i.name}</span>
                  <button onClick={() => remove(i.id)} style={{ background: "none", border: "none", cursor: "pointer", color: T.mistDim, display: "flex" }}><Trash2 size={14} /></button>
                </div>
                <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1, textTransform: "uppercase", color: T.goldDim }}>
                  {[TYPES.find((t) => t[0] === i.item_type)?.[1], i.rarity, fmtCoins(i) || i.value].filter(Boolean).join(" · ")}
                </div>
                {i.description && <div style={{ fontFamily: BODY, fontSize: 13.5, color: T.mist, marginTop: 3 }}>{i.description}</div>}
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <div style={{ padding: "18px 20px", borderRadius: 6, background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` }}>
      <div style={{ fontFamily: ORNATE, fontSize: 17, color: T.parch, marginBottom: 8 }}>{title}</div>
      <Divider />
      {children}
    </div>
  );
}
const Empty = ({ children }) => <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14, color: T.mistDim }}>{children}</div>;
const Err = ({ children }) => <div style={{ fontFamily: BODY, fontSize: 13, color: T.emberBright, marginTop: 6 }}>{children}</div>;
const inp = { width: "100%", marginTop: 0, padding: "9px 11px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 15, outline: "none", boxSizing: "border-box" };
const ghost = { display: "inline-flex", alignItems: "center", gap: 6, padding: "9px 15px", borderRadius: 3, background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase" };
