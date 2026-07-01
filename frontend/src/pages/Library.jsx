/** "Mon antre" : retrouve ses personnages (inventaire + bourse) et ses campagnes. */
import { useEffect, useState } from "react";
import { Shield, Swords, Trash2, ChevronRight, Plus, Coins, ScrollText, Flag, Archive } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { ScreenTitle, Divider, SectionLabel } from "../components/ornaments.jsx";
import { listCharacters, deleteCharacter, updateMoney } from "../api/characters.js";
import { listAssignments, fmtCoins } from "../api/forge.js";
import { listCampaigns, activateCampaign, endCampaign, deleteCampaign, getArchive } from "../api/campaigns.js";
import { activateElement, deactivateElement } from "../api/library.js";
import { useActiveCampaign } from "../campaign/ActiveCampaignContext.jsx";
import { Modal } from "../components/Modal.jsx";
import { CharacterSheet } from "../components/CharacterSheet.jsx";
import { ABILITIES, fmtMod } from "../lib/pointBuy.js";
import { computeSheet, fmtMod as fmtSheetMod } from "../lib/sheet.js";

export function Library({ setView, embedded = false, only }) {
  const { setActive } = useActiveCampaign();
  const showPersos = !only || only === "personnages";
  const showCamps = !only || only === "campagnes";
  const [characters, setCharacters] = useState([]);
  const [campaigns, setCampaigns] = useState([]);
  const [open, setOpen] = useState(null);
  const [inv, setInv] = useState({});
  const [sheet, setSheet] = useState(null);   // personnage dont la fiche est ouverte
  const [archive, setArchive] = useState(null); // archive de campagne affichee

  const refresh = () => {
    listCharacters().then(setCharacters).catch(() => {});
    listCampaigns().then(setCampaigns).catch(() => {});
  };
  useEffect(refresh, []);

  const enter = async (c) => {
    try { const s = await activateCampaign(c.id); setActive(s); } catch { setActive(c); }
    setView("gm");
  };
  const remove = async (id) => { await deleteCharacter(id); refresh(); };
  const toggleAventure = async (char, campaignId, inIt) => {
    try {
      if (inIt) await deactivateElement({ element_type: "character", element_id: char.id, campaign_id: campaignId });
      else await activateElement({ element_type: "character", element_id: char.id, campaign_id: campaignId });
      refresh();
    } catch { /* ignore */ }
  };
  const endCamp = async (c) => {
    if (!window.confirm(`Terminer "${c.name}" ? La campagne passe en archive ; les quetes (reussies/echouees) sont conservees.`)) return;
    try { const a = await endCampaign(c.id); setArchive(a); refresh(); } catch { /* ignore */ }
  };
  const delCamp = async (c) => {
    if (!window.confirm(`Supprimer definitivement "${c.name}" ? Les personnages seront detaches, les quetes et l'historique effaces. Action irreversible.`)) return;
    try { await deleteCampaign(c.id); refresh(); } catch { /* ignore */ }
  };
  const openArchive = async (c) => { const a = await getArchive(c.id); if (a) setArchive(a); };

  const body = (
    <>
      {showPersos && (<>
        <SectionLabel icon={Shield} text={`Personnages (${characters.length})`} />
        <button onClick={() => setView("character")} style={ghost}><Plus size={13} /> Nouveau personnage</button>

        {characters.length === 0 ? (
          <Empty text="Aucun personnage. Forgez votre premier heros." />
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12, marginTop: 12 }}>
            {characters.map((c) => (
              <div key={c.id} className="lift" style={card}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ fontFamily: ORNATE, fontSize: 18, color: T.parch }}>{c.name}</span>
                  <span style={{ fontFamily: DISPLAY, fontSize: 11, color: T.gold }}>Niv. {c.class_level}</span>
                </div>
                <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14.5, color: T.mistDim }}>
                  {c.race} {c.character_class}
                </div>
                {campaigns.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <span style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.mistDim }}>Aventures :</span>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 5 }}>
                      {campaigns.map((cp) => {
                        const inIt = (c.campaigns || []).includes(cp.id);
                        return (
                          <button key={cp.id} onClick={() => toggleAventure(c, cp.id, inIt)} style={{
                            padding: "4px 10px", borderRadius: 2, cursor: "pointer",
                            background: inIt ? "rgba(201,162,75,.16)" : "transparent",
                            border: `1px solid ${inIt ? T.gold : T.line}`, color: inIt ? T.goldBright : T.mistDim,
                            fontFamily: BODY, fontSize: 13 }} title={inIt ? "Quitter cette aventure" : "Rejoindre cette aventure"}>
                            {inIt ? "✦ " : "+ "}{cp.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
                <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
                  <button onClick={() => {
                    const next = open === c.id ? null : c.id;
                    setOpen(next);
                    if (next && !inv[c.id]) listAssignments("character", c.id).then((a) => setInv((m) => ({ ...m, [c.id]: a })));
                  }} style={link}>
                    {open === c.id ? "Masquer" : "Details"} <ChevronRight size={12} />
                  </button>
                  <button onClick={() => setSheet(c)} style={link}><ScrollText size={12} /> Fiche</button>
                  <button onClick={() => remove(c.id)} style={{ ...link, color: T.mistDim }}><Trash2 size={12} /> Supprimer</button>
                </div>
                {open === c.id && (
                  <div style={{ marginTop: 10, borderTop: `1px solid ${T.line}`, paddingTop: 10 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6 }}>
                      {ABILITIES.map(([k, l]) => (
                        <div key={k} style={{ textAlign: "center", fontFamily: BODY, fontSize: 13, color: T.mist }}>
                          <div style={{ fontFamily: DISPLAY, fontSize: 9, color: T.mistDim }}>{l}</div>
                          {c[k]} <span style={{ color: T.mistDim }}>({fmtMod(c[k])})</span>
                        </div>
                      ))}
                    </div>
                    {(() => { const sh = computeSheet(c); return (
                      <div style={{ display: "flex", gap: 14, marginTop: 8, flexWrap: "wrap", fontFamily: BODY, fontSize: 13.5, color: T.mist }}>
                        <span>CA <b style={{ color: T.parch }}>{sh.ac}</b></span>
                        <span>Initiative <b style={{ color: T.parch }}>{fmtSheetMod(sh.initiative)}</b></span>
                        <span>Maitrise <b style={{ color: T.parch }}>{fmtSheetMod(sh.profBonus)}</b></span>
                        <span>PV max <b style={{ color: T.parch }}>{c.max_hp || "—"}</b></span>
                      </div>
                    ); })()}
                    {c.description && <p style={{ fontFamily: BODY, fontSize: 14.5, color: T.mist, marginTop: 10, fontStyle: "italic" }}>{c.description}</p>}
                    {c.spells && c.spells.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim }}>Sorts</span>
                        <div style={{ fontFamily: BODY, fontSize: 14, color: T.mist, marginTop: 3 }}>
                          {c.spells.map((sp) => sp.replace(/-/g, " ")).join(", ")}
                        </div>
                      </div>
                    )}
                    <Purse character={c} onSaved={(updated) => setCharacters((list) => list.map((x) => x.id === updated.id ? { ...x, ...updated } : x))} />
                    <div style={{ marginTop: 10 }}>
                      <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim }}>Inventaire</span>
                      {(inv[c.id] && inv[c.id].length > 0) ? (
                        <div style={{ marginTop: 4 }}>
                          {inv[c.id].map((a) => {
                            const coins = fmtCoins(a.item);
                            return (
                              <div key={a.id} style={{ display: "flex", justifyContent: "space-between", fontFamily: BODY, fontSize: 14, color: T.mist, padding: "2px 0" }}>
                                <span>{a.item.name}{a.quantity > 1 ? ` ×${a.quantity}` : ""}</span>
                                <span style={{ color: T.goldDim, fontSize: 12.5 }}>{coins || a.item.value || ""}</span>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div style={{ fontFamily: BODY, fontSize: 14, color: T.mistDim, marginTop: 3, fontStyle: "italic" }}>Aucun objet. Forge et attribue depuis La Forge.</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </>)}

      {showCamps && (<>
        {showPersos && <div style={{ height: 24 }} />}
        <SectionLabel icon={Swords} text={`Campagnes (${campaigns.length})`} />
        <button onClick={() => setView("create")} style={ghost}><Plus size={13} /> Forger une campagne</button>

        {campaigns.length === 0 ? (
          <Empty text="Aucune campagne. Vous pouvez en creer une si vous etes Maitre du Jeu." />
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12, marginTop: 12 }}>
            {campaigns.map((c) => {
              const ended = c.status === "ended";
              const isDm = c.role === "dm";
              return (
              <div key={c.id} className="lift" style={{ ...card, opacity: ended ? 0.82 : 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 6 }}>
                  <span style={{ fontFamily: ORNATE, fontSize: 18, color: T.parch }}>{c.name}</span>
                  <div style={{ display: "flex", gap: 5 }}>
                    {ended && <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1, textTransform: "uppercase", color: T.mistDim, border: `1px solid ${T.line}`, borderRadius: 2, padding: "2px 7px" }}>Terminee</span>}
                    <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1, textTransform: "uppercase", color: T.gold, border: `1px solid ${T.goldDim}`, borderRadius: 2, padding: "2px 7px" }}>
                      {isDm ? "Maitre" : "Joueur"}
                    </span>
                  </div>
                </div>
                {c.description && <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14, color: T.mistDim, marginTop: 2 }}>{c.description}</div>}
                {!ended && (
                  <button onClick={() => enter(c)} className="glow" style={{ ...ghost, marginTop: 12 }}>
                    Entrer a la Table <ChevronRight size={13} />
                  </button>
                )}
                <div style={{ display: "flex", gap: 12, marginTop: 10, flexWrap: "wrap" }}>
                  <button onClick={() => openArchive(c)} style={link}><Archive size={12} /> Archive des quetes</button>
                  {isDm && !ended && <button onClick={() => endCamp(c)} style={link}><Flag size={12} /> Terminer</button>}
                  {isDm && <button onClick={() => delCamp(c)} style={{ ...link, color: T.mistDim }}><Trash2 size={12} /> Supprimer</button>}
                </div>
              </div>
              );
            })}
          </div>
        )}
      </>)}
    </>
  );

  const modals = (
    <>
      {archive && (
        <Modal title={`Archive — ${archive.name}`} width={620} onClose={() => setArchive(null)}>
          <ArchiveView archive={archive} />
        </Modal>
      )}
      {sheet && (
        <Modal title={`Fiche — ${sheet.name}`} width={940} onClose={() => setSheet(null)}>
          <CharacterSheet character={sheet}
            onSaved={(u) => { setCharacters((list) => list.map((x) => x.id === u.id ? { ...x, ...u } : x)); setSheet(u); }} />
        </Modal>
      )}
    </>
  );

  if (embedded) return (<>{body}{modals}</>);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Vos creations" title="Mon antre" />
      <div style={{ padding: "0 48px" }}><Divider /></div>
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px", maxWidth: 920, margin: "0 auto", width: "100%" }}>
        {body}
      </div>
      {modals}
    </div>
  );
}

const card = { padding: "14px 16px", borderRadius: 5, background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` };
const ghost = { display: "inline-flex", alignItems: "center", gap: 7, marginTop: 10, padding: "7px 13px", borderRadius: 3,
  background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
  fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" };
const link = { display: "inline-flex", alignItems: "center", gap: 4, background: "none", border: "none", cursor: "pointer",
  fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase", color: T.gold };

function ArchiveView({ archive }) {
  const Section = ({ title, items, color }) => (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase", color, marginBottom: 6 }}>{title} ({items.length})</div>
      {items.length === 0 ? <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14, color: T.mistDim }}>Aucune.</div>
        : items.map((q, i) => (
          <div key={i} style={{ padding: "8px 10px", marginBottom: 6, borderRadius: 4, background: T.ink, border: `1px solid ${T.line}` }}>
            <div style={{ fontFamily: ORNATE, fontSize: 15, color: T.parch }}>{q.title}</div>
            {q.objective && <div style={{ fontFamily: BODY, fontSize: 13.5, color: T.mistDim, fontStyle: "italic" }}>{q.objective}</div>}
            {q.reward && <div style={{ fontFamily: BODY, fontSize: 13, color: T.goldDim }}>Recompense : {q.reward}</div>}
          </div>
        ))}
    </div>
  );
  return (
    <div>
      <Section title="Quetes reussies" items={archive.won} color="#7bb37b" />
      <Section title="Quetes echouees" items={archive.failed} color={T.emberBright} />
    </div>
  );
}

function Empty({ text }) {
  return <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim, marginTop: 10 }}>{text}</div>;
}

/** Bourse editable d'un personnage (po / pa / pc). */
function Purse({ character, onSaved }) {
  const [m, setM] = useState({ gold: character.gold || 0, silver: character.silver || 0, copper: character.copper || 0 });
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const dirty = m.gold !== (character.gold || 0) || m.silver !== (character.silver || 0) || m.copper !== (character.copper || 0);
  const set = (k, v) => { setM((s) => ({ ...s, [k]: Math.max(0, Number(v) || 0) })); setSaved(false); };
  const save = async () => {
    setBusy(true);
    try { const updated = await updateMoney(character.id, m); onSaved(updated); setSaved(true); }
    catch { /* ignore */ } finally { setBusy(false); }
  };
  const coin = (label, k, color) => (
    <label style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <input type="number" min="0" value={m[k]} onChange={(e) => set(k, e.target.value)} style={purseInp} />
      <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color }}>{label}</span>
    </label>
  );
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <Coins size={13} color={T.gold} />
        <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim }}>Bourse</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 6, flexWrap: "wrap" }}>
        {coin("po", "gold", "#d9b65a")}
        {coin("pa", "silver", "#c2c8cf")}
        {coin("pc", "copper", "#c08457")}
        {dirty && (
          <button onClick={save} disabled={busy} style={{ padding: "5px 12px", borderRadius: 3, background: "transparent",
            border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 9,
            letterSpacing: 1.5, textTransform: "uppercase", opacity: busy ? 0.6 : 1 }}>Enregistrer</button>
        )}
        {saved && !dirty && <span style={{ fontFamily: BODY, fontSize: 12.5, color: T.gold }}>Enregistre ✓</span>}
      </div>
    </div>
  );
}

const purseInp = { width: 64, padding: "5px 8px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none" };
