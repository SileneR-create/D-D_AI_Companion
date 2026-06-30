/**
 * Formulaire de creation de personnage (page autonome + wizard).
 * Point-buy guide + calculateur de modificateurs + texte libre + selection de
 * sorts pour les classes lanceuses (charges depuis l'API selon classe/niveau).
 */
import { useEffect, useMemo, useState } from "react";
import { Minus, Plus, Wand2, Dice5, Sparkles, Info } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Divider } from "./ornaments.jsx";
import { ABILITIES, STANDARD_ARRAY, BUDGET, audit, fmtMod } from "../lib/pointBuy.js";
import { getClassSpells, getSpellDetail } from "../api/characters.js";
import { cantripsKnown, spellsKnown } from "../lib/spellLimits.js";
import { SUBCLASSES, BACKGROUNDS, backgroundByName } from "../lib/charOptions.js";
import { ABILITY_FR, SKILLS } from "../lib/sheet.js";
import { getAdvice } from "../lib/charAdvice.js";

const SKILL_LABEL = Object.fromEntries(SKILLS.map(([k, l]) => [k, l]));

const RACES = ["Humain", "Elfe", "Demi-elfe", "Nain", "Halfelin", "Demi-orc", "Tieffelin", "Drakeide", "Gnome"];
const CLASSES = ["Barbare", "Barde", "Clerc", "Druide", "Ensorceleur", "Guerrier", "Magicien", "Moine", "Paladin", "Rodeur", "Roublard", "Occultiste"];
const ALIGNMENTS = ["Loyal Bon", "Neutre Bon", "Chaotique Bon", "Loyal Neutre", "Neutre", "Chaotique Neutre", "Loyal Mauvais", "Neutre Mauvais", "Chaotique Mauvais"];

const empty = {
  name: "", race: RACES[0], character_class: CLASSES[0], class_level: 1,
  alignment: "", background: "", subclass: "", bg_primary: "", bg_secondary: "", description: "",
  strength: 8, dexterity: 8, constitution: 8, intelligence: 8, wisdom: 8, charisma: 8,
  gold: 0, silver: 0, copper: 0,
};

export function CharacterForm({ onSave, submitLabel = "Creer le personnage", campaignId = null }) {
  const [form, setForm] = useState(empty);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // --- Sorts ---
  const [spellData, setSpellData] = useState({ caster: false, max_spell_level: 0, spells: [] });
  const [selected, setSelected] = useState([]);
  const [query, setQuery] = useState("");
  const [loadingSpells, setLoadingSpells] = useState(false);
  const [details, setDetails] = useState({});
  const [openSpell, setOpenSpell] = useState(null);

  const showDetail = async (index) => {
    setOpenSpell((cur) => (cur === index ? null : index));
    if (!details[index]) {
      const d = await getSpellDetail(index);
      if (d) setDetails((m) => ({ ...m, [index]: d }));
    }
  };

  const level = Number(form.class_level) || 1;
  const scores = useMemo(
    () => Object.fromEntries(ABILITIES.map(([k]) => [k, Number(form[k]) || 8])),
    [form],
  );
  const a = audit(level, scores);

  const levelByIndex = useMemo(
    () => Object.fromEntries(spellData.spells.map((s) => [s.index, s.level])),
    [spellData],
  );
  const cantripCap = cantripsKnown(form.character_class, level);
  const spellCap = spellsKnown(form.character_class, level, scores);
  const selCantrips = selected.filter((i) => levelByIndex[i] === 0).length;
  const selSpells = selected.filter((i) => levelByIndex[i] > 0).length;

  // Recharge les sorts quand la classe ou le niveau change.
  useEffect(() => {
    let alive = true;
    setLoadingSpells(true);
    getClassSpells(form.character_class, level).then((d) => {
      if (!alive) return;
      setSpellData(d);
      // Conserve uniquement les sorts encore disponibles.
      const ok = new Set(d.spells.map((s) => s.index));
      setSelected((cur) => cur.filter((i) => ok.has(i)));
      setLoadingSpells(false);
    });
    return () => { alive = false; };
  }, [form.character_class, level]);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  const bg = backgroundByName(form.background);
  // Choix d'historique : auto-renseigne les 2 carac eligibles au bonus (+2 / +1).
  const chooseBackground = (name) => {
    const b = backgroundByName(name);
    setForm((f) => ({ ...f, background: name, bg_primary: b ? b.abilities[0] : "", bg_secondary: b ? b.abilities[1] : "" }));
  };
  // Bonus de caracteristiques de l'historique : +2 a la primaire, +1 a la secondaire (distinctes).
  const bonusMap = {};
  if (bg && form.bg_primary) bonusMap[form.bg_primary] = 2;
  if (bg && form.bg_secondary && form.bg_secondary !== form.bg_primary) bonusMap[form.bg_secondary] = 1;
  const finalScore = (k) => (Number(form[k]) || 8) + (bonusMap[k] || 0);
  // Reinitialise la sous-classe si elle ne correspond plus a la classe.
  useEffect(() => {
    setForm((f) => ((SUBCLASSES[f.character_class] || []).includes(f.subclass) ? f : { ...f, subclass: "" }));
  }, [form.character_class]);
  const bump = (k, d) => setForm((f) => {
    const v = Math.max(8, Math.min(20, (Number(f[k]) || 8) + d));
    return { ...f, [k]: v };
  });
  const applyStandard = () => setForm((f) => ({
    ...f, ...Object.fromEntries(ABILITIES.map(([k], i) => [k, STANDARD_ARRAY[i]])),
  }));
  const toggleSpell = (idx) => setSelected((cur) => {
    if (cur.includes(idx)) return cur.filter((i) => i !== idx);
    const isCantrip = levelByIndex[idx] === 0;
    const c = cur.filter((i) => levelByIndex[i] === 0).length;
    const s = cur.length - c;
    if (isCantrip && c >= cantripCap) return cur;       // plafond de sorts mineurs
    if (!isCantrip && s >= spellCap) return cur;        // plafond de sorts
    return [...cur, idx];
  });

  const submit = async () => {
    if (!form.name.trim()) { setError("Donnez un nom au personnage."); return; }
    if (!a.valid) { setError("Repartition des caracteristiques invalide pour ce niveau."); return; }
    setBusy(true); setError(null);
    try {
      await onSave({
        name: form.name, race: form.race, character_class: form.character_class,
        class_level: level, alignment: form.alignment || null, background: form.background || null,
        description: form.description || null, ...scores, spells: selected,
        subclass: form.subclass || null,
        skill_proficiencies: bg ? bg.skills : [],
        feats: bg ? bg.feat : null,
        other_proficiencies: bg ? `Outils : ${bg.tool}` : null,
        ability_bonus: bonusMap,
        gold: Number(form.gold) || 0, silver: Number(form.silver) || 0, copper: Number(form.copper) || 0,
        ...(campaignId ? { campaign_id: campaignId } : {}),
      });
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  // Regroupe les sorts par niveau pour l'affichage.
  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const g = {};
    for (const s of spellData.spells) {
      if (q && !s.name.toLowerCase().includes(q)) continue;
      (g[s.level] = g[s.level] || []).push(s);
    }
    return g;
  }, [spellData, query]);

  return (
    <div style={{ maxWidth: 720, margin: "0 auto" }}>
      <Panel>
        <Row>
          <div style={{ flex: 2 }}><Field label="Nom" value={form.name} onChange={(v) => set("name", v)} /></div>
          <div style={{ width: 90 }}><Field label="Niveau" type="number" value={form.class_level} onChange={(v) => set("class_level", v)} /></div>
        </Row>
        <Row>
          <div style={{ flex: 1 }}><Field label="Race" value={form.race} onChange={(v) => set("race", v)} options={RACES} /></div>
          <div style={{ flex: 1 }}><Field label="Classe" value={form.character_class} onChange={(v) => set("character_class", v)} options={CLASSES} /></div>
          <div style={{ flex: 1 }}><Field label="Alignement" value={form.alignment} onChange={(v) => set("alignment", v)} options={["", ...ALIGNMENTS]} /></div>
        </Row>
        <Row>
          <div style={{ flex: 1 }}><Field label="Sous-classe" value={form.subclass} onChange={(v) => set("subclass", v)} options={["", ...(SUBCLASSES[form.character_class] || [])]} /></div>
          <div style={{ flex: 1 }}><Field label="Historique" value={form.background} onChange={chooseBackground} options={["", ...BACKGROUNDS.map((b) => b.name)]} /></div>
        </Row>
        {bg && (
          <div style={{ padding: "12px 14px", borderRadius: 5, background: T.ink, border: `1px solid ${T.goldDim}`, marginBottom: 12 }}>
            <div style={{ fontFamily: BODY, fontSize: 14.5, color: T.mist }}>
              <b style={{ color: T.gold }}>{form.background}</b> — competences acquises :{" "}
              {bg.skills.map((k) => SKILL_LABEL[k]).join(", ")}
            </div>
            <div style={{ fontFamily: BODY, fontSize: 13.5, color: T.mistDim, marginTop: 3 }}>
              Outil : {bg.tool} · Don d'origine : {bg.feat}
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginTop: 8 }}>
              <span style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 1, textTransform: "uppercase", color: T.goldDim }}>Bonus de caracteristiques</span>
              <label style={{ fontFamily: BODY, fontSize: 13, color: T.mist }}>+2&nbsp;
                <select value={form.bg_primary} onChange={(e) => set("bg_primary", e.target.value)} style={miniSel}>
                  {bg.abilities.map((k) => <option key={k} value={k}>{ABILITY_FR[k]}</option>)}
                </select>
              </label>
              <label style={{ fontFamily: BODY, fontSize: 13, color: T.mist }}>+1&nbsp;
                <select value={form.bg_secondary} onChange={(e) => set("bg_secondary", e.target.value)} style={miniSel}>
                  {bg.abilities.filter((k) => k !== form.bg_primary).map((k) => <option key={k} value={k}>{ABILITY_FR[k]}</option>)}
                </select>
              </label>
            </div>
            <div style={{ fontFamily: BODY, fontSize: 13, color: T.gold, marginTop: 6 }}>
              Caracteristiques finales : {ABILITIES.map(([k]) => `${ABILITY_FR[k]} ${finalScore(k)}`).join(" · ")}
            </div>
          </div>
        )}

        {(() => { const adv = getAdvice(form.character_class); return adv ? (
          <div style={{ padding: "12px 14px", borderRadius: 5, background: "rgba(201,162,75,.07)", border: `1px solid ${T.goldDim}`, marginBottom: 12 }}>
            <div style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.gold, marginBottom: 4 }}>Conseils de l'assistant — {form.character_class}</div>
            <div style={{ fontFamily: BODY, fontSize: 14.5, color: T.mist }}>{adv.role}.</div>
            <div style={{ fontFamily: BODY, fontSize: 14, color: T.mist, marginTop: 4 }}>
              Caracteristiques a privilegier : <b style={{ color: T.parch }}>{adv.priorities.join(" > ")}</b>
              {" · "}Competences conseillees : <b style={{ color: T.parch }}>{adv.skills.join(", ")}</b>
            </div>
            <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: T.mistDim, marginTop: 4 }}>{adv.tip}</div>
          </div>
        ) : null; })()}

        <Divider label="Caracteristiques (point-buy de base)" />
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 12 }}>
          <button onClick={applyStandard} className="glow" style={btnGhost}>
            <Wand2 size={13} /> Tableau standard (15·14·13·12·10·8)
          </button>
          <Badge ok={a.budgetUsed <= BUDGET} label={`Points d'achat ${a.budgetUsed}/${BUDGET}`} />
          {a.asi > 0 && <Badge ok={a.bonusUsed <= a.asi} label={`Ameliorations ${a.bonusUsed}/${a.asi}`} />}
          {a.asi === 0 && <span style={hint}>Niveau 1 : pas de boost (max 15 / score)</span>}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
          {ABILITIES.map(([k, lbl]) => (
            <div key={k} style={{ background: T.panel, border: `1px solid ${T.line}`, borderRadius: 4, padding: "10px 12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 1.5, color: T.gold }}>{lbl}</span>
                <span style={{ fontFamily: BODY, fontSize: 13, color: T.mistDim }}>mod {fmtMod(scores[k])}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, marginTop: 6 }}>
                <Stepper onClick={() => bump(k, -1)} icon={Minus} />
                <span style={{ fontFamily: ORNATE, fontSize: 24, color: T.parch, minWidth: 28, textAlign: "center" }}>{scores[k]}</span>
                <Stepper onClick={() => bump(k, +1)} icon={Plus} />
              </div>
            </div>
          ))}
        </div>

        {/* --- Sorts (classes lanceuses) --- */}
        {spellData.caster && (
          <>
            <Divider label="Sorts" />
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
              <span style={{ fontFamily: BODY, fontSize: 14.5, color: T.mist }}>
                <Sparkles size={13} color={T.gold} style={{ verticalAlign: -2, marginRight: 6 }} />
                {spellData.max_spell_level === 0
                  ? "Cette classe ne lance pas encore de sorts a ce niveau."
                  : `Niveau max ${spellData.max_spell_level} — sorts mineurs ${selCantrips}/${cantripCap}, sorts ${selSpells}/${spellCap}`}
              </span>
              {spellData.spells.length > 0 && (
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Rechercher un sort..."
                  style={{ padding: "7px 10px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`,
                    color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none", width: 200 }} />
              )}
            </div>

            {loadingSpells ? (
              <div style={{ ...hint, marginTop: 10 }}>Consultation du grimoire des sorts…</div>
            ) : spellData.spells.length === 0 ? (
              <div style={{ ...hint, marginTop: 10 }}>
                Aucun sort disponible (ou liste injoignable hors ligne). Vous pourrez les ajouter plus tard.
              </div>
            ) : (
              <div style={{ marginTop: 10, maxHeight: 240, overflowY: "auto", paddingRight: 6 }}>
                {Object.keys(grouped).map(Number).sort((x, y) => x - y).map((lvl) => (
                  <div key={lvl} style={{ marginBottom: 10 }}>
                    <div style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase", color: T.goldDim, marginBottom: 6 }}>
                      {lvl === 0 ? "Sorts mineurs" : `Niveau ${lvl}`}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {grouped[lvl].map((s) => {
                        const on = selected.includes(s.index);
                        const isC = s.level === 0;
                        const full = !on && ((isC && selCantrips >= cantripCap) || (!isC && selSpells >= spellCap));
                        const open = openSpell === s.index;
                        const d = details[s.index];
                        return (
                          <div key={s.index}>
                            <div style={{ display: "flex", gap: 6 }}>
                              <button onClick={() => toggleSpell(s.index)} disabled={full} style={{
                                flex: 1, textAlign: "left", padding: "7px 10px", borderRadius: 3,
                                cursor: full ? "not-allowed" : "pointer", opacity: full ? 0.4 : 1,
                                background: on ? "rgba(201,162,75,.12)" : T.panel,
                                border: `1px solid ${on ? T.gold : T.line}`, color: on ? T.parch : T.mist,
                                fontFamily: BODY, fontSize: 14.5,
                              }}>
                                {on ? "✦ " : ""}{s.name}
                              </button>
                              <button onClick={() => showDetail(s.index)} title="Description" style={{
                                width: 34, borderRadius: 3, cursor: "pointer",
                                background: open ? "rgba(201,162,75,.12)" : "transparent",
                                border: `1px solid ${T.line}`, color: T.gold,
                                display: "flex", alignItems: "center", justifyContent: "center",
                              }}>
                                <Info size={13} />
                              </button>
                            </div>
                            {open && (
                              <div style={{ padding: "8px 12px", marginTop: 4, borderRadius: 3, background: T.ink, border: `1px solid ${T.line}` }}>
                                {!d ? (
                                  <span style={hint}>Consultation du grimoire des sorts…</span>
                                ) : (
                                  <>
                                    <div style={{ fontFamily: DISPLAY, fontSize: 9, letterSpacing: 0.5, color: T.goldDim, marginBottom: 5 }}>
                                      {[d.school, d.casting_time, d.range && ("Portee " + d.range), d.duration].filter(Boolean).join(" · ")}
                                    </div>
                                    <div style={{ fontFamily: BODY, fontSize: 14, color: T.mist, whiteSpace: "pre-wrap", maxHeight: 170, overflowY: "auto" }}>
                                      {d.desc}
                                    </div>
                                  </>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <Divider label="Bourse de depart" />
        <Row>
          <div style={{ flex: 1 }}><Field label="Pieces d'or (po)" type="number" value={form.gold} onChange={(v) => set("gold", v)} /></div>
          <div style={{ flex: 1 }}><Field label="Pieces d'argent (pa)" type="number" value={form.silver} onChange={(v) => set("silver", v)} /></div>
          <div style={{ flex: 1 }}><Field label="Pieces de cuivre (pc)" type="number" value={form.copper} onChange={(v) => set("copper", v)} /></div>
        </Row>

        <Divider label="Recit" />
        <Field label="Description libre (apparence, histoire, personnalite...)" textarea rows={5}
          value={form.description} onChange={(v) => set("description", v)} />

        {error && <div style={{ fontFamily: BODY, fontSize: 14, color: T.emberBright, marginTop: 10 }}>{error}</div>}
        <button onClick={submit} disabled={busy || !a.valid} className="glow" style={{
          ...btnGhost, marginTop: 16, padding: "11px 24px", opacity: busy || !a.valid ? 0.5 : 1,
          cursor: busy ? "wait" : a.valid ? "pointer" : "not-allowed",
        }}>
          <Dice5 size={14} /> {submitLabel}
        </button>
      </Panel>
    </div>
  );
}

/* --------------------------------- UI bits ------------------------------- */
const btnGhost = {
  display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 3,
  background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
  fontFamily: DISPLAY, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase",
};
const hint = { fontFamily: BODY, fontStyle: "italic", fontSize: 13.5, color: T.mistDim };
const miniSel = { padding: "4px 8px", borderRadius: 3, background: T.panel, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 13, cursor: "pointer", outline: "none" };

function Panel({ children }) {
  return (
    <div style={{ padding: "22px 24px", borderRadius: 6,
      background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}` }}>
      {children}
    </div>
  );
}
function Row({ children }) { return <div style={{ display: "flex", gap: 12 }}>{children}</div>; }
function Stepper({ onClick, icon: Icon }) {
  return (
    <button onClick={onClick} style={{ width: 26, height: 26, borderRadius: "50%", cursor: "pointer",
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "transparent", border: `1px solid ${T.goldDim}`, color: T.gold }}>
      <Icon size={13} />
    </button>
  );
}
function Badge({ ok, label }) {
  return (
    <span style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase",
      padding: "4px 10px", borderRadius: 2, border: `1px solid ${ok ? T.goldDim : T.ember}`,
      color: ok ? T.gold : T.emberBright }}>{label}</span>
  );
}
function Field({ label, value, onChange, options, textarea, rows = 2, type = "text" }) {
  const base = { width: "100%", marginTop: 5, padding: "10px 12px", borderRadius: 3,
    background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 16, outline: "none", boxSizing: "border-box" };
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 2, textTransform: "uppercase", color: T.mistDim }}>{label}</label>
      {options ? (
        <select value={value} onChange={(e) => onChange(e.target.value)} style={{ ...base, cursor: "pointer" }}>
          {options.map((o) => <option key={o} value={o}>{o || "—"}</option>)}
        </select>
      ) : textarea ? (
        <textarea value={value} onChange={(e) => onChange(e.target.value)} rows={rows} style={{ ...base, resize: "vertical" }} />
      ) : (
        <input type={type} value={value} onChange={(e) => onChange(e.target.value)} style={base} />
      )}
    </div>
  );
}
