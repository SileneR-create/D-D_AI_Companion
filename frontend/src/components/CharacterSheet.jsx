/**
 * Feuille de personnage complete (regles 2024).
 * Valeurs calculees (initiative, CA, maitrise, sauvegardes, competences, DD de sorts)
 * + champs editables (maitrises, PV, CA, vitesse, bourse, textes). Enregistrement via PATCH /sheet.
 */
import { useEffect, useMemo, useState } from "react";
import { Shield, Zap, Footprints, Award, Eye, Heart, Sparkles, Coins, Save, Star } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Divider } from "./ornaments.jsx";
import { updateSheet } from "../api/characters.js";
import { listAssignments, fmtCoins } from "../api/forge.js";
import {
  ABILITY_KEYS, ABILITY_FR, computeSheet, fmtMod, classData, spellAbility,
} from "../lib/sheet.js";

const EDITABLE = [
  "subclass", "size", "speed", "max_hp", "current_hp", "temp_hp", "hit_dice_used",
  "death_succ", "death_fail", "inspiration", "armor_class", "spell_ability",
  "save_proficiencies", "skill_proficiencies", "languages", "feats", "species_traits",
  "class_features", "other_proficiencies", "personality", "alignment", "background",
  "gold", "silver", "copper", "electrum", "platinum",
];

export function CharacterSheet({ character, onSaved }) {
  const init = useMemo(() => {
    const o = {};
    for (const k of EDITABLE) o[k] = character[k] ?? (Array.isArray(character[k]) ? [] : "");
    o.save_proficiencies = character.save_proficiencies || [];
    o.skill_proficiencies = character.skill_proficiencies || [];
    o.inspiration = !!character.inspiration;
    return o;
  }, [character]);

  const [f, setF] = useState(init);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [inv, setInv] = useState([]);

  useEffect(() => { listAssignments("character", character.id).then(setInv).catch(() => {}); }, [character.id]);

  const set = (k, v) => { setF((s) => ({ ...s, [k]: v })); setSaved(false); };
  const merged = { ...character, ...f };
  const S = computeSheet(merged);

  const toggle = (field, key) => set(field, f[field].includes(key) ? f[field].filter((x) => x !== key) : [...f[field], key]);

  const save = async () => {
    setBusy(true);
    try {
      const payload = {
        ...f,
        speed: Number(f.speed) || 0, max_hp: Number(f.max_hp) || 0, current_hp: Number(f.current_hp) || 0,
        temp_hp: Number(f.temp_hp) || 0, hit_dice_used: Number(f.hit_dice_used) || 0,
        death_succ: Number(f.death_succ) || 0, death_fail: Number(f.death_fail) || 0,
        armor_class: f.armor_class === "" || f.armor_class === null ? null : Number(f.armor_class),
        gold: Number(f.gold) || 0, silver: Number(f.silver) || 0, copper: Number(f.copper) || 0,
        electrum: Number(f.electrum) || 0, platinum: Number(f.platinum) || 0,
        spell_ability: f.spell_ability || null,
      };
      const updated = await updateSheet(character.id, payload);
      onSaved?.(updated); setSaved(true);
    } catch { /* ignore */ } finally { setBusy(false); }
  };

  const cd = classData(character.character_class);

  return (
    <div>
      <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim, marginBottom: 4 }}>
        {character.race} {character.character_class}{f.subclass ? ` (${f.subclass})` : ""} — niveau {character.class_level}
        {character.campaign_name ? ` · ${character.campaign_name}` : ""}
      </div>

      {/* Statistiques de combat */}
      <div style={statRow}>
        <Stat icon={Shield} label="Classe d'armure" value={S.ac} hint={merged.armor_class == null ? "10 + DEX" : "saisie"} />
        <Stat icon={Zap} label="Initiative" value={fmtMod(S.initiative)} hint="mod. DEX" />
        <Stat icon={Award} label="Maitrise" value={fmtMod(S.profBonus)} hint={`niv ${character.class_level}`} />
        <Stat icon={Eye} label="Perception passive" value={S.passivePerception} hint="10 + SAG" />
        <Stat icon={Footprints} label="Vitesse" value={`${f.speed || 0} m`} editable
          onChange={(v) => set("speed", v)} raw={f.speed} />
      </div>

      {/* Points de vie + dés de vie + jets de mort */}
      <div style={{ ...statRow, marginTop: 10 }}>
        <NumBox icon={Heart} label="PV max" v={f.max_hp} onChange={(v) => set("max_hp", v)} hint={`de v. d${S.hitDie}`} />
        <NumBox label="PV actuels" v={f.current_hp} onChange={(v) => set("current_hp", v)} />
        <NumBox label="PV temp." v={f.temp_hp} onChange={(v) => set("temp_hp", v)} />
        <NumBox label={`Des de vie (/${character.class_level})`} v={f.hit_dice_used} onChange={(v) => set("hit_dice_used", v)} hint="depenses" />
        <div style={box}>
          <Lbl>Jets de mort</Lbl>
          <div style={{ display: "flex", gap: 10, marginTop: 4, alignItems: "center" }}>
            <MiniNum label="succes" v={f.death_succ} onChange={(v) => set("death_succ", v)} color="#7bb37b" />
            <MiniNum label="echecs" v={f.death_fail} onChange={(v) => set("death_fail", v)} color={T.emberBright} />
          </div>
        </div>
        <button onClick={() => set("inspiration", !f.inspiration)} style={{
          ...box, cursor: "pointer", textAlign: "center",
          border: `1px solid ${f.inspiration ? T.gold : T.line}`, color: f.inspiration ? T.goldBright : T.mistDim }}>
          <Star size={16} color={f.inspiration ? T.gold : T.mistDim} fill={f.inspiration ? T.gold : "none"} />
          <Lbl>Inspiration</Lbl>
        </button>
      </div>

      <Divider label="Caracteristiques & sauvegardes" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        {ABILITY_KEYS.map((k) => {
          const st = S.savingThrows.find((x) => x.key === k);
          return (
            <div key={k} style={{ ...box, display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ textAlign: "center", minWidth: 56 }}>
                <Lbl>{ABILITY_FR[k]}</Lbl>
                <div style={{ fontFamily: ORNATE, fontSize: 22, color: T.parch }}>{character[k]}</div>
                <div style={{ fontFamily: BODY, fontSize: 13, color: T.mistDim }}>mod {fmtMod(S.mods[k])}</div>
              </div>
              <button onClick={() => toggle("save_proficiencies", k)} style={{
                flex: 1, padding: "6px 8px", borderRadius: 3, cursor: "pointer", textAlign: "left",
                background: st.proficient ? "rgba(201,162,75,.12)" : "transparent",
                border: `1px solid ${st.proficient ? T.gold : T.line}`, color: st.proficient ? T.goldBright : T.mist,
                fontFamily: BODY, fontSize: 13.5 }}>
                {st.proficient ? "✦ " : "○ "}Sauvegarde <b style={{ float: "right" }}>{fmtMod(st.value)}</b>
              </button>
            </div>
          );
        })}
      </div>

      <Divider label="Competences" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 6 }}>
        {S.skills.map((sk) => (
          <button key={sk.key} onClick={() => toggle("skill_proficiencies", sk.key)} style={{
            display: "flex", alignItems: "center", justifyContent: "space-between", padding: "7px 11px", borderRadius: 3,
            cursor: "pointer", background: sk.proficient ? "rgba(201,162,75,.12)" : "transparent",
            border: `1px solid ${sk.proficient ? T.gold : T.line}`, color: sk.proficient ? T.goldBright : T.mist,
            fontFamily: BODY, fontSize: 14 }}>
            <span>{sk.proficient ? "✦ " : "○ "}{sk.label} <span style={{ color: T.mistDim, fontSize: 11 }}>({ABILITY_FR[sk.ability]})</span></span>
            <b>{fmtMod(sk.value)}</b>
          </button>
        ))}
      </div>

      {/* Incantation */}
      {(S.caster || cd.spell) && (
        <>
          <Divider label="Incantation" />
          <div style={statRow}>
            <div style={box}>
              <Lbl>Carac. d'incantation</Lbl>
              <select value={f.spell_ability || ""} onChange={(e) => set("spell_ability", e.target.value)}
                style={{ ...inp, marginTop: 4, cursor: "pointer" }}>
                <option value="">{cd.spell ? `Auto (${ABILITY_FR[cd.spell]})` : "Aucune"}</option>
                {ABILITY_KEYS.map((k) => <option key={k} value={k}>{ABILITY_FR[k]}</option>)}
              </select>
            </div>
            <Stat icon={Sparkles} label="DD des sorts" value={S.spellSaveDC ?? "—"} hint="8 + maitrise + mod" />
            <Stat icon={Sparkles} label="Attaque de sort" value={S.spellAttack != null ? fmtMod(S.spellAttack) : "—"} hint="maitrise + mod" />
          </div>
          {character.spells && character.spells.length > 0 && (
            <div style={{ fontFamily: BODY, fontSize: 14, color: T.mist, marginTop: 8 }}>
              <Lbl>Sorts connus</Lbl>
              <div style={{ marginTop: 3 }}>{character.spells.map((s) => s.replace(/-/g, " ")).join(", ")}</div>
            </div>
          )}
        </>
      )}

      <Divider label="Bourse" />
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <MiniNum label="po" v={f.gold} onChange={(v) => set("gold", v)} color="#d9b65a" />
        <MiniNum label="pa" v={f.silver} onChange={(v) => set("silver", v)} color="#c2c8cf" />
        <MiniNum label="pe" v={f.electrum} onChange={(v) => set("electrum", v)} color="#9fb86f" />
        <MiniNum label="pc" v={f.copper} onChange={(v) => set("copper", v)} color="#c08457" />
        <MiniNum label="pp" v={f.platinum} onChange={(v) => set("platinum", v)} color="#cfe0ef" />
      </div>

      <Divider label="Equipement" />
      {inv.length === 0 ? (
        <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 14, color: T.mistDim }}>Aucun objet attribue. Forge et attribue depuis La Forge.</div>
      ) : (
        <div>
          {inv.map((a) => (
            <div key={a.id} style={{ display: "flex", justifyContent: "space-between", fontFamily: BODY, fontSize: 14, color: T.mist, padding: "2px 0" }}>
              <span>{a.item.name}{a.quantity > 1 ? ` ×${a.quantity}` : ""}</span>
              <span style={{ color: T.goldDim, fontSize: 12.5 }}>{fmtCoins(a.item) || a.item.value || ""}</span>
            </div>
          ))}
        </div>
      )}

      <Divider label="Maitrises & langues" />
      <Area label="Armures, armes, outils" v={f.other_proficiencies} onChange={(v) => set("other_proficiencies", v)} />
      <Area label="Langues" v={f.languages} onChange={(v) => set("languages", v)} />

      <Divider label="Traits, dons & capacites" />
      <Area label="Traits d'espece" v={f.species_traits} onChange={(v) => set("species_traits", v)} />
      <Area label="Dons" v={f.feats} onChange={(v) => set("feats", v)} />
      <Area label="Capacites de classe" v={f.class_features} onChange={(v) => set("class_features", v)} />

      <Divider label="Identite" />
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <TxtField label="Sous-classe" v={f.subclass} onChange={(v) => set("subclass", v)} />
        <TxtField label="Taille" v={f.size} onChange={(v) => set("size", v)} w={90} />
        <TxtField label="Alignement" v={f.alignment} onChange={(v) => set("alignment", v)} />
        <TxtField label="Historique" v={f.background} onChange={(v) => set("background", v)} />
      </div>
      <div style={{ marginTop: 8 }}><Area label="Histoire & personnalite" v={f.personality} onChange={(v) => set("personality", v)} rows={4} /></div>

      <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 18 }}>
        <button onClick={save} disabled={busy} className="glow" style={{
          display: "inline-flex", alignItems: "center", gap: 8, padding: "11px 22px", borderRadius: 3,
          background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: busy ? "wait" : "pointer",
          fontFamily: DISPLAY, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase", opacity: busy ? 0.6 : 1 }}>
          <Save size={14} /> Enregistrer la fiche
        </button>
        {saved && <span style={{ fontFamily: BODY, fontSize: 14, color: T.gold }}>Enregistre ✓</span>}
      </div>
    </div>
  );
}

/* --------------------------------- UI bits ------------------------------- */
const statRow = { display: "flex", gap: 10, flexWrap: "wrap" };
const box = { flex: 1, minWidth: 120, padding: "10px 12px", borderRadius: 5, background: T.ink, border: `1px solid ${T.line}` };
const inp = { width: "100%", padding: "7px 9px", borderRadius: 3, background: T.panel, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 14, outline: "none", boxSizing: "border-box" };
const Lbl = ({ children }) => <div style={{ fontFamily: DISPLAY, fontSize: 8.5, letterSpacing: 1.2, textTransform: "uppercase", color: T.goldDim }}>{children}</div>;

function Stat({ icon: Icon, label, value, hint, editable, onChange, raw }) {
  return (
    <div style={{ ...box, textAlign: "center" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
        {Icon && <Icon size={13} color={T.gold} />}<Lbl>{label}</Lbl>
      </div>
      {editable ? (
        <input type="number" min="0" value={raw} onChange={(e) => onChange(e.target.value)} style={{ ...inp, marginTop: 4, textAlign: "center", fontFamily: ORNATE, fontSize: 20 }} />
      ) : (
        <div style={{ fontFamily: ORNATE, fontSize: 26, color: T.parch, marginTop: 2 }}>{value}</div>
      )}
      {hint && <div style={{ fontFamily: BODY, fontSize: 11, color: T.mistDim, fontStyle: "italic" }}>{hint}</div>}
    </div>
  );
}
function NumBox({ icon: Icon, label, v, onChange, hint }) {
  return (
    <div style={{ ...box, textAlign: "center" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
        {Icon && <Icon size={13} color={T.gold} />}<Lbl>{label}</Lbl>
      </div>
      <input type="number" min="0" value={v} onChange={(e) => onChange(e.target.value)} style={{ ...inp, marginTop: 4, textAlign: "center", fontFamily: ORNATE, fontSize: 20 }} />
      {hint && <div style={{ fontFamily: BODY, fontSize: 11, color: T.mistDim, fontStyle: "italic" }}>{hint}</div>}
    </div>
  );
}
function MiniNum({ label, v, onChange, color }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
      <input type="number" min="0" value={v} onChange={(e) => onChange(e.target.value)} style={{ ...inp, width: 70, textAlign: "center" }} />
      <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color }}>{label}</span>
    </label>
  );
}
function TxtField({ label, v, onChange, w }) {
  return (
    <div style={{ flex: w ? "0 0 auto" : 1, minWidth: w || 150, width: w }}>
      <Lbl>{label}</Lbl>
      <input value={v || ""} onChange={(e) => onChange(e.target.value)} style={{ ...inp, marginTop: 4 }} />
    </div>
  );
}
function Area({ label, v, onChange, rows = 2 }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <Lbl>{label}</Lbl>
      <textarea value={v || ""} onChange={(e) => onChange(e.target.value)} rows={rows} style={{ ...inp, marginTop: 4, resize: "vertical" }} />
    </div>
  );
}
