/**
 * Moteur de QCM guide (question apres question) reutilisable.
 * - steps : [{ key, q, type:'text'|'single'|'multi', options?, optional?, textarea?, placeholder?, skipIf? }]
 * - compose(answers) -> texte recapitulatif
 * - onCreate(answers) -> Promise (action finale)
 */
import { useMemo, useState } from "react";
import { ChevronRight, ChevronLeft, Check } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";

export function Qcm({ steps: allSteps, compose, onCreate, createLabel = "Creer" }) {
  const [answers, setAnswers] = useState({});
  const [step, setStep] = useState(0);
  const [review, setReview] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const steps = useMemo(() => allSteps.filter((s) => !(s.skipIf && s.skipIf(answers))), [allSteps, answers]);
  const cur = steps[step];
  const set = (k, v) => setAnswers((a) => ({ ...a, [k]: v }));
  const toggle = (k, opt) => setAnswers((a) => {
    const c = a[k] || []; return { ...a, [k]: c.includes(opt) ? c.filter((x) => x !== opt) : [...c, opt] };
  });

  const next = () => {
    if (cur.type === "text" && !cur.optional && !(answers[cur.key] || "").trim()) { setErr("Ce champ est requis."); return; }
    setErr(null);
    if (step < steps.length - 1) setStep(step + 1); else setReview(true);
  };
  const back = () => { if (review) setReview(false); else if (step > 0) setStep(step - 1); };
  const create = async () => {
    setBusy(true); setErr(null);
    try { await onCreate(answers); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  if (review) {
    return (
      <>
        <div style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: T.gold, marginBottom: 8 }}>Recapitulatif</div>
        {answers.name && <div style={{ fontFamily: ORNATE, fontSize: 18, color: T.parch, marginBottom: 6 }}>{answers.name}</div>}
        <p style={{ fontFamily: BODY, fontSize: 15.5, lineHeight: 1.7, color: T.mist, background: T.ink, border: `1px solid ${T.line}`, borderRadius: 4, padding: "12px 14px" }}>
          {compose(answers)}
        </p>
        {err && <Err>{err}</Err>}
        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
          <button onClick={back} style={ghost}><ChevronLeft size={13} /> Retour</button>
          <button onClick={create} disabled={busy} className="glow" style={{ ...gold, opacity: busy ? 0.6 : 1 }}><Check size={14} /> {createLabel}</button>
        </div>
      </>
    );
  }

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 2, textTransform: "uppercase", color: T.mistDim }}>Etape {step + 1} / {steps.length}</span>
        <div style={{ display: "flex", gap: 4 }}>{steps.map((_, i) => <span key={i} style={{ width: 18, height: 3, borderRadius: 2, background: i <= step ? T.gold : T.line }} />)}</div>
      </div>
      <div style={{ fontFamily: ORNATE, fontSize: 19, color: T.parch, marginBottom: 14 }}>{cur.q}</div>

      {cur.type === "text" && (cur.textarea
        ? <textarea autoFocus rows={3} style={{ ...inp, resize: "vertical" }} value={answers[cur.key] || ""} onChange={(e) => set(cur.key, e.target.value)} placeholder={cur.placeholder} />
        : <input autoFocus style={inp} value={answers[cur.key] || ""} onChange={(e) => set(cur.key, e.target.value)} placeholder={cur.placeholder} onKeyDown={(e) => e.key === "Enter" && next()} />)}

      {cur.type === "single" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {cur.options.map((o) => {
            const on = answers[cur.key] === o;
            return <button key={o} onClick={() => { set(cur.key, o); setTimeout(next, 110); }} style={{ ...choice, ...(on ? choiceOn : {}) }}>{o}</button>;
          })}
        </div>
      )}

      {cur.type === "multi" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {cur.options.map((o) => {
            const on = (answers[cur.key] || []).includes(o);
            return <button key={o} onClick={() => toggle(cur.key, o)} style={{ ...choice, ...(on ? choiceOn : {}) }}>{on ? "✦ " : ""}{o}</button>;
          })}
        </div>
      )}

      {err && <Err>{err}</Err>}
      <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
        {step > 0 && <button onClick={back} style={ghost}><ChevronLeft size={13} /> Retour</button>}
        <div style={{ flex: 1 }} />
        {cur.type !== "single" && <button onClick={next} className="glow" style={gold}>Suivant <ChevronRight size={14} /></button>}
      </div>
    </>
  );
}

const Err = ({ children }) => <div style={{ fontFamily: BODY, fontSize: 14, color: T.emberBright, marginTop: 8 }}>{children}</div>;
const inp = { width: "100%", padding: "10px 12px", borderRadius: 3, background: T.ink, border: `1px solid ${T.line}`, color: T.parch, fontFamily: BODY, fontSize: 16, outline: "none", boxSizing: "border-box" };
const gold = { display: "inline-flex", alignItems: "center", gap: 8, padding: "10px 18px", borderRadius: 3, background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer", fontFamily: DISPLAY, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase" };
const ghost = { display: "inline-flex", alignItems: "center", gap: 6, padding: "10px 16px", borderRadius: 3, background: "transparent", border: `1px solid ${T.line}`, color: T.mistDim, cursor: "pointer", fontFamily: DISPLAY, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase" };
const choice = { padding: "12px 14px", textAlign: "left", borderRadius: 4, cursor: "pointer", background: T.panel, border: `1px solid ${T.line}`, color: T.mist, fontFamily: BODY, fontSize: 15.5 };
const choiceOn = { background: "rgba(201,162,75,.12)", border: `1px solid ${T.gold}`, color: T.parch };
