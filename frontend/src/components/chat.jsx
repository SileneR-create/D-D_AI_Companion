/**
 * Pieces de conversation du theme Grimoire.
 *
 * `Conversation` est *presentational* : elle recoit l'etat produit par
 * `useChatStream` (messages API + brouillon en streaming) et le traduit en
 * bulles "Oracle / Chercheur", avec un indicateur anime pendant la generation.
 */
import { useEffect, useRef } from "react";
import { Feather, Sparkles, BookOpen, Gem, Send } from "lucide-react";
import { T, DISPLAY, BODY } from "../theme.js";
import { Tag } from "./ornaments.jsx";
import { toolLabel } from "../lib/toolLabels.js";

/** Trois points animes (recherche en cours). */
function TypingDots() {
  return (
    <span style={{ display: "inline-flex", gap: 4, marginLeft: 4 }}>
      {[0, 1, 2].map((i) => (
        <span key={i} className="dot" style={{
          width: 6, height: 6, borderRadius: "50%", background: T.gold,
          animationDelay: `${i * 0.18}s`,
        }} />
      ))}
    </span>
  );
}

/** Bandeau "l'Oracle travaille" : consultation d'un outil ou reflexion. */
function ThinkingLine({ toolName }) {
  const text = toolName ? toolLabel(toolName) : "L'Oracle consulte les pages anciennes";
  return (
    <span style={{ display: "inline-flex", alignItems: "center", color: T.mistDim, animation: "oraclePulse 1.6s infinite" }}>
      {toolName && <Gem size={13} color={T.gold} strokeWidth={1.4} style={{ marginRight: 6 }} />}
      {text}
      <TypingDots />
    </span>
  );
}

/** Une bulle d'echange. `m` = { who, text, pending, toolName, source, sigil }. */
export function Exchange({ m }) {
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
        whiteSpace: "pre-wrap",
      }}>
        {m.pending ? <ThinkingLine toolName={m.toolName} /> : m.text}
      </p>
      {oracle && !m.pending && (m.source || m.sigil) && (
        <div style={{ display: "flex", gap: 16, marginTop: 10, paddingLeft: 26 }}>
          {m.source && <Tag icon={BookOpen} text={m.source} />}
          {m.sigil && <Tag icon={Gem} text={toolLabel(m.sigil)} accent />}
        </div>
      )}
    </div>
  );
}

/**
 * Liste defilante d'echanges.
 */
export function Conversation({ messages, draft, streaming, toolName, source }) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, draft, streaming, toolName]);

  return (
    <div style={{ flex: 1, overflowY: "auto", paddingRight: 6, minWidth: 0 }}>
      {messages.map((m, i) => (
        <Exchange key={i} m={{
          who: m.role === "user" ? "seeker" : "oracle",
          text: m.content,
          source: m.role === "user" ? null : source,
          sigil: m.tool || null,
        }} />
      ))}
      {streaming && (
        <Exchange m={{
          who: "oracle",
          text: draft,
          pending: !draft,
          toolName,
          source,
          sigil: toolName || null,
        }} />
      )}
      <div ref={endRef} />
    </div>
  );
}

/** Champ de saisie ("encrier"). */
export function Inkwell({ value, setValue, onSend, placeholder, disabled }) {
  return (
    <div style={{ padding: "12px 0 0" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "9px 18px", borderRadius: 4,
        background: T.ink, border: `1px solid ${T.line}`, boxShadow: "inset 0 1px 0 rgba(201,162,75,.08)" }}>
        <Feather size={18} color={T.gold} strokeWidth={1.3} />
        <input value={value} onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !disabled && onSend()}
          placeholder={placeholder} disabled={disabled}
          style={{ flex: 1, background: "transparent", border: "none", outline: "none",
            fontFamily: BODY, fontSize: 18, color: T.parch }} />
        <button onClick={onSend} className="glow" disabled={disabled} style={{
          display: "inline-flex", alignItems: "center", gap: 8,
          cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1,
          padding: "8px 18px", borderRadius: 3, background: "transparent",
          border: `1px solid ${T.gold}`, color: T.gold,
          fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2, textTransform: "uppercase" }}>
          Invoquer <Send size={13} />
        </button>
      </div>
    </div>
  );
}
