/** Petits ornements reutilisables du theme Grimoire. */
import { Gem } from "lucide-react";
import { T, DISPLAY, ORNATE } from "../theme.js";

export function Filigree({ flip }) {
  return (
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none"
      style={{ transform: flip ? "scaleX(-1)" : "none", opacity: .8 }}>
      <path d="M2 2 H18 M2 2 V18 M2 2 C18 6 22 10 26 26 C30 30 36 30 40 26"
        stroke={T.gold} strokeWidth="1" fill="none" strokeLinecap="round" />
      <circle cx="40" cy="26" r="2" fill={T.gold} />
    </svg>
  );
}

export function Divider({ label }) {
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

export function ScreenTitle({ eyebrow, title }) {
  return (
    <div style={{ textAlign: "center", padding: "26px 0 6px" }}>
      <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 5, textTransform: "uppercase", color: T.gold }}>{eyebrow}</div>
      <h1 style={{ fontFamily: ORNATE, fontSize: 30, color: T.parch, margin: "6px 0 0", fontWeight: 700, letterSpacing: 1 }}>{title}</h1>
    </div>
  );
}

export function Tag({ icon: Icon, text, accent }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 1.5, textTransform: "uppercase", color: accent ? T.gold : T.mistDim }}>
      <Icon size={12} strokeWidth={1.4} /> {text}
    </span>
  );
}

export function SectionLabel({ icon: Icon, text }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "20px 0 10px" }}>
      <Icon size={14} color={T.gold} strokeWidth={1.4} />
      <span style={{ fontFamily: DISPLAY, fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase", color: T.gold }}>{text}</span>
      <span style={{ flex: 1, height: 1, background: T.line }} />
    </div>
  );
}
