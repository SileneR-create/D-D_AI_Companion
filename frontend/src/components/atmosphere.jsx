/**
 * Chrome visuel du theme Grimoire : styles globaux, ambiance de fond, et blason.
 */
import { T } from "../theme.js";

/** Polices Google + petites regles globales (hover, selection, placeholder). */
export function GlobalStyles() {
  return (
    <style>{`@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Cinzel+Decorative:wght@700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');
      *::selection{ background:${T.gold}; color:${T.void}; }
      .lift{ transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
      .lift:hover{ transform: translateY(-4px); border-color:${T.gold}!important; box-shadow:0 16px 40px rgba(0,0,0,.55); }
      .glow:hover{ color:${T.goldBright}!important; }
      input::placeholder{ color:${T.mistDim}; font-style:italic; }
      @keyframes oracleDots { 0%,80%,100%{ opacity:.2 } 40%{ opacity:1 } }
      @keyframes oraclePulse { 0%,100%{ opacity:.45 } 50%{ opacity:1 } }
      @keyframes d20Spin { from{ transform:rotate(0deg) } to{ transform:rotate(360deg) } }
      .dot{ animation: oracleDots 1.2s infinite both; }
    `}</style>
  );
}

/** Fond anime : degrades, grain, constellations. */
export function Atmosphere() {
  const noise =
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='200' height='200' filter='url(%23n)' opacity='0.5'/></svg>\")";
  return (
    <>
      <div style={{ position: "absolute", inset: 0, background:
        `radial-gradient(120% 80% at 50% -10%, #241a30 0%, ${T.void} 60%)` }} />
      <div style={{ position: "absolute", inset: 0, background:
        "radial-gradient(100% 100% at 50% 120%, rgba(166,64,46,.18), transparent 55%)" }} />
      <div style={{ position: "absolute", inset: 0, backgroundImage: noise, opacity: 0.04, mixBlendMode: "overlay" }} />
      <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0.5 }}>
        {[[120,90],[300,60],[480,140],[760,80],[920,180],[1180,70],[1040,260],[200,300]].map(([x,y],i)=>(
          <circle key={i} cx={x} cy={y} r={i%3===0?1.6:1} fill={T.gold} opacity={0.35} />
        ))}
      </svg>
    </>
  );
}

/** Blason : dragon love dans un cercle astrolabe. */
export function Crest({ size = 120 }) {
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
      <g stroke={T.gold} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <path d="M60 32 C40 34 30 52 38 70 C44 84 64 88 76 78" />
        <path d="M76 78 C86 70 86 56 78 50 C72 46 64 48 62 56" />
        <path d="M60 32 C57 28 58 24 63 23 C66 22 70 24 70 28 L66 30" fill={T.gold} />
        <path d="M52 50 L40 40 L46 54 L38 52 L48 62" />
        <path d="M62 56 C58 60 58 66 63 67" />
      </g>
      <circle cx="64.5" cy="27" r="1.1" fill={T.emberBright} />
    </svg>
  );
}
