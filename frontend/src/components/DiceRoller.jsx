/**
 * DiceRoller — bouton flottant + overlay de lancer de de 20.
 *
 * Clic sur le bouton -> overlay assombri, le d20 "roule" (defilement rapide de
 * valeurs) puis se fige sur un resultat aleatoire 1-20. Critique (20) et echec
 * critique (1) sont mis en valeur. Clic sur le fond ou Echap pour fermer.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Dices } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";

export function DiceRoller() {
  const [open, setOpen] = useState(false);
  const [rolling, setRolling] = useState(false);
  const [value, setValue] = useState(20);
  const timer = useRef(null);

  const roll = useCallback(() => {
    setOpen(true);
    setRolling(true);
    clearInterval(timer.current);
    const start = Date.now();
    timer.current = setInterval(() => {
      setValue(1 + Math.floor(Math.random() * 20));
      if (Date.now() - start > 850) {
        clearInterval(timer.current);
        setValue(1 + Math.floor(Math.random() * 20));
        setRolling(false);
      }
    }, 60);
  }, []);

  useEffect(() => () => clearInterval(timer.current), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const crit = !rolling && value === 20;
  const fumble = !rolling && value === 1;
  const accent = crit ? T.goldBright : fumble ? T.emberBright : T.gold;

  return (
    <>
      {/* Bouton flottant */}
      <button onClick={roll} title="Lancer un de 20" style={{
        position: "fixed", right: 26, bottom: 26, zIndex: 40,
        width: 58, height: 58, borderRadius: "50%", cursor: "pointer",
        display: "flex", alignItems: "center", justifyContent: "center",
        background: `radial-gradient(circle at 50% 35%, ${T.panel2}, ${T.ink})`,
        border: `1px solid ${T.gold}`, color: T.gold,
        boxShadow: "0 8px 24px rgba(0,0,0,.55)",
      }}>
        <Dices size={26} strokeWidth={1.4} />
      </button>

      {/* Overlay */}
      {open && (
        <div onClick={() => !rolling && setOpen(false)} style={{
          position: "fixed", inset: 0, zIndex: 50,
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          background: "rgba(8,6,12,.78)", backdropFilter: "blur(3px)", cursor: rolling ? "default" : "pointer",
        }}>
          <D20 value={value} rolling={rolling} accent={accent} />

          <div style={{ marginTop: 26, textAlign: "center" }}>
            {rolling ? (
              <div style={{ fontFamily: DISPLAY, fontSize: 13, letterSpacing: 4, textTransform: "uppercase", color: T.mistDim }}>
                Le sort en est jete...
              </div>
            ) : (
              <>
                <div style={{ fontFamily: ORNATE, fontSize: 20, color: accent }}>
                  {crit ? "Coup critique !" : fumble ? "Echec critique !" : `Vous obtenez ${value}`}
                </div>
                <div style={{ fontFamily: BODY, fontStyle: "italic", fontSize: 15, color: T.mistDim, marginTop: 6 }}>
                  Cliquez n'importe ou pour fermer — ou relancez le de.
                </div>
                <button onClick={(e) => { e.stopPropagation(); roll(); }} style={{
                  marginTop: 16, padding: "8px 20px", borderRadius: 3, cursor: "pointer",
                  background: "transparent", border: `1px solid ${T.gold}`, color: T.gold,
                  fontFamily: DISPLAY, fontSize: 11, letterSpacing: 2, textTransform: "uppercase",
                }}>
                  Relancer
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

/** Silhouette d'icosaedre (d20) avec la valeur au centre. */
function D20({ value, rolling, accent }) {
  return (
    <div style={{ position: "relative", width: 200, height: 200, animation: rolling ? "d20Spin .6s linear infinite" : "none" }}>
      <svg width="200" height="200" viewBox="0 0 200 200">
        <defs>
          <radialGradient id="d20g" cx="50%" cy="38%" r="65%">
            <stop offset="0%" stopColor="#2A2030" />
            <stop offset="100%" stopColor="#100B16" />
          </radialGradient>
        </defs>
        {/* contour hexagonal */}
        <polygon points="100,8 180,54 180,146 100,192 20,146 20,54"
          fill="url(#d20g)" stroke={accent} strokeWidth="2.5" />
        {/* facettes internes */}
        <polygon points="100,8 180,54 100,72 20,54" fill="none" stroke={accent} strokeWidth="1" opacity="0.5" />
        <polygon points="20,54 100,72 60,150 20,146" fill="none" stroke={accent} strokeWidth="1" opacity="0.4" />
        <polygon points="180,54 100,72 140,150 180,146" fill="none" stroke={accent} strokeWidth="1" opacity="0.4" />
        <polygon points="100,72 140,150 100,192 60,150" fill="none" stroke={accent} strokeWidth="1" opacity="0.5" />
        <polygon points="60,150 140,150 100,192" fill="none" stroke={accent} strokeWidth="1" opacity="0.6" />
      </svg>
      <div style={{
        position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: ORNATE, fontSize: 56, color: accent, textShadow: "0 2px 12px rgba(0,0,0,.6)",
      }}>
        {value}
      </div>
    </div>
  );
}
