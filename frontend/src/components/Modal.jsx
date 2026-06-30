/** Modale reutilisable (overlay assombri + panneau centre). */
import { useEffect } from "react";
import { X } from "lucide-react";
import { T, ORNATE } from "../theme.js";

export function Modal({ title, onClose, children, width = 540 }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, zIndex: 60, display: "flex", alignItems: "flex-start", justifyContent: "center",
      background: "rgba(8,6,12,.74)", backdropFilter: "blur(3px)", overflowY: "auto", padding: "48px 16px",
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width, maxWidth: "100%", borderRadius: 8, position: "relative",
        background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.line}`,
        boxShadow: "0 24px 60px rgba(0,0,0,.6)", padding: "22px 24px",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <h2 style={{ fontFamily: ORNATE, fontSize: 20, color: T.parch, margin: 0 }}>{title}</h2>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: T.mistDim, display: "flex" }}>
            <X size={20} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
