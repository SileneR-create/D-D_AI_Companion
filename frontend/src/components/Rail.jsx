/** Barre laterale allegee : 3 modules (Grimoire / Maitre / Solo). Un clic deplie
 *  les outils du module dans un volet lateral. + compte / etat backend. */
import { useState } from "react";
import { BookOpen, Swords, Castle, Hammer, Sword, Drama, Flame, LogOut, ChevronRight, ChevronLeft } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Crest } from "./atmosphere.jsx";
import { useAuth } from "../auth/AuthContext.jsx";

const MODULES = [
  { key: "grimoire", label: "Grimoire", icon: BookOpen, tools: [
    { id: "rules", label: "Le Grimoire", sub: "Regles & oracle", icon: BookOpen },
  ] },
  { key: "md", label: "Maitre", icon: Swords, tools: [
    { id: "gm", label: "La Table", sub: "Tableau de bord", icon: Swords },
    { id: "antre", label: "L'Antre", sub: "Persos, campagnes, PNJ, lieux", icon: Castle },
    { id: "arsenal", label: "L'Arsenal", sub: "Objets & equipement", icon: Sword },
    { id: "forge", label: "La Forge", sub: "Armes & objets magiques", icon: Hammer },
  ] },
  { key: "solo", label: "Solo", icon: Drama, tools: [
    { id: "solo", label: "Aventure en solo", sub: "Narrateur IA", icon: Drama },
  ] },
];

export function Rail({ view, setView, online, isMobile = false, open = true, setOpen }) {
  const { user, logout } = useAuth();
  const [openKey, setOpenKey] = useState(null);
  if (!open) return null;
  const go = (id) => { setView(id); setOpenKey(null); if (isMobile && setOpen) setOpen(false); };

  const aside = (
    <aside style={{
      width: 96, flexShrink: 0, display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "space-between", padding: "20px 0",
      position: isMobile ? "fixed" : "relative", left: 0, top: 0,
      height: isMobile ? "100vh" : "auto", zIndex: isMobile ? 60 : "auto",
      background: `linear-gradient(180deg, ${T.ink}, ${T.void})`, borderRight: `1px solid ${T.line}`,
    }}>
      {/* Voile de fermeture au clic exterieur */}
      {openKey && <div onClick={() => setOpenKey(null)} style={{ position: "fixed", inset: 0, zIndex: 20 }} />}

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 26, position: "relative", zIndex: 30 }}>
        <button onClick={() => go("home")} title="Seuil"
          style={{ background: "none", border: "none", cursor: "pointer", padding: 0 }}>
          <Crest size={50} />
        </button>

        {setOpen && (
          <button onClick={() => setOpen(false)} title="Replier la barre" style={{
            display: "inline-flex", alignItems: "center", gap: 3, background: "transparent",
            border: `1px solid ${T.line}`, borderRadius: 8, padding: "4px 7px", cursor: "pointer",
            color: T.mistDim, fontFamily: DISPLAY, fontSize: 7.5, letterSpacing: 1, textTransform: "uppercase" }}>
            <ChevronLeft size={12} /> Replier
          </button>
        )}

        <nav style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {MODULES.map((m) => {
            const Icon = m.icon;
            const active = m.tools.some((t) => t.id === view);
            const open = openKey === m.key;
            return (
              <div key={m.key} style={{ position: "relative" }}>
                <button onClick={() => setOpenKey(open ? null : m.key)} title={m.label} style={{
                  width: 62, height: 62, borderRadius: 14, cursor: "pointer",
                  display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 3,
                  background: (active || open) ? "rgba(201,162,75,.12)" : "transparent",
                  border: `1px solid ${(active || open) ? T.gold : "transparent"}`,
                  color: (active || open) ? T.goldBright : T.mistDim,
                }}>
                  <Icon size={20} strokeWidth={1.4} />
                  <span style={{ fontFamily: DISPLAY, fontSize: 8, letterSpacing: 1.2, textTransform: "uppercase" }}>{m.label}</span>
                </button>

                {open && (
                  <div style={{
                    position: "absolute", left: "100%", top: 0, marginLeft: 12, zIndex: 31, width: 234,
                    background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`, border: `1px solid ${T.gold}`,
                    borderRadius: 8, padding: 8, boxShadow: "0 20px 48px rgba(0,0,0,.55)",
                  }}>
                    <div style={{ fontFamily: ORNATE, fontSize: 16, color: T.parch, padding: "4px 8px 8px" }}>{m.label}</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      {m.tools.map((t) => {
                        const TIcon = t.icon;
                        const on = t.id === view;
                        return (
                          <button key={t.id} onClick={() => go(t.id)} className="glow" style={{
                            display: "flex", alignItems: "center", gap: 10, padding: "9px 10px", borderRadius: 5, cursor: "pointer",
                            background: on ? "rgba(201,162,75,.14)" : "rgba(201,162,75,.04)",
                            border: `1px solid ${on ? T.gold : T.line}`, color: on ? T.parch : T.mist, textAlign: "left",
                          }}>
                            <TIcon size={16} color={T.gold} strokeWidth={1.4} />
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div style={{ fontFamily: BODY, fontSize: 14.5 }}>{t.label}</div>
                              <div style={{ fontFamily: BODY, fontSize: 11.5, color: T.mistDim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.sub}</div>
                            </div>
                            <ChevronRight size={14} color={T.gold} />
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16, position: "relative", zIndex: 30 }}>
        <Flame size={18} strokeWidth={1.3} color={online ? T.emberBright : T.line}
          title={online ? "Oracle connecte" : "Oracle injoignable"} />
        {user && (
          <button onClick={logout} title={`Se deconnecter (${user.username})`} style={{
            width: 44, height: 44, borderRadius: "50%", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center", position: "relative",
            background: "rgba(201,162,75,.10)", border: `1px solid ${T.goldDim}`, color: T.gold,
          }}>
            <LogOut size={15} strokeWidth={1.5} />
          </button>
        )}
        {user && (
          <span style={{ fontFamily: DISPLAY, fontSize: 8, letterSpacing: 1, textTransform: "uppercase", color: T.mistDim, maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {user.username}
          </span>
        )}
      </div>
    </aside>
  );

  if (isMobile) {
    return (
      <>
        <div onClick={() => setOpen && setOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(8,6,12,.6)", zIndex: 55 }} />
        {aside}
      </>
    );
  }
  return aside;
}
