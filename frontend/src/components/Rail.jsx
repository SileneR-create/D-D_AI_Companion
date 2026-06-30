/** Barre de navigation laterale (Seuil / Grimoire / La Table) + compte. */
import { Compass, BookOpen, Swords, Castle, Hammer, Archive, Sword, Flame, LogOut } from "lucide-react";
import { T, DISPLAY } from "../theme.js";
import { Crest } from "./atmosphere.jsx";
import { useAuth } from "../auth/AuthContext.jsx";

const ITEMS = [
  { id: "home", label: "Seuil", icon: Compass },
  { id: "rules", label: "Grimoire", icon: BookOpen },
  { id: "gm", label: "La Table", icon: Swords },
  { id: "library", label: "Mon antre", icon: Castle },
  { id: "forge", label: "La Forge", icon: Hammer },
  { id: "library2", label: "Archives", icon: Archive },
  { id: "arsenal", label: "Arsenal", icon: Sword },
];

export function Rail({ view, setView, online }) {
  const { user, logout } = useAuth();
  return (
    <aside style={{
      width: 96, display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "space-between", padding: "26px 0",
      background: `linear-gradient(180deg, ${T.ink}, ${T.void})`,
      borderRight: `1px solid ${T.line}`,
    }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 30 }}>
        <button onClick={() => setView("home")} title="Seuil"
          style={{ background: "none", border: "none", cursor: "pointer", padding: 0 }}>
          <Crest size={52} />
        </button>
        <nav style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {ITEMS.map(({ id, label, icon: Icon }) => {
            const on = view === id;
            return (
              <button key={id} onClick={() => setView(id)} title={label}
                style={{
                  width: 60, height: 60, borderRadius: 14, cursor: "pointer",
                  display: "flex", flexDirection: "column", alignItems: "center",
                  justifyContent: "center", gap: 3,
                  background: on ? "rgba(201,162,75,.12)" : "transparent",
                  border: `1px solid ${on ? T.gold : "transparent"}`,
                  color: on ? T.goldBright : T.mistDim,
                }}>
                <Icon size={19} strokeWidth={1.4} />
                <span style={{ fontFamily: DISPLAY, fontSize: 8, letterSpacing: 1.5, textTransform: "uppercase" }}>{label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
        {/* Braise = etat du backend */}
        <Flame size={18} strokeWidth={1.3} color={online ? T.emberBright : T.line}
          title={online ? "Oracle connecte" : "Oracle injoignable"} />
        {/* Pastille utilisateur + deconnexion */}
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
}
