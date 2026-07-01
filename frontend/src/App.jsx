/**
 * App — racine de l'interface "Grimoire".
 *
 * Gate l'application derriere l'authentification : tant qu'aucun utilisateur
 * n'est connecte, on affiche l'ecran de connexion. Sinon, le rail + les vues.
 */
import { useEffect, useState } from "react";
import { Menu } from "lucide-react";
import { T, BODY, DISPLAY } from "./theme.js";
import { getHealth } from "./api";
import { useAuth } from "./auth/AuthContext.jsx";
import { useViewport } from "./hooks/useViewport.js";
import { GlobalStyles, Atmosphere } from "./components/atmosphere.jsx";
import { Rail } from "./components/Rail.jsx";
import { DiceRoller } from "./components/DiceRoller.jsx";
import { Threshold } from "./pages/Threshold.jsx";
import { Grimoire } from "./pages/Grimoire.jsx";
import { Table } from "./pages/Table.jsx";
import { Wizard } from "./pages/Wizard.jsx";
import { CharacterCreate } from "./pages/CharacterCreate.jsx";
import { Library } from "./pages/Library.jsx";
import { Forge } from "./pages/Forge.jsx";
import { Bibliotheque } from "./pages/Bibliotheque.jsx";
import { Antre } from "./pages/Antre.jsx";
import { Arsenal } from "./pages/Arsenal.jsx";
import { Solo } from "./pages/Solo.jsx";
import { AuthScreen } from "./pages/AuthScreen.jsx";
import { ErrorBoundary } from "./components/ErrorBoundary.jsx";

export default function App() {
  const { user, loading } = useAuth();
  const [view, setView] = useState("home");
  const [online, setOnline] = useState(false);
  const { isMobile } = useViewport();
  const [railOpen, setRailOpen] = useState(true);
  // Sur mobile la barre laterale est repliee par defaut (drawer) ; ouverte sur desktop.
  useEffect(() => { setRailOpen(!isMobile); }, [isMobile]);
  // Sur mobile, changer de vue referme le drawer.
  useEffect(() => { if (isMobile) setRailOpen(false); }, [view, isMobile]);

  useEffect(() => {
    if (!user) return;
    let alive = true;
    const ping = () => getHealth().then(() => alive && setOnline(true)).catch(() => alive && setOnline(false));
    ping();
    const id = setInterval(ping, 30000);
    return () => { alive = false; clearInterval(id); };
  }, [user]);

  return (
    <div style={{ position: "relative", minHeight: "100vh", background: T.void, color: T.mist, fontFamily: BODY, overflow: "hidden" }}>
      <GlobalStyles />
      <Atmosphere />

      {loading ? (
        <div style={{ position: "relative", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
          fontFamily: DISPLAY, letterSpacing: 4, textTransform: "uppercase", color: T.mistDim }}>
          Ouverture du grimoire...
        </div>
      ) : !user ? (
        <div style={{ position: "relative" }}><AuthScreen /></div>
      ) : (
        <>
          <div style={{ position: "relative", display: "flex", height: "100vh" }}>
            <Rail view={view} setView={setView} online={online} isMobile={isMobile} open={railOpen} setOpen={setRailOpen} />
            {!railOpen && (
              <button onClick={() => setRailOpen(true)} title="Menu" style={{
                position: "fixed", top: 14, left: 14, zIndex: 45, width: 44, height: 44, borderRadius: 10,
                display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer",
                background: T.panel2, border: `1px solid ${T.gold}`, color: T.gold }}>
                <Menu size={20} />
              </button>
            )}
            <main style={{ flex: 1, minWidth: 0, minHeight: 0, display: "flex", flexDirection: "column" }}>
              <ErrorBoundary key={view} onHome={() => setView("home")}>
              {view === "home" && <Threshold setView={setView} />}
              {view === "rules" && <Grimoire />}
              {view === "gm" && <Table setView={setView} />}
              {view === "create" && <Wizard setView={setView} />}
              {view === "character" && <CharacterCreate setView={setView} />}
              {(view === "antre" || view === "library" || view === "library2") && <Antre setView={setView} />}
              {view === "forge" && <Forge />}
              {view === "arsenal" && <Arsenal setView={setView} />}
              {view === "solo" && <Solo setView={setView} />}
              </ErrorBoundary>
            </main>
          </div>
          <DiceRoller />
        </>
      )}
    </div>
  );
}
