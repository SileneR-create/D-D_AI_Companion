/**
 * Campagne active : { id, name, role } partagee entre le wizard et La Table.
 * Persistee en localStorage pour survivre aux rechargements.
 */
import { createContext, useContext, useEffect, useState } from "react";
import { listCampaigns } from "../api/campaigns.js";

const KEY = "dnd_active_campaign";
const ActiveCampaignContext = createContext(null);

function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || null; } catch { return null; }
}

export function ActiveCampaignProvider({ children }) {
  const [active, setActiveState] = useState(load);

  const setActive = (c) => {
    setActiveState(c);
    if (c) localStorage.setItem(KEY, JSON.stringify(c));
    else localStorage.removeItem(KEY);
  };

  // Nettoie si l'utilisateur se deconnecte (le jeton disparait).
  useEffect(() => {
    const onStorage = () => { if (!localStorage.getItem("dnd_token")) setActiveState(null); };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Reconcilie la campagne active persistee UNE FOIS au montage : si elle
  // n'existe plus en BDD (supprimee, autre compte, base reinitialisee), on la
  // retire. On ne le fait PAS a chaque changement pour ne pas effacer une
  // campagne fraichement creee, et jamais sur une liste vide (erreur reseau).
  useEffect(() => {
    const a = load();
    if (!a?.id || !localStorage.getItem("dnd_token")) return;
    listCampaigns()
      .then((list) => { if (list.length && !list.some((c) => c.id === a.id)) setActive(null); })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <ActiveCampaignContext.Provider value={{ active, setActive }}>
      {children}
    </ActiveCampaignContext.Provider>
  );
}

export function useActiveCampaign() {
  const ctx = useContext(ActiveCampaignContext);
  if (!ctx) throw new Error("useActiveCampaign hors de ActiveCampaignProvider");
  return ctx;
}
