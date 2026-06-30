/**
 * useCampaign — etat de la campagne active (compagnons, quetes, presage).
 *
 * `refresh` permet de re-synchroniser apres un echange avec le Maitre du Jeu
 * (qui a pu creer un personnage, faire avancer une quete, etc.).
 */
import { useCallback, useEffect, useState } from "react";
import { getCampaignState } from "../api";

const EMPTY = { active: false, companions: [], quests: [], counts: {} };

export function useCampaign() {
  const [state, setState] = useState(EMPTY);

  const refresh = useCallback(() => {
    getCampaignState().then(setState).catch(() => setState(EMPTY));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { campaign: state, refresh };
}
