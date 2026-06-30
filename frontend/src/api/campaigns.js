/** Appels campagnes, PNJ, lieux, quetes et recrutement. */
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function req(method, path, body) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: authHeaders(body ? { "Content-Type": "application/json" } : {}),
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error("Backend injoignable.");
  }
  if (!res.ok) {
    let d = `(${res.status})`;
    try { d = (await res.json()).detail || d; } catch { /* ignore */ }
    throw new Error(d);
  }
  return res.status === 204 ? null : res.json();
}

// Campagnes
export const createCampaign = (data) => req("POST", "/api/gamemaster/campaigns", data);
export const activateCampaign = (id) => req("POST", `/api/gamemaster/campaigns/${id}/activate`, {});
export async function listCampaigns() {
  try { return (await req("GET", "/api/gamemaster/campaigns")).campaigns; } catch { return []; }
}
export const endCampaign = (id) => req("POST", `/api/gamemaster/campaigns/${id}/end`, {});
export const deleteCampaign = (id) => req("DELETE", `/api/gamemaster/campaigns/${id}`);
export async function getArchive(id) {
  try { return await req("GET", `/api/gamemaster/campaigns/${id}/archive`); } catch { return null; }
}

// PNJ
export const createNpc = (cid, data) => req("POST", `/api/gamemaster/campaigns/${cid}/npcs`, data);
export const deleteNpc = (cid, id) => req("DELETE", `/api/gamemaster/campaigns/${cid}/npcs/${id}`);
export async function listNpcs(cid) {
  try { return (await req("GET", `/api/gamemaster/campaigns/${cid}/npcs`)).npcs; } catch { return []; }
}

// Lieux
export const createLocation = (cid, data) => req("POST", `/api/gamemaster/campaigns/${cid}/locations`, data);
export async function listLocations(cid) {
  try { return (await req("GET", `/api/gamemaster/campaigns/${cid}/locations`)).locations; } catch { return []; }
}

// Quetes
export const createQuest = (cid, data) => req("POST", `/api/gamemaster/campaigns/${cid}/quests`, data);
export const updateQuest = (cid, qid, data) => req("PATCH", `/api/gamemaster/campaigns/${cid}/quests/${qid}`, data);
export const deleteQuest = (cid, qid) => req("DELETE", `/api/gamemaster/campaigns/${cid}/quests/${qid}`);
export async function listQuests(cid) {
  try { return (await req("GET", `/api/gamemaster/campaigns/${cid}/quests`)).quests; } catch { return []; }
}

// Recrutement
export const recruitCharacter = (cid, characterId) =>
  req("POST", `/api/gamemaster/campaigns/${cid}/recruit`, { character_id: characterId });

// Historique de conversation
export async function listMessages(cid) {
  try { return (await req("GET", `/api/gamemaster/campaigns/${cid}/messages`)).messages; } catch { return []; }
}
