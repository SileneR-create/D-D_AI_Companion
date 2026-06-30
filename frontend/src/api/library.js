/** Bibliotheque transversale : elements crees + activation par campagne. */
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function req(method, path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method, headers: authHeaders(body ? { "Content-Type": "application/json" } : {}),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`(${res.status})`);
  return res.status === 204 ? null : res.json();
}

export async function getLibrary() {
  try { return await req("GET", "/api/library"); }
  catch { return { campaigns: [], npcs: [], locations: [], items: [] }; }
}
export const createLibraryNpc = (d) => req("POST", "/api/library/npc", d);
export const createLibraryLocation = (d) => req("POST", "/api/library/location", d);
export const activateElement = (d) => req("POST", "/api/library/activate", d);
export const deactivateElement = (d) => req("POST", "/api/library/deactivate", d);
