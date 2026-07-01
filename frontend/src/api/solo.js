/** Aventures en solo : génération + reprise. */
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function req(method, path, body) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method, headers: authHeaders(body ? { "Content-Type": "application/json" } : {}),
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch { throw new Error("Backend injoignable."); }
  if (!res.ok) {
    let d = `(${res.status})`;
    try { d = (await res.json()).detail || d; } catch { /* ignore */ }
    throw new Error(d);
  }
  return res.status === 204 ? null : res.json();
}

export const startSolo = (data) => req("POST", "/api/solo/start", data);
export async function listSoloAdventures() {
  try { return (await req("GET", "/api/solo/adventures")).adventures; } catch { return []; }
}
