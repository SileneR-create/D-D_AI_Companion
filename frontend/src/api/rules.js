/** Recherche rapide de regles (SRD) : index + detail. */
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function getReferenceIndex() {
  try {
    const r = await fetch(`${API_BASE}/api/rules/reference`, { headers: authHeaders() });
    if (!r.ok) return { available: false, items: [] };
    return r.json();
  } catch { return { available: false, items: [] }; }
}

export async function getReferenceDetail(type, index) {
  try {
    const r = await fetch(`${API_BASE}/api/rules/reference/${type}/${index}`, { headers: authHeaders() });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}
