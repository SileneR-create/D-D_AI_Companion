/** Catalogue d'objets D&D (SRD), detail, attribution, monstres. */
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function _get(path, fallback) {
  try {
    const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
    if (!res.ok) return fallback;
    return res.json();
  } catch { return fallback; }
}

export const getCatalog = () => _get("/api/arsenal/catalog", { available: false, categories: [] });
export const getItemDetail = (kind, index) => _get(`/api/arsenal/item/${kind}/${index}`, null);
export async function getMonsters() { return (await _get("/api/arsenal/monsters", { monsters: [] })).monsters; }
export const getMonsterDetail = (index) => _get(`/api/arsenal/monster/${index}`, null);

export async function assignArsenal(data) {
  const res = await fetch(`${API_BASE}/api/arsenal/assign`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(data),
  });
  if (!res.ok) { let d = `(${res.status})`; try { d = (await res.json()).detail || d; } catch { /* */ } throw new Error(d); }
  return res.json();
}
