/** Appels Forge : bibliotheque d'objets + assignation. */
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
  } catch { throw new Error("Backend injoignable."); }
  if (!res.ok) {
    let d = `(${res.status})`;
    try { d = (await res.json()).detail || d; } catch { /* ignore */ }
    throw new Error(d);
  }
  return res.status === 204 ? null : res.json();
}

/** "12 po . 3 pa . 5 pc" a partir des champs gp/sp/cp (vide si tout a 0). */
export function fmtCoins(i) {
  const parts = [];
  if (i?.gp) parts.push(`${i.gp} po`);
  if (i?.sp) parts.push(`${i.sp} pa`);
  if (i?.cp) parts.push(`${i.cp} pc`);
  return parts.join(" · ");
}

export const createItem = (data) => req("POST", "/api/forge/items", data);
export const deleteItem = (id) => req("DELETE", `/api/forge/items/${id}`);
export async function listItems() {
  try { return (await req("GET", "/api/forge/items")).items; } catch { return []; }
}
export const assignItem = (data) => req("POST", "/api/forge/assign", data);
export const deleteAssignment = (id) => req("DELETE", `/api/forge/assignments/${id}`);
export async function listAssignments(targetType, targetId) {
  try { return (await req("GET", `/api/forge/assignments/${targetType}/${targetId}`)).assignments; }
  catch { return []; }
}
