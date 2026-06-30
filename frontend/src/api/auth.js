/** Appels d'authentification (inscription, connexion, profil). */
import { authHeaders, setToken } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function register({ username, password, email }) {
  const res = await call("/api/auth/register", { username, password, email: email || null });
  return res; // UserRead
}

export async function login({ username, password }) {
  const data = await call("/api/auth/login", { username, password });
  setToken(data.access_token);
  return data;
}

/** Profil de l'utilisateur connecte (null si non authentifie). */
export async function me() {
  try {
    const res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders() });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export function logout() {
  setToken(null);
}

/** POST JSON avec remontee d'erreur explicite (statut + detail backend). */
async function call(path, body) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Backend injoignable (verifiez qu'il tourne sur le port 8000).");
  }
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(detail || `Erreur serveur (${res.status}).`);
  }
  return res.json();
}

async function safeDetail(res) {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => x.msg).join(", ");
    return null;
  } catch {
    return null;
  }
}
