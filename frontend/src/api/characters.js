/** Appels personnages (avec ou sans campagne). */
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function createCharacter(data) {
  let res;
  try {
    res = await fetch(`${API_BASE}/api/characters`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(data),
    });
  } catch {
    throw new Error("Backend injoignable.");
  }
  if (!res.ok) {
    let d = `(${res.status})`;
    try { d = (await res.json()).detail || d; } catch { /* ignore */ }
    throw new Error(d);
  }
  return res.json();
}

export async function listCharacters() {
  const res = await fetch(`${API_BASE}/api/characters`, { headers: authHeaders() });
  if (!res.ok) return [];
  return (await res.json()).characters;
}

export async function deleteCharacter(id) {
  await fetch(`${API_BASE}/api/characters/${id}`, { method: "DELETE", headers: authHeaders() });
}

/** Met a jour la bourse (po/pa/pc) d'un personnage. */
export async function updateMoney(id, money) {
  const res = await fetch(`${API_BASE}/api/characters/${id}/money`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(money),
  });
  if (!res.ok) throw new Error(`(${res.status})`);
  return res.json();
}

/** Met a jour les champs editables de la feuille de personnage. */
export async function updateSheet(id, fields) {
  const res = await fetch(`${API_BASE}/api/characters/${id}/sheet`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(fields),
  });
  if (!res.ok) throw new Error(`(${res.status})`);
  return res.json();
}

/** Sorts disponibles pour une classe a un niveau donne. */
export async function getClassSpells(cls, level) {
  try {
    const res = await fetch(
      `${API_BASE}/api/characters/spells?cls=${encodeURIComponent(cls)}&level=${level}`,
      { headers: authHeaders() },
    );
    if (!res.ok) return { caster: false, max_spell_level: 0, spells: [] };
    return res.json();
  } catch {
    return { caster: false, max_spell_level: 0, spells: [] };
  }
}

/** Detail d'un sort (description, ecole, portee, duree). */
export async function getSpellDetail(index) {
  try {
    const res = await fetch(`${API_BASE}/api/characters/spell/${index}`, { headers: authHeaders() });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
