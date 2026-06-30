/**
 * Client API — LE point de contact unique avec le backend FastAPI.
 *
 * Tout le reste du frontend passe par ce module ; aucun composant ne doit
 * appeler `fetch` directement. C'est le pendant cote React du contrat defini
 * dans `backend/schemas.py`.
 *
 * Endpoints :
 *   GET  /api/health                  -> { status }
 *   GET  /api/{domain}/models         -> { models: string[] }
 *   POST /api/{domain}/chat           -> flux SSE (text/event-stream)
 *   GET  /api/rules/sources           -> { sources: RagSource[] }
 *   POST /api/rules/documents         -> { sources: RagSource[] }   (multipart)
 *   GET  /api/gamemaster/state        -> CampaignState
 *
 * `domain` vaut "rules" (assistant Regles) ou "gamemaster" (Maitre du Jeu).
 */

// En dev, Vite proxifie "/api" vers http://localhost:8000 (voir vite.config.js).
import { authHeaders } from "./token.js";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/** Domaines disponibles, alignes sur les routeurs FastAPI. */
export const DOMAINS = Object.freeze({
  RULES: "rules",
  GAMEMASTER: "gamemaster",
});

/** Verifie l'etat du backend. */
export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Health check echoue (${res.status})`);
  return res.json();
}

/** Liste les modeles Ollama disponibles pour un domaine. */
export async function listModels(domain) {
  const res = await fetch(`${API_BASE}/api/${domain}/models`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`listModels(${domain}) -> ${res.status}`);
  const data = await res.json();
  return data.models;
}

/**
 * Ouvre un chat en streaming et invoque des callbacks au fil des evenements SSE.
 *
 * @param {string} domain - "rules" | "gamemaster".
 * @param {object} payload - { message, history, model } (cf. ChatRequest).
 * @param {object} handlers - { onToken, onClear, onToolCall, onDone, onError }.
 * @param {AbortSignal} [signal] - pour annuler la requete.
 */
export async function streamChat(domain, payload, handlers = {}, signal) {
  const { onToken, onClear, onToolCall, onDone, onError } = handlers;

  let res;
  try {
    res = await fetch(`${API_BASE}/api/${domain}/chat`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        message: payload.message,
        history: payload.history ?? [],
        model: payload.model,
        role: payload.role ?? "dm",
        campaign_id: payload.campaignId ?? null,
        resend: payload.resend ?? false,
      }),
      signal,
    });
  } catch (e) {
    if (e?.name === "AbortError") return;
    onError?.("Impossible de joindre l'API. Le backend est-il lance sur le port 8000 ?");
    return;
  }

  if (!res.ok || !res.body) {
    onError?.(`Requete /chat echouee (${res.status}).`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Les evenements SSE sont separes par une ligne vide.
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        let evt;
        try {
          evt = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }
        dispatch(evt, { onToken, onClear, onToolCall, onDone, onError });
      }
    }
  } catch (e) {
    if (e?.name !== "AbortError") onError?.("Flux interrompu : " + (e?.message || e));
  }
}

/** Aiguille un evenement SSE vers le bon callback. */
function dispatch(evt, { onToken, onClear, onToolCall, onDone, onError }) {
  switch (evt.type) {
    case "token":
      onToken?.(evt.content ?? "");
      break;
    case "clear":
      onClear?.();
      break;
    case "tool_call":
      onToolCall?.(evt.name ?? "");
      break;
    case "done":
      onDone?.();
      break;
    case "error":
      onError?.(evt.content ?? "Erreur inconnue");
      break;
    default:
      break;
  }
}

/** Recupere l'etat synthetique de la campagne active (cote Gamemaster). */
export async function getCampaignState() {
  const res = await fetch(`${API_BASE}/api/gamemaster/state`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`getCampaignState -> ${res.status}`);
  return res.json();
}

/** Liste les documents indexes dans le RAG (assistant Regles). */
export async function listSources() {
  const res = await fetch(`${API_BASE}/api/rules/sources`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`listSources -> ${res.status}`);
  const data = await res.json();
  return data.sources;
}

/**
 * Aspire tout un site (meme domaine) et l'indexe dans le RAG.
 * @param {string} url
 * @param {number} [maxPages]
 */
export async function crawlSite(url, maxPages) {
  const res = await fetch(`${API_BASE}/api/rules/crawl`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ url, max_pages: maxPages || null }),
  });
  if (!res.ok) {
    let d = `(${res.status})`;
    try { d = (await res.json()).detail || d; } catch { /* ignore */ }
    throw new Error(d);
  }
  return (await res.json()).sources;
}

/**
 * Envoie un document (PDF / txt / md) a indexer dans le RAG.
 * @param {File} file
 * @returns {Promise<Array>} la liste des sources mise a jour.
 */
export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/rules/documents`, { method: "POST", headers: authHeaders(), body: form });
  if (!res.ok) {
    let detail = `(${res.status})`;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  const data = await res.json();
  return data.sources;
}
