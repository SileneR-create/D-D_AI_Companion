/** Stockage du jeton JWT (localStorage) + fabrique d'en-tetes d'autorisation. */
const KEY = "dnd_token";

export function getToken() {
  return localStorage.getItem(KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(KEY, token);
  else localStorage.removeItem(KEY);
}

/** Fusionne l'en-tete Authorization si un jeton est present. */
export function authHeaders(extra = {}) {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : { ...extra };
}
