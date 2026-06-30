/**
 * useSources — documents indexes dans le RAG + upload de nouveaux fichiers.
 */
import { useCallback, useEffect, useState } from "react";
import { listSources, uploadDocument } from "../api";

export function useSources() {
  const [sources, setSources] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(() => {
    listSources().then(setSources).catch(() => {});
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const upload = useCallback(async (file) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setSources(await uploadDocument(file));
    } catch (e) {
      setError(e?.message || "Echec de l'envoi");
    } finally {
      setBusy(false);
    }
  }, []);

  return { sources, busy, error, upload, refresh };
}
