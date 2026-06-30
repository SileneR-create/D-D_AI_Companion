/** Recupere la liste des modeles Ollama disponibles pour un domaine. */
import { useEffect, useState } from "react";
import { listModels } from "../api/client.js";

export function useModels(domain) {
  const [models, setModels] = useState([]);
  useEffect(() => {
    let alive = true;
    listModels(domain).then((m) => alive && setModels(m)).catch(() => {});
    return () => { alive = false; };
  }, [domain]);
  return models;
}
