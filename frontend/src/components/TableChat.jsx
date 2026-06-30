/**
 * Zone de chat de La Table, remontee (key) par campagne. L'historique persistant
 * est charge en `initial` ; campaignId fait persister les nouveaux messages.
 */
import { useEffect, useRef, useState } from "react";
import { RotateCw } from "lucide-react";
import { T, DISPLAY } from "../theme.js";
import { DOMAINS } from "../api";
import { useChatStream } from "../hooks/useChatStream.js";
import { Conversation, Inkwell } from "./chat.jsx";

export function TableChat({ campaignId, role, model, initial, sidebar, onTurn }) {
  const [input, setInput] = useState("");
  const { messages, draft, streaming, toolName, send, resend, pending } =
    useChatStream(DOMAINS.GAMEMASTER, { model, role, campaignId, initial });

  const wasStreaming = useRef(false);
  useEffect(() => {
    if (wasStreaming.current && !streaming) onTurn?.();
    wasStreaming.current = streaming;
  }, [streaming, onTurn]);

  const onSend = () => { send(input); setInput(""); };

  return (
    <div style={{ flex: 1, display: "flex", gap: 28, padding: "4px 48px 14px", minHeight: 0 }}>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", minHeight: 0 }}>
        <Conversation messages={messages} draft={draft} streaming={streaming} toolName={toolName} source="Gamemaster MCP" />
        {pending && !streaming && (
          <div style={{ display: "flex", justifyContent: "center", padding: "4px 0 0" }}>
            <button onClick={resend} className="glow" style={{
              display: "inline-flex", alignItems: "center", gap: 7, padding: "8px 16px", borderRadius: 3,
              background: "transparent", border: `1px solid ${T.gold}`, color: T.gold, cursor: "pointer",
              fontFamily: DISPLAY, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase" }}>
              <RotateCw size={13} /> Relancer la derniere demande
            </button>
          </div>
        )}
        <Inkwell value={input} setValue={setInput} onSend={onSend} disabled={streaming}
          placeholder={role === "dm" ? "Parle au Maitre du Jeu..." : "Parle a la table..."} />
      </div>
      {sidebar}
    </div>
  );
}
