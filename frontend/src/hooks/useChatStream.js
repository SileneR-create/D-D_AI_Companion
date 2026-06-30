/**
 * useChatStream — hook React encapsulant une conversation en streaming.
 *
 * `send` envoie un nouveau message ; `resend` relance la derniere demande restee
 * sans reponse (ex: page changee) SANS la dupliquer.
 */
import { useCallback, useRef, useState } from "react";
import { streamChat } from "../api/client.js";

export function useChatStream(domain, { model = "mistral:7b-instruct", initial = [], role = "dm", campaignId = null } = {}) {
  const [messages, setMessages] = useState(initial);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [toolName, setToolName] = useState(null);
  const abortRef = useRef(null);

  const run = useCallback(
    async (text, history, { resend = false } = {}) => {
      setDraft("");
      setStreaming(true);
      setToolName(null);
      const controller = new AbortController();
      abortRef.current = controller;
      let acc = "";
      let lastTool = null;
      let settled = false;
      try {
        await streamChat(
          domain,
          { message: text, history, model, role, campaignId, resend },
          {
            onToken: (t) => { acc += t; setDraft(acc); },
            onClear: () => { acc = ""; setDraft(""); },
            onToolCall: (name) => { lastTool = name; setToolName(name); },
            onDone: () => { settled = true; setMessages((prev) => [...prev, { role: "assistant", content: acc, tool: lastTool }]); },
            onError: (msg) => { settled = true; setMessages((prev) => [...prev, { role: "assistant", content: `Le voile se trouble : ${msg}` }]); },
          },
          controller.signal,
        );
        if (!settled && acc) setMessages((prev) => [...prev, { role: "assistant", content: acc, tool: lastTool }]);
      } finally {
        setDraft(""); setStreaming(false); setToolName(null);
      }
    },
    [domain, model, role, campaignId],
  );

  const send = useCallback(
    async (text) => {
      if (!text?.trim() || streaming) return;
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      await run(text, history);
    },
    [messages, streaming, run],
  );

  const resend = useCallback(
    async () => {
      if (streaming) return;
      const last = messages[messages.length - 1];
      if (!last || last.role !== "user") return;
      const history = messages.slice(0, -1).map((m) => ({ role: m.role, content: m.content }));
      await run(last.content, history, { resend: true });
    },
    [messages, streaming, run],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const pending = messages.length > 0 && messages[messages.length - 1].role === "user";

  return { messages, draft, streaming, toolName, send, resend, stop, pending };
}
