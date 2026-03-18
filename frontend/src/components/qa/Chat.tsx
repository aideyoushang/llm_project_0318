"use client";

import { useMemo, useState } from "react";

import { chat } from "@/lib/qa";
import CitationBadge from "@/components/qa/CitationBadge";

type Reference = Record<string, unknown>;

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [references, setReferences] = useState([] as Reference[]);
  const [busy, setBusy] = useState(false);
  const apiUrl = useMemo(() => process.env.NEXT_PUBLIC_PYTHON_API_URL ?? "", []);

  async function onSend() {
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    setAnswer("");
    setReferences([]);
    try {
      const result = await chat({ apiUrl, question: q });
      setAnswer(result.answer);
      setReferences(result.references ?? []);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ display: "flex", gap: 12 }}>
        <input
          value={question}
          onChange={(e: any) => setQuestion(e.target.value)}
          placeholder="Ask about hotels based on TripAdvisor reviews..."
          style={{ flex: 1, padding: 10, border: "1px solid #ddd", borderRadius: 6 }}
        />
        <button
          onClick={onSend}
          disabled={busy}
          style={{ padding: "10px 14px", border: "1px solid #ddd", borderRadius: 6 }}
        >
          Send
        </button>
      </div>
      <div style={{ marginTop: 16, padding: 12, border: "1px solid #eee", borderRadius: 6 }}>
        <div style={{ whiteSpace: "pre-wrap" }}>{answer}</div>
        {references.length > 0 ? (
          <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 8 }}>
            {references.slice(0, 8).map((r: Reference, idx: number) => (
              <CitationBadge key={idx} reference={r} index={idx + 1} />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
