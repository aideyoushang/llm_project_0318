from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

SERVICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_DIR.parent

sys.path.insert(0, SERVICE_DIR.as_posix())
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, SRC_DIR.as_posix())

from rag_service.modules.rag_system import RagSystem  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(title="hotel-review-rag-service")

    allow_origins_env = os.environ.get("ALLOW_ORIGINS", "*")
    allow_origins = [o.strip() for o in allow_origins_env.split(",") if o.strip()]
    if not allow_origins:
        allow_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag = RagSystem()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/chat")
    def chat(payload: dict) -> dict:
        return rag.chat(payload)

    @app.post("/api/v1/chat/stream")
    def chat_stream(payload: dict) -> StreamingResponse:
        return StreamingResponse(
            rag.chat_stream(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/demo")
    def demo() -> HTMLResponse:
        html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>hotel-review-rag-service demo</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }
      .row { display: flex; gap: 8px; align-items: center; }
      input { flex: 1; padding: 8px; }
      button { padding: 8px 12px; }
      pre { background: #f6f8fa; padding: 12px; overflow: auto; }
      .refs { display: grid; gap: 8px; }
      .ref { border: 1px solid #e5e7eb; padding: 10px; border-radius: 8px; }
      .meta { font-size: 12px; color: #6b7280; margin-bottom: 6px; }
      .stage { font-size: 12px; color: #111827; }
      .claims { display: grid; gap: 8px; }
      .claim { border: 1px solid #e5e7eb; padding: 10px; border-radius: 8px; }
    </style>
  </head>
  <body>
    <h2>RAG SSE Demo</h2>
    <div class="row">
      <input id="q" value="How is the hotel location recently?" />
      <button id="go">Send</button>
      <button id="stop">Stop</button>
    </div>
    <p class="stage" id="stage"></p>
    <h3>Answer</h3>
    <pre id="answer"></pre>
    <h3>Claims</h3>
    <div class="claims" id="claims"></div>
    <h3>References</h3>
    <div class="refs" id="refs"></div>
    <h3>Intent</h3>
    <pre id="intent"></pre>
    <script>
      const $ = (id) => document.getElementById(id);
      let ctrl = null;

      function resetUI() {
        $("stage").textContent = "";
        $("answer").textContent = "";
        $("intent").textContent = "";
        $("claims").innerHTML = "";
        $("refs").innerHTML = "";
      }

      function renderClaims(claims) {
        $("claims").innerHTML = "";
        for (const c of claims) {
          const el = document.createElement("div");
          el.className = "claim";
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = `ref_ids=${(c.ref_ids || []).join(",")}`;
          const txt = document.createElement("div");
          txt.textContent = c.text || "";
          el.appendChild(meta);
          el.appendChild(txt);
          $("claims").appendChild(el);
        }
      }

      function renderRefs(refs) {
        $("refs").innerHTML = "";
        for (const r of refs) {
          const el = document.createElement("div");
          el.className = "ref";
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = `doc_id=${r.doc_id} chunk_id=${r.chunk_id} post_date=${r.post_date} source=${r.source}`;
          const txt = document.createElement("div");
          txt.textContent = r.chunk_text || "";
          el.appendChild(meta);
          el.appendChild(txt);
          $("refs").appendChild(el);
        }
      }

      async function run() {
        if (ctrl) ctrl.abort();
        ctrl = new AbortController();
        resetUI();

        const resp = await fetch("/api/v1/chat/stream", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question: $("q").value }),
          signal: ctrl.signal,
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\\n\\n")) !== -1) {
            const raw = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            const line = raw.split("\\n").find((l) => l.startsWith("data:"));
            if (!line) continue;
            const dataStr = line.slice(5).trim();
            if (dataStr === "[DONE]") return;
            let obj;
            try { obj = JSON.parse(dataStr); } catch { continue; }
            if (obj.type === "stage") $("stage").textContent = `stage: ${obj.content}`;
            if (obj.type === "answer_chunk") $("answer").textContent += obj.content;
            if (obj.type === "claims") renderClaims(obj.content || []);
            if (obj.type === "references") renderRefs(obj.content || []);
            if (obj.type === "intent") $("intent").textContent = JSON.stringify(obj.content, null, 2);
          }
        }
      }

      $("go").addEventListener("click", () => run());
      $("stop").addEventListener("click", () => { if (ctrl) ctrl.abort(); });
    </script>
  </body>
</html>
"""
        return HTMLResponse(html)

    return app


app = create_app()
