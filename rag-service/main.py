from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
        def event_gen():
            result = rag.chat(payload)
            answer = str(result.get("answer") or "")
            references = result.get("references") or []
            intent = result.get("intent") or {}
            chunk_size = 200
            for i in range(0, len(answer), chunk_size):
                data = json.dumps({"type": "answer_chunk", "content": answer[i : i + chunk_size]}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            data = json.dumps({"type": "references", "content": references}, ensure_ascii=False)
            yield f"data: {data}\n\n"
            data = json.dumps({"type": "intent", "content": intent}, ensure_ascii=False)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app


app = create_app()
