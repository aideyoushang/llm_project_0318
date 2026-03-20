from __future__ import annotations

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
        return StreamingResponse(rag.chat_stream(payload), media_type="text/event-stream")

    return app


app = create_app()
