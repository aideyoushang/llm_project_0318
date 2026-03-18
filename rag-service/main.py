from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
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

    return app


app = create_app()

