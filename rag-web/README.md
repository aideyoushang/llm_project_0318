# rag-web

This is a minimal web UI for the hotel-review RAG backend.

## Features

- POST SSE client for `/api/v1/chat/stream`
- Live answer streaming
- Claims list with reference highlighting
- References and intent display

## Run (dev)

Start backend first (on the same server):

```bash
cd /root/csw_test
python -m uvicorn main:app --app-dir rag-service --host 0.0.0.0 --port 8000
```

Start frontend:

```bash
cd /root/csw_test/rag-web
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Vite proxies `/api/*` and `/healthz` to `http://127.0.0.1:8000` via [vite.config.ts](vite.config.ts).
