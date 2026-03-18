import { postJson } from "@/lib/api";

export type ChatRequest = {
  apiUrl: string;
  question: string;
};

export type ChatResponse = {
  answer: string;
  references?: Record<string, unknown>[];
  intent?: Record<string, unknown>;
};

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  const base = req.apiUrl.trim();
  if (!base) {
    throw new Error("Missing NEXT_PUBLIC_PYTHON_API_URL");
  }
  const url = `${base.replace(/\/+$/, "")}/api/v1/chat`;
  return await postJson<ChatResponse>(url, { question: req.question });
}

