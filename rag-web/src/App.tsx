import { useMemo, useRef, useState } from 'react'
import './App.css'

type Stage =
  | 'intent'
  | 'retrieve'
  | 'rerank'
  | 'rerank_llm'
  | 'generate'
  | 'generate_fallback'
  | 'finalize'

type SseEvent =
  | { type: 'stage'; content: Stage }
  | { type: 'answer_chunk'; content: string }
  | { type: 'claims'; content: Claim[] }
  | { type: 'references'; content: Reference[] }
  | { type: 'intent'; content: unknown }
  | { type: 'retriever_status'; content: unknown }

type Claim = { text: string; ref_ids: number[] }

type Reference = {
  doc_id?: number
  chunk_id?: number
  post_date?: string
  source?: string
  chunk_text?: string
  sources?: Record<string, { rank: number; w: number }>
  score?: number
}

const API_STREAM_PATH = '/api/v1/chat/stream'

function App() {
  const [question, setQuestion] = useState('How is the hotel location recently?')
  const [stage, setStage] = useState<string>('')
  const [answer, setAnswer] = useState('')
  const [claims, setClaims] = useState<Claim[]>([])
  const [references, setReferences] = useState<Reference[]>([])
  const [intent, setIntent] = useState<unknown>(null)
  const [selectedClaimIdx, setSelectedClaimIdx] = useState<number | null>(null)
  const [running, setRunning] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const selectedRefIds = useMemo(() => {
    if (selectedClaimIdx === null) return new Set<number>()
    const c = claims[selectedClaimIdx]
    if (!c || !Array.isArray(c.ref_ids)) return new Set<number>()
    return new Set<number>(c.ref_ids)
  }, [claims, selectedClaimIdx])

  async function run() {
    if (!question.trim()) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setRunning(true)
    setStage('')
    setAnswer('')
    setClaims([])
    setReferences([])
    setIntent(null)
    setSelectedClaimIdx(null)

    const resp = await fetch(API_STREAM_PATH, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: ctrl.signal,
    })

    if (!resp.ok || !resp.body) {
      setStage(`http_error:${resp.status}`)
      setRunning(false)
      return
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx = buf.indexOf('\n\n')
        while (idx !== -1) {
          const rawEvent = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          idx = buf.indexOf('\n\n')
          const dataLine = rawEvent
            .split('\n')
            .map((l) => l.trimEnd())
            .find((l) => l.startsWith('data:'))
          if (!dataLine) continue
          const dataStr = dataLine.slice(5).trim()
          if (!dataStr) continue
          if (dataStr === '[DONE]') {
            setRunning(false)
            return
          }
          let obj: unknown
          try {
            obj = JSON.parse(dataStr)
          } catch {
            continue
          }
          const ev = obj as SseEvent
          if (ev.type === 'stage') setStage(ev.content)
          if (ev.type === 'answer_chunk') setAnswer((prev) => prev + ev.content)
          if (ev.type === 'claims') setClaims(ev.content || [])
          if (ev.type === 'references') setReferences(ev.content || [])
          if (ev.type === 'intent') setIntent(ev.content)
        }
      }
    } finally {
      setRunning(false)
    }
  }

  function stop() {
    abortRef.current?.abort()
    abortRef.current = null
    setRunning(false)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="title">Hotel Review RAG</div>
        <div className="status">{running ? 'running' : stage ? `stage:${stage}` : 'idle'}</div>
      </header>

      <section className="controls">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question in English…"
          disabled={running}
        />
        <button onClick={run} disabled={running || !question.trim()}>
          Send
        </button>
        <button onClick={stop} disabled={!running}>
          Stop
        </button>
      </section>

      <main className="grid">
        <section className="panel">
          <div className="panelTitle">Answer</div>
          <pre className="answer">{answer || (running ? '' : '—')}</pre>
        </section>

        <section className="panel">
          <div className="panelTitle">Claims</div>
          <div className="list">
            {claims.length ? (
              claims.map((c, idx) => (
                <button
                  key={idx}
                  className={idx === selectedClaimIdx ? 'item itemActive' : 'item'}
                  onClick={() => setSelectedClaimIdx(idx === selectedClaimIdx ? null : idx)}
                  type="button"
                >
                  <div className="itemMeta">ref_ids: {(c.ref_ids || []).join(', ')}</div>
                  <div className="itemText">{c.text}</div>
                </button>
              ))
            ) : (
              <div className="empty">—</div>
            )}
          </div>
        </section>

        <section className="panel panelWide">
          <div className="panelTitle">References</div>
          <div className="list">
            {references.length ? (
              references.map((r, idx) => (
                <div
                  key={`${r.doc_id ?? 'd'}:${r.chunk_id ?? idx}`}
                  className={selectedRefIds.has(idx) ? 'ref refActive' : 'ref'}
                >
                  <div className="refMeta">
                    <span>ref_id:{idx}</span>
                    <span>doc_id:{r.doc_id ?? '—'}</span>
                    <span>chunk_id:{r.chunk_id ?? '—'}</span>
                    <span>date:{r.post_date ?? '—'}</span>
                    <span>source:{r.source ?? '—'}</span>
                  </div>
                  <div className="refText">{r.chunk_text ?? ''}</div>
                </div>
              ))
            ) : (
              <div className="empty">—</div>
            )}
          </div>
        </section>

        <section className="panel panelWide">
          <div className="panelTitle">Intent</div>
          <pre className="json">{intent ? JSON.stringify(intent, null, 2) : '—'}</pre>
        </section>
      </main>
    </div>
  )
}

export default App
