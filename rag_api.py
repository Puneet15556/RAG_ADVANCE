"""
FastAPI wrapper around rag.py / pipeline.py.

- Loads models ONCE at startup (warm pool, no per-request reload).
- /search   -> full retrieval results (text + source + page + score)
- /answer   -> only the LLM answer; book names are returned separately so the
               frontend can hide them behind a "Show sources" button.
- /         -> minimal browser UI demonstrating the show/hide-sources flow.

Install:
    pip install fastapi uvicorn

Set your LLM key:
    $env:OPENAI_API_KEY = "sk-..."

Run:
    uvicorn rag_api:app --host 0.0.0.0 --port 8000

Open:
    http://localhost:8000/         (browser UI)
    http://localhost:8000/docs     (Swagger API explorer)
    
    
"""

from __future__ import annotations
import rag as core
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import json


from re import sub as re_sub, DOTALL as RE_DOTALL                              

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()







# ===================================================================
# LIFESPAN: load models once at startup
# ===================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] loading models + index (one-time)...")
    core.load_search_components()
    print("[api] ready")
    yield
    print("[api] shutting down")


app = FastAPI(title="RAG API", lifespan=lifespan)


# ===================================================================
# LLM CLIENT — Groq (OpenAI-compatible, fast)
# ===================================================================
LLM_MODEL = "llama-3.1-8b-instant"   # ~0.5s. Swap to "llama-3.3-70b-versatile" for quality.

_groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ===================================================================
# RESPONSE MODELS
# ===================================================================
class SearchHit(BaseModel):
    score:  float
    text:   str
    source: str
    page:   int


class AnswerResponse(BaseModel):
    answer:  str
    sources: List[str]    # unique book names; hidden in UI until user clicks


# ===================================================================
# HELPERS
# ===================================================================
def _unique_book_names(ranked) -> List[str]:
    """Deduplicated list of book filenames from a ranked result set."""
    seen: list[str] = []
    for hit, _score in ranked:
        name = Path(hit[2]).name           # hit = (rowid, text, source, page, doc_id)
        if name not in seen:
            seen.append(name)
    return seen


def _build_answer_prompt(query: str, ranked) -> str:
    context = "\n\n".join(
        f"[{i}] {hit[1]}" for i, (hit, _score) in enumerate(ranked, 1)
    )
    return (
        "Answer the question using ONLY the context below. "
        "Be concise. Use markdown formatting (bold, bullets) where helpful. "
        "If the context does not contain the answer, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def _clean(text: str) -> str:
    """Strip reasoning/tool-call leftovers."""
    text = re_sub(r"<think>.*?</think>", "", text, flags=RE_DOTALL)
    text = re_sub(r"<tool_call>.*?</tool_call>", "", text, flags=RE_DOTALL)
    return text


def _call_llm(prompt: str) -> str:
    """Non-streaming LLM call."""
    response = _groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _clean(response.choices[0].message.content or "").strip()


def _stream_llm(prompt: str):
    """Generator yielding LLM tokens one chunk at a time."""
    response = _groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,                       # ← the magic
    )
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            yield token


        


# ===================================================================
# ENDPOINTS
# ===================================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=List[SearchHit])
def search_endpoint(q: str, top_k: int = 5):
    """Full retrieval: top-K reranked chunks with text + metadata."""
    if not q.strip():
        raise HTTPException(400, "empty query")

    ranked = core.search(q, top_k=top_k, verbose=False)
    return [
        SearchHit(
            score  = float(score),
            text   = hit[1],
            source = Path(hit[2]).name,
            page   = int(hit[3]),
        )
        for hit, score in ranked
    ]


@app.get("/answer", response_model=AnswerResponse)
def answer_endpoint(q: str, top_k: int = 5):
    """Non-streaming: returns full answer in one JSON blob."""
    if not q.strip():
        raise HTTPException(400, "empty query")
    ranked = core.search(q, top_k=top_k, verbose=False)
    if not ranked:
        return AnswerResponse(answer="No relevant documents found.", sources=[])
    prompt = _build_answer_prompt(q, ranked)
    return AnswerResponse(
        answer=_call_llm(prompt),
        sources=_unique_book_names(ranked),
    )
    


@app.get("/answer/stream")
def answer_stream_endpoint(q: str, top_k: int = 5):
    """
    Streaming: sends tokens as Server-Sent Events.

    Event format:
        data: {"type": "token", "text": "..."}\\n\\n
        data: {"type": "sources", "sources": [...]}\\n\\n
        data: [DONE]\\n\\n
    """
    if not q.strip():
        raise HTTPException(400, "empty query")

    ranked = core.search(q, top_k=top_k, verbose=False)
    if not ranked:
        def empty():
            yield 'data: {"type":"token","text":"No relevant documents found."}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(empty(), media_type="text/event-stream")

    sources = _unique_book_names(ranked)
    prompt  = _build_answer_prompt(q, ranked)

    def event_stream():
        # Stream LLM tokens (clean each one cheaply on the way out)
        for token in _stream_llm(prompt):
            cleaned = _clean(token)
            if cleaned:
                yield f"data: {json.dumps({'type':'token','text':cleaned})}\n\n"
        # After completion, send the source list
        yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    


# ===================================================================
# BROWSER UI — served from chat_ui.html (next to this file)
# ===================================================================
_CHAT_UI_PATH = Path(__file__).parent / "chat_ui.html"


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(_CHAT_UI_PATH)


# ── legacy embedded HTML kept below for reference; not used ───────
_HOMEPAGE = """<!DOCTYPE html>
<html>
<head>
<title>RAG Chat</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; }
  h1 { color: #1a3d6d; }
  #q { width: 70%; padding: 0.6em; font-size: 1em; }
  button { padding: 0.6em 1.2em; font-size: 1em; cursor: pointer; }
  #answer { margin-top: 1.5em; padding: 1em; background: #f6f8fb; border-radius: 8px;
            min-height: 2em; line-height: 1.5; }
  #answer p { margin: 0.4em 0; }
  #answer ul, #answer ol { margin: 0.4em 0 0.4em 1.4em; padding: 0; }
  #answer strong { color: #1a3d6d; }
  #srcBtn { display: none; margin-top: 1em; background: #eef; border: 1px solid #99c; }
  #sources { display: none; margin-top: 0.5em; padding: 0.8em; background: #fffbe6;
             border-left: 4px solid #d8a44a; border-radius: 4px; }
  #sources ul { margin: 0.4em 0 0 1.2em; padding: 0; }
  #status { color: #888; font-size: 0.85em; margin-top: 0.4em; }
</style>
</head>
<body>
<h1>Ask your documents</h1>

<input id="q" placeholder="What is Soka Gakkai about?" />
<button onclick="ask()">Ask</button>

<div id="answer">Type a question above and click Ask.</div>
<div id="status"></div>

<button id="srcBtn" onclick="toggleSources()">Show sources</button>
<div id="sources"></div>

<script>
let lastSources = [];
let rawAnswer  = "";

function setStatus(msg) { document.getElementById('status').innerText = msg; }

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;

  rawAnswer = "";
  document.getElementById('answer').innerHTML = "";
  document.getElementById('srcBtn').style.display = 'none';
  document.getElementById('sources').style.display = 'none';
  setStatus('Searching...');

  const t0 = performance.now();
  let firstToken = true;

  try {
    const response = await fetch('/answer/stream?q=' + encodeURIComponent(q));
    if (!response.ok) throw new Error('HTTP ' + response.status);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines are separated by blank lines (\\n\\n)
      const lines = buffer.split('\\n\\n');
      buffer = lines.pop();               // keep partial line for next iteration

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') {
          setStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
          continue;
        }
        try {
          const msg = JSON.parse(payload);
          if (msg.type === 'token') {
            if (firstToken) {
              setStatus(`First token in ${((performance.now() - t0) / 1000).toFixed(1)}s...`);
              firstToken = false;
            }
            rawAnswer += msg.text;
            document.getElementById('answer').innerHTML = marked.parse(rawAnswer);
          } else if (msg.type === 'sources') {
            lastSources = msg.sources || [];
            if (lastSources.length) {
              document.getElementById('srcBtn').style.display = 'inline-block';
              document.getElementById('srcBtn').innerText = 'Show sources';
            }
          }
        } catch (e) { /* ignore parse errors */ }
      }
    }
  } catch (e) {
    document.getElementById('answer').innerText = 'Error: ' + e.message;
    setStatus('');
  }
}

function toggleSources() {
  const div = document.getElementById('sources');
  const btn = document.getElementById('srcBtn');
  if (div.style.display === 'none' || div.style.display === '') {
    div.innerHTML = '<b>Sources:</b><ul>' +
      lastSources.map(s => '<li>' + s + '</li>').join('') + '</ul>';
    div.style.display = 'block';
    btn.innerText = 'Hide sources';
  } else {
    div.style.display = 'none';
    btn.innerText = 'Show sources';
  }
}

document.getElementById('q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') ask();
});
</script>
</body>
</html>"""


# (Old embedded `home()` removed — new one defined above serves chat_ui.html)
