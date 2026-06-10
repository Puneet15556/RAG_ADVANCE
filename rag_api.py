"""
FastAPI wrapper around rag.py / pipeline.py.

- Loads models ONCE at startup (warm pool, no per-request reload).
- /search           -> full retrieval results (text + source + page + score)
- /answer           -> non-streaming JSON answer + book names
- /answer/stream    -> Server-Sent Events: token-by-token streaming  ⭐
- /                 -> browser UI that consumes the streaming endpoint

Install:
    pip install fastapi uvicorn python-dotenv

.env file (next to this file):
    GROQ_API_KEY=gsk_...

Run:
    uvicorn rag_api:app --host 0.0.0.0 --port 8000

Open:
    http://localhost:8000/
"""

from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path
from re import sub as re_sub, DOTALL as RE_DOTALL
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# Load .env first (so GROQ_API_KEY is available)
load_dotenv()

# Import existing pipeline logic - works for both rag.py and pipeline.py
try:
    import rag as core
except ImportError:
    import pipeline as core


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
    sources: List[str]


# ===================================================================
# HELPERS
# ===================================================================
def _unique_book_names(ranked) -> List[str]:
    """Deduplicated list of book filenames from a ranked result set."""
    seen: list[str] = []
    for hit, _score in ranked:
        name = Path(hit[2]).name
        if name not in seen:
            seen.append(name)
    return seen


def _build_answer_prompt(
    query: str,
    ranked,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> str:
    """Build prompt with optional conversation history + rolling summary."""
    context = "\n\n".join(
        f"[{i}] {hit[1]}" for i, (hit, _score) in enumerate(ranked, 1)
    )

    convo_block = ""
    if summary:
        convo_block += f"\n[Earlier conversation summary]\n{summary}\n"
    if history:
        convo_block += "\n[Recent conversation]\n"
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            convo_block += f"{role}: {msg.get('content', '')}\n"

    return (
        "You answer a question grounded ONLY in the document context below.\n\n"
        "STRICT RULES:\n"
        "1. Every factual claim MUST come from the [Document context]. "
        "   Do NOT use general knowledge, training data, or guesses.\n"
        "2. If the answer is not in the documents, reply with exactly: "
        "   'The documents don't cover that.' and stop. Do NOT speculate.\n"
        "3. Cite documents inline with [number] tags - only for claims "
        "   actually backed by that chunk.\n"
        "4. You may use the conversation history to understand what is "
        "   being asked, but never to invent facts.\n"
        "5. Be concise. Use markdown (bold, bullets) where it helps."
        f"{convo_block}\n"
        f"\n[Document context]\n{context}\n\n"
        f"User: {query}\nAssistant:"
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


# ───── Conditional query rewriting (fixes "he/she/this" follow-ups) ─────
_PRONOUNS = {
    "he", "she", "it", "they", "this", "that", "those", "these",
    "him", "her", "them", "his", "their", "its", "theirs",
}


def _needs_rewrite(query: str, history: list) -> bool:
    """Cheap heuristic: rewrite only when query is likely context-dependent."""
    if not history:
        return False
    words = set(query.lower().replace("?", "").replace(".", "").split())
    if words & _PRONOUNS:
        return True
    if len(query.split()) < 5:        # very short follow-up like "and Toda?"
        return True
    return False


def _rewrite_query(query: str, history: list) -> str:
    """Use the LLM to make a follow-up question retrieval-friendly."""
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in history[-4:]
    )
    prompt = (
        "You rewrite follow-up questions so retrieval can find the right docs.\n\n"
        "RULES:\n"
        "1. Replace pronouns (he/she/him/her/they/this/that) with explicit names.\n"
        "2. PREFER the person or entity the USER most recently asked about, "
        "   NOT a name that only appeared inside the assistant's reply.\n"
        "3. If the user's last question was 'Tell me about X' and the next "
        "   says 'when did he die' - rewrite using X, even if the assistant's "
        "   answer mentioned other people.\n"
        "4. If the question is already standalone, return it unchanged.\n"
        "5. Return ONLY the rewritten question on one line. No preamble, "
        "   no quotes, no explanation.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Latest question: {query}\n\n"
        "Standalone question:"
    )
    rewritten = _call_llm(prompt).strip().strip('"').strip("'")
    return rewritten if rewritten else query


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
    return {"status": "ok", "model": LLM_MODEL}


@app.get("/search", response_model=List[SearchHit])
def search_endpoint(q: str, top_k: int = 5):
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


class ChatMessage(BaseModel):
    role:    str          # "user" or "assistant"
    content: str


class StreamRequest(BaseModel):
    q:       str
    top_k:   int                       = 5
    history: List[ChatMessage]         = []
    summary: str                       = ""


@app.post("/answer/stream")
def answer_stream_endpoint(body: StreamRequest):
    """
    Streaming with conversation memory.
    Body: { q, top_k, history: [{role, content}, ...], summary }
    """
    q = body.q.strip()
    if not q:
        raise HTTPException(400, "empty query")

    # ── History-aware retrieval: rewrite if the query depends on context ──
    history = [m.model_dump() for m in body.history]
    search_q = q
    if _needs_rewrite(q, history):
        search_q = _rewrite_query(q, history)
        print(f"[rewrite] '{q}'  ->  '{search_q}'")

    ranked = core.search(search_q, top_k=body.top_k, verbose=False)
    if not ranked:
        def empty():
            yield 'data: {"type":"token","text":"No relevant documents found."}\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(empty(), media_type="text/event-stream")

    sources = _unique_book_names(ranked)
    # NOTE: send the ORIGINAL question to the LLM, not the rewritten one.
    # Rewrite only existed to help retrieval; the user-facing prompt stays natural.
    prompt  = _build_answer_prompt(q, ranked, history=history, summary=body.summary)

    def event_stream():
        for token in _stream_llm(prompt):
            cleaned = _clean(token)
            if cleaned:
                yield f"data: {json.dumps({'type':'token','text':cleaned})}\n\n"
        yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SummarizeRequest(BaseModel):
    messages:         List[ChatMessage]
    previous_summary: str = ""


@app.post("/summarize")
def summarize_endpoint(body: SummarizeRequest):
    """Compress old chat turns into a brief summary."""
    if not body.messages:
        return {"summary": body.previous_summary}

    lines = [f"{m.role.capitalize()}: {m.content}" for m in body.messages]
    convo = "\n".join(lines)

    prompt = (
        "Summarize the following conversation in 2-4 sentences. "
        "Preserve key facts, entities, and the user's apparent interests. "
        "Be neutral and concise.\n\n"
    )
    if body.previous_summary:
        prompt += f"Previous summary:\n{body.previous_summary}\n\nNew turns to incorporate:\n{convo}\n\nUpdated summary:"
    else:
        prompt += f"Conversation:\n{convo}\n\nSummary:"

    return {"summary": _call_llm(prompt)}


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
