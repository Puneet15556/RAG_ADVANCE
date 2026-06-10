# 📚 RAG Chat — Production-Grade Retrieval-Augmented Generation

A self-contained Q&A system over your own PDF corpus.
Built end-to-end: **document parsing → chunking → embedding → hybrid retrieval → reranking → streaming LLM answers with Conversational Chat Memory and with citations**.

> Engineered as a portfolio piece to demonstrate AI-architecture decisions, not as a wrapper around a managed API. Every layer was picked with explicit trade-offs in mind.

---
## 🤗 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Puneet666/RAG_ADVANCE) · Running on free-tier CPU · 

## 🧩 Architecture

```
                   docs/*.pdf
                       │
        ┌──────────────▼──────────────┐
        │   STAGE 1: PARSE             │   Docling (smart: tables/OCR/scans)
        │   + fallback to pypdf        │   ↳ skip Docling if pypdf gets clean text
        └──────────────┬──────────────┘
                       ▼
        ┌──────────────▼──────────────┐
        │   STAGE 2: CHUNK             │   RecursiveCharacterTextSplitter
        │   512 tokens, 80 overlap     │   length = tiktoken (real token count)
        └──────────────┬──────────────┘
                       ▼
        ┌──────────────▼──────────────┐
        │   STAGE 3: EMBED             │   BAAI/bge-small-en-v1.5 (384-dim)
        │   ↳ Matryoshka truncate      │
        └──────────────┬──────────────┘
                       ▼
        ┌──────────────▼──────────────┐
        │   STAGE 4: INDEX             │   TurboVec (4-bit quantized) + SQLite FTS5
        │                              │   ↳ shared rowid links dense + BM25
        └──────────────┬──────────────┘
                       ▼  data/index.tvim  +  data/chunks.db
─────────────────────────────────────────────────────────  (offline ends here)
                       │
        ┌──────────────▼──────────────┐
        │   STAGE 5: QUERY             │
        │   • Dense search (TurboVec)  │
        │   • BM25 search (SQLite FTS) │
        │   • RRF fusion               │
        │   • Cross-encoder rerank     │   ms-marco-MiniLM-L-6-v2 (90 MB)
        └──────────────┬──────────────┘
                       ▼
        ┌──────────────▼──────────────┐
        │   STAGE 6: ANSWER            │   Groq (Llama-3.1-8b-instant)
        │   • Grounded prompt          │   ↳ streaming via SSE
        │   • Inline [N] citations     │   ↳ <think> tags stripped
        └──────────────┬──────────────┘
                       ▼
                   user browser
              (chat_ui.html — Claude-style, marked.js)
```

**Two-process production pattern:** FastAPI (port 7860) serves both the JSON API and the static HTML UI. No CORS, no separate frontend container.

---

## 🎯 Stack Decisions (The Architect's Reasoning)

Every component was a deliberate choice. Here's the why, not just the what:

### 1. PDF Parsing — Docling (with pypdf fallback)

| Considered | Why rejected / chosen |
|---|---|
| `pypdf` alone | Fails silently on scans, mangles tables — bad chunks → bad retrieval |
| `pdfplumber` | Better tables but no OCR, no layout reasoning |
| `LlamaParse` | Best quality but $0.003/page = ~$150 for 50k pages |
| `Marker` | Excellent, slightly less robust than Docling on edge cases |
| **`Docling` (IBM, open source)** ✅ | SOTA on layouts/tables, **free**, built-in OCR for scans |

**Hybrid path:** try `pypdf` first (fast); fall back to Docling only if avg chars/page < 100 (likely scan). Avoids 10-30× slowdown on clean PDFs.

### 2. Embedding Model — `bge-small-en-v1.5`

| Considered | Why rejected / chosen |
|---|---|
| `text-embedding-3-small` (OpenAI) | Recurring API cost; vendor lock-in |
| `mxbai-embed-large-v1` (670 MB) | Higher MTEB (64.7) but too heavy for free-tier CPU |
| `bge-large-en-v1.5` (1.3 GB) | Same quality issue at smaller scale |
| `nomic-embed-text-v1.5` (270 MB) | Strong, but bge-small is cheaper to host |
| **`bge-small-en-v1.5` (130 MB)** ✅ | MTEB 62.2 (matches OpenAI 3-small), 384-dim native, fits free-tier RAM |

**Trade-off:** ~2 points of MTEB for 5× smaller footprint and 10× faster CPU inference. For a free-tier deployment, this is the right curve.

### 3. Vector Store — TurboVec (4-bit quantized)

| Considered | Why rejected / chosen |
|---|---|
| Pinecone | Paid, vendor lock-in |
| Qdrant | Excellent but needs a separate container — won't fit single HF Space |
| FAISS | Fast but no quantization out of the box, no native hybrid |
| pgvector | Requires Postgres — overkill for a single-process app |
| **`TurboVec` (in-process, 4-bit)** ✅ | No server, no container, ~8× smaller than fp32, beats FAISS-PQ benchmarks |

**Result:** entire vector index for 300 chunks weighs **0.1 MB**. Scales linearly — 50k chunks ≈ 17 MB.

### 4. Hybrid Retrieval — Dense + BM25, Fused with RRF

Pure vector search misses exact terms ("EBITDA", "Section 4.2"). Pure BM25 misses paraphrases ("revenue decline" ≈ "earnings drop"). **Hybrid catches both.**

- **Dense branch:** `TurboVec.search(query_vec, k=50)`
- **Sparse branch:** `SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 50` — SQLite FTS5 uses Okapi BM25 natively.
- **Fusion:** Reciprocal Rank Fusion (`Σ 1/(60 + rank)`) — robust to score-scale mismatch, no tuning required.

### 5. Reranker — `ms-marco-MiniLM-L-6-v2` (90 MB)

| Considered | Why rejected / chosen |
|---|---|
| `bge-reranker-large` (560 MB) | Best quality, but 6× heavier and 10× slower |
| `bge-reranker-base` (280 MB) | Middle ground; still slow on CPU |
| **`MiniLM-L-6-v2` (90 MB)** ✅ | ~95% of `bge-reranker-large` quality at 1/6 the size |

Bi-encoder retrieval gives candidates fast; cross-encoder reranking surgically picks the best — same pattern Google Search has used for 20 years.

### 6. LLM — Groq Llama-3.1-8b-instant (streaming)

| Considered | Why rejected / chosen |
|---|---|
| OpenAI gpt-4o-mini | $0.15/1M tokens, ~2-3s latency |
| Anthropic Claude Haiku | ~$0.25/1M tokens, slightly slower |
| Self-hosted Llama via vLLM | Best long-term but needs GPU |
| OpenRouter qwen3-32b | Free but ~7s latency (reasoning model) |
| **Groq Llama-3.1-8b-instant** ✅ | ~500 tok/sec, free tier 14k req/day, OpenAI-compatible |

**14× faster than the previous Qwen3 setup. Same backend, one line of config.**

### 7. Frontend — Hand-written HTML, NOT Streamlit

Tried Streamlit first. Verdict: **~5 sec overhead per query** vs the same backend.

| | Streamlit | Hand-written HTML (current) |
|---|---|---|
| First token | ~7s | **0.4s** |
| Total response | ~7.4s | **1.6s** |
| Polish | Default look | Production-grade (Claude-style) |
| LOC | ~180 | ~320 |

Streamlit is great for internal data dashboards. For high-frequency token streaming, browser-native `fetch + ReadableStream` is unbeatable. **No production chat UI in industry uses Streamlit** (ChatGPT, Claude.ai, Perplexity all run Next.js).

### 8. Conversational Layer — Memory, Rewriting, Anti-Hallucination

A single-turn Q&A tool is a demo. A real product remembers, disambiguates pronouns, and admits what it doesn't know. Three additions promote this stack from "Q&A toy" to "conversation partner."

#### 8a. Conversation Memory With Auto-Summarization

**Pattern:** `ConversationSummaryBufferMemory` (industry term).

- Browser keeps the live conversation in `localStorage` (per-chat, free, zero server state).
- After every turn, if `messages.length > 8`, the oldest are POSTed to `/summarize`. The LLM compresses them into a 2-4 sentence summary; only the last 4 messages stay verbatim.
- Subsequent turns send `{summary, recent[], new_question}` — context stays bounded forever, no matter how long the chat runs.
- Same Groq Llama-3.1-8b handles both answers and summaries — one LLM, two roles, ~$0/mo on the free tier.

| Without summarization | With summarization |
|---|---|
| Turn 20 prompt = ~10,000 tokens | Turn 20 prompt = ~2,500 tokens (bounded) |
| Slow, expensive, eventually breaks context window | Constant cost, constant latency |

#### 8b. History-Aware Query Rewriting (Conditional)

**Pattern:** Contextual query reformulation / anaphora resolution.

```
User:  Who is Tsunesaburo Makiguchi?
Bot:   Founder of Soka Gakkai...

User:  When did he die?
       └─ NAIVE retrieval: searches for "when did he die" → finds nothing about Makiguchi
       └─ THIS PIPELINE: LLM rewrites → "When did Tsunesaburo Makiguchi die?" → correct chunks retrieved
```

- A cheap heuristic (`_needs_rewrite`) skips the rewrite for standalone questions — saves an LLM call ~50-70% of turns.
- When triggered, a tight system prompt prefers the **subject the USER most recently asked about**, not names mentioned only in the assistant's reply (fixes the "after him" ambiguity bug).
- The original question is what gets sent to the answer LLM — only retrieval uses the rewrite. Keeps the prompt natural.

#### 8c. Strict Anti-Hallucination Clause

The answer prompt now uses 5 numbered rules with a **literal refusal string**:

> "If the answer is not in the documents, reply with exactly: 'The documents don't cover that.' and stop."

Observable behavior — when the bot can't ground a claim, it refuses cleanly instead of leaking training-data knowledge with fake `Sources [3]` chips. This makes evaluation deterministic.

#### 8d. Multi-Conversation Sidebar (ChatGPT-Style)

- All conversations stored in `localStorage` as `{id, title, messages, summary, createdAt, lastActive}`.
- Sidebar shows past chats sorted by recency, click to switch, × to delete.
- Auto-titles from the first user question (first 40 chars).
- Per-chat memory + summarization runs independently — switching chats reloads that chat's full history.
- Mobile-responsive: `☰` hamburger toggles the sidebar.

**Why localStorage and not a database (for now):** zero server-side state means HF Spaces free tier can scale to N users without disk pressure. Privacy is a free side effect — the server never sees conversation content. Trade-off: no cross-device sync. That's the right call until there are paying users (then move to Postgres — see roadmap below).

---

## 📊 What's Inside (Numbers)

| | |
|---|---|
| Embedding dim | 384 (bge-small native) |
| Vectors / index size | 4-bit quantized via TurboVec |
| Reranker | MiniLM-L-6 (90 MB) |
| LLM | Llama-3.1-8b via Groq |
| Streaming protocol | Server-Sent Events |
| Backend | FastAPI + uvicorn (single process) |
| Frontend | Static HTML (no framework) |
| End-to-end latency | ~1.6s · first token ~0.4s |
| Free-tier RAM usage | ~3 GB |
| Free-tier disk usage | ~1.8 GB |

---

## 🚀 Live Examples

### Example 1: Factual Lookup
```
Q:  Who founded Soka Gakkai and when?

A:  Soka Gakkai was founded by **Tsunesaburo Makiguchi**
    and **Josei Toda** in **1930** in Japan, originally as
    the Soka Kyoiku Gakkai (Society for Value-Creating
    Education).

📖 Sources: 3 documents
⏱  0.42s · 1.61s total
```

### Example 2: Conceptual Synthesis (Multi-Chunk)
```
Q:  What is Soka Gakkai about?

A:  Soka Gakkai is a global Buddhist organization rooted
    in the teachings of Nichiren Daishonin, centered on
    promoting peace, culture, and education.

    Key principles:
    • Respect for the dignity of life
    • "Human revolution" — self-driven inner transformation
    • Grassroots dialogue across 192 countries
    • Engagement on nuclear disarmament, sustainability,
      and human rights

📖 Sources: 4 documents
⏱  0.38s · 2.04s total
```

The LLM is instructed to use **only** the retrieved context. If the question can't be answered from the corpus, it says so honestly rather than hallucinating.

---

## 🛠 Local Development

```bash
# 1. Clone & install
git clone <this-repo>
cd <repo>
pip install -r requirements-ingest.txt    # heavier: includes Docling

# 2. Drop PDFs in docs/
mkdir docs && cp ~/your-pdfs/*.pdf docs/

# 3. Build the index (one-time, ~minutes)
python rag.py ingest

# 4. Run the server
uvicorn rag_api:app --port 7860

# 5. Open http://localhost:7860
```

For production: `requirements.txt` (lean — no Docling, no chunking deps) is enough.

---

## 📦 What's In This Repo

```
├── rag.py / pipeline.py     Core retrieval pipeline (parse → chunk → embed → index)
├── rag_api.py               FastAPI server: /search, /answer, /answer/stream
├── chat_ui.html             Claude-style chat UI with streaming + sources toggle
├── data/
│   ├── index.tvim           Pre-built TurboVec vector index
│   └── chunks.db            Pre-built SQLite (text + BM25 + metadata)
├── Dockerfile               HF Spaces build instructions
├── requirements.txt         Server-only dependencies (~1.5 GB)
├── requirements-ingest.txt  Includes Docling for re-ingesting locally
└── README.md                You are here
```

---

## 🧠 The AI-Architect Mindset Behind This

This project isn't a tutorial-quality demo. Every choice traces back to a real production trade-off:

1. **Cost-Latency-Quality triangle:** picked Groq + bge-small + MiniLM to hit free-tier hosting *and* sub-2s answers.
2. **Offline vs online split:** heavy work (Docling, embedding 50k chunks) runs locally on a beefy machine; only the ~2 MB index ships to the server. **Server stays lean ~3 GB RAM.**
3. **Vector-space invariant:** same model and same dimension on both indexing and querying — enforced via constants, not hope.
4. **Hybrid by default:** dense + BM25 + rerank is the industry-standard trio. No production system uses pure vector search.
5. **Streaming is non-negotiable:** perceived latency drops 10× without changing the backend. This is the single most underused technique in RAG UX.
6. **Decoupled architecture:** swap Streamlit for HTML for React without touching `pipeline.py`. That's the real win.

---

## 🗺 Future Roadmap — The Production Evolution

This roadmap is the architectural growth path from "portfolio project" to "real product." Each upgrade is tagged with the **trigger** (when to add it) and the **trade-off** (what it costs).

### Phase 1 — Quality (Retrieval & Generation)

| # | Upgrade | Trigger | What it does |
|---|---|---|---|
| 1.1 | **Multi-query expansion** | Tricky multi-faceted questions get incomplete answers | LLM generates 3 phrasings of the query, retrieves with each, RRF-merges. Used by Perplexity. |
| 1.2 | **Contextual Retrieval** (Anthropic, 2024) | Want top-quartile retrieval quality | At ingest, prepend a 1-sentence "this chunk is about X in document Y" before embedding. Anthropic reports 49% retrieval-failure reduction. |
| 1.3 | **Adaptive retrieval skip** | Some questions don't need docs | Tiny classifier decides whether to retrieve at all. Generic questions answered from training; document questions go through RAG. Cuts cost + latency for ~30% of queries. |
| 1.4 | **HyDE (Hypothetical Document Embedding)** | Sparse / vague queries miss recall | LLM drafts a fake answer, embeds THAT, retrieves chunks similar to it. Counterintuitive but high impact. |

### Phase 2 — State & Memory (From Browser to Backend)

| # | Upgrade | Trigger | What it does |
|---|---|---|---|
| 2.1 | **PostgreSQL via Supabase** (free tier 500 MB) | Need cross-device sync, multi-user, audit log | Conversations + messages live in Postgres. Frontend stays mostly the same — just becomes a thin client over auth'd API. The standard production move. |
| 2.2 | **pgvector for memory + chunks** | Want one DB for everything | Postgres + pgvector extension stores both chunk embeddings AND conversation embeddings. One DB to back up, one schema to evolve. |
| 2.3 | **Redis cache ** | Hot conversations slow under load | Recent messages + summaries cached in Redis, sub-ms reads, drops Postgres load ~10×. Standard caching layer in front of any production DB. |
| 2.4 | **Conversation-level RAG** | User asks "what did we discuss about X?" | Past conversations get embedded into the same vector store as docs. Bot can search its own past chats — episodic memory. |
| 2.5 | **User profile memory** (ChatGPT-style) | Want personalization that survives sessions | Background job extracts "facts about user" from each chat (interests, expertise level, tone preference) and injects into future system prompts. |

### Phase 3 — The Big Architectural Bets (Optional)

| # | Upgrade | Trigger | What it does |
|---|---|---|---|
| 6.1 | **Self-hosted LLM (vLLM + GPU)** | Volume justifies it (~5M queries/month+) | Drop Groq dependency. Continuous batching, KV-cache reuse, speculative decoding. Most cost-effective at scale. |
| 6.2 | **GraphRAG (Microsoft, 2024)** | Complex multi-hop questions over relationship-heavy corpora | Extract entities + relationships into a knowledge graph; retrieve via graph traversal instead of vectors. |
| 6.3 | **Agentic mode (tools + planning)** | Need actions, not just answers | Add tool calls (database lookup, web search, calculator); LLM plans multi-step solutions. Cursor-style. |
| 6.4 | **Multimodal (images + PDFs as images)** | Diagrams, charts, slides matter | Swap to a vision-language model; embed page images alongside text. ColPali-style retrieval. |

---

## 📐 The Decision Framework Behind The Roadmap

Each phase above isn't chosen by ambition — it's triggered by a **measurable threshold**. This is how senior AI architects and product managers actually reason about evolution:

| Decision question | What it forces you to clarify |
|---|---|
| **Where does time actually go?** (measure, don't guess) | Surfaces the real bottleneck — usually the LLM, not the parts you assume |
| **Whose latency budget is this?** (mine vs. someone else's API) | You can optimize what you own; you can only swap what you don't |
| **Cost-Latency-Quality triangle: pick 2, sacrifice 1** | Forces explicit prioritization per use case |
| **Stateless vs. stateful: who owns the source of truth?** | Drives scalability ceiling — stateless backends scale infinitely |
| **Synchronous vs. asynchronous: what's on the user's wait path?** | Anything not on the critical path moves to a queue |
| **Per-tenant isolation: shared or sharded?** | Determines enterprise sales-readiness |
| **Failure modes: what happens when X is down?** | Forces fallback chains, retries, graceful degradation |

The roadmap above isn't a checklist — it's a **decision tree**. Each upgrade exists because there's a moment when the trade-off flips. Knowing **when** to add each is the architect's actual skill — the AI product manager's is knowing which to ship first.

---

## 📜 License

  MIT.

---

> Built as a deliberate exercise in AI-architecture decision-making — from embedding model selection to streaming UX. Every layer has a defensible reason.
