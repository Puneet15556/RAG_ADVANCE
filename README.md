# 📚 RAG Chat — Production-Grade Retrieval-Augmented Generation

A self-contained Q&A system over your own PDF corpus.
Built end-to-end: **document parsing → chunking → embedding → hybrid retrieval → reranking → streaming LLM answers with citations**.

> Engineered as a portfolio piece to demonstrate AI-architecture decisions, not as a wrapper around a managed API. Every layer was picked with explicit trade-offs in mind.

---

---

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
| **`bge-small-en-v1.5` (130 MB)** ✅ | MTEB 62.2 (matches OpenAI 3-small), 384-dim native, fits free-tier RAM |

**Trade-off:** ~2 points of MTEB for 5× smaller footprint and 10× faster CPU inference. For a free-tier deployment, this is the right curve.

### 3. Vector Store — TurboVec (4-bit quantized)

| Considered | Why rejected / chosen |
|---|---|
| Pinecone | Paid, vendor lock-in |
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
| **`MiniLM-L-6-v2` (90 MB)** ✅ | ~95% of `bge-reranker-large` quality at 1/6 the size |

Bi-encoder retrieval gives candidates fast; cross-encoder reranking surgically picks the best — same pattern Google Search has used for 20 years.

### 6. LLM — Groq Llama-3.1-8b-instant (streaming)

| Considered | Why rejected / chosen |
|---|---|
| OpenAI gpt-4o-mini | $0.15/1M tokens, ~2-3s latency |
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

Streamlit is great for internal data dashboards. For high-frequency token streaming, browser-native `fetch + ReadableStream` is unbeatable. CAN ALSO USED DIFFERENT FRAMEWORKS WITH OUR API LIKE Next.js.

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

For production: `requirements-server.txt` (lean — no Docling, no chunking deps) is enough.

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

## 📜 License
---

> Built as a deliberate exercise in AI-architecture decision-making — from embedding model selection to streaming UX. Every layer has a defensible reason.
