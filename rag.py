"""
RAG pipeline using TurboVec (dense) + SQLite FTS5 (BM25) + cross-encoder rerank.
No servers, no containers — everything runs in one Python process against two files.

Install:
    pip install pypdf langchain tiktoken sentence-transformers \
                turbovec pandas pyarrow torch

Run:
    python pipeline.py ingest                    # parse -> chunk -> embed -> build index
    python pipeline.py search "your question"    # hybrid retrieve + rerank
"""

from __future__ import annotations

import sqlite3
import sys
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import tiktoken
import torch
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from turbovec import IdMapIndex
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ===================================================================
# CONFIG
# ===================================================================
PDF_DIR    = Path("docs")
DATA_DIR   = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
EMB_DIR    = DATA_DIR / "emb"
INDEX_FILE = DATA_DIR / "index.tvim"
DB_FILE    = DATA_DIR / "chunks.db"

CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 80
EMBEDDING_DIM  = 384         # Matryoshka truncation
QUANT_BITS     = 4          # TurboVec 4-bit quantization
GPU_BATCH      = 256
DOCS_PER_SHARD = 5_000
PARSE_WORKERS  = 1      # Docling uses ~3GB per worker - lower for safety
                            # Use 8+ only if you have 32+ GB RAM
DOCLING_TIMEOUT_SEC = 60    # Kill Docling on a single PDF after this many seconds

# DENSE_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5" 
RERANKER_NAME    = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 90 MB
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ===================================================================
# UTILITIES
# ===================================================================
def shard(items, size):
    """Yield lists of `size` items at a time."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=lambda text: len(tokenizer.encode(text)),
)


# ===================================================================
# STAGE 1 + 2: PARSE PDFs + CHUNK
# ===================================================================
def _parse_with_pypdf(pdf_path: Path) -> list[dict]:
    """Fast path. Returns [] if no usable text (e.g. scanned PDF)."""
    pages = []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return []

    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "doc_id": pdf_path.stem,
                "source": str(pdf_path),
                "page":   page_num,
                "text":   text,
            })
    return pages


def _parse_with_docling(pdf_path: Path) -> list[dict]:
    """Slow path: handles tables, multi-column, OCR for scanned pages."""
    from docling.document_converter import DocumentConverter
    from collections import defaultdict

    global _docling_converter
    if "_docling_converter" not in globals() or _docling_converter is None:
        _docling_converter = DocumentConverter()

    result = _docling_converter.convert(str(pdf_path))
    doc = result.document

    page_texts = defaultdict(list)
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if text and getattr(item, "prov", None):
            page_texts[item.prov[0].page_no].append(text)

    return [
        {
            "doc_id": pdf_path.stem,
            "source": str(pdf_path),
            "page":   page_no - 1,                   # 1-indexed -> 0-indexed
            "text":   "\n\n".join(texts).strip(),
        }
        for page_no, texts in sorted(page_texts.items())
        if "\n\n".join(texts).strip()
    ]


# Minimum characters per page to consider pypdf's output "good enough".
# Pages below this likely had only images/scans -> use Docling for OCR.
PYPDF_MIN_CHARS_PER_PAGE = 100

# Above this page count, skip Docling/OCR entirely (likely huge image PDFs that OOM).
MAX_PAGES_FOR_DOCLING = 30


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Hybrid parser: pypdf first, Docling only as a small-PDF fallback.
    - Clean text PDFs:        parsed in ~0.1s with pypdf
    - Tiny scanned PDFs:      parsed with Docling + OCR
    - Big/image-heavy PDFs:   pypdf only (avoid OOM)
    """
    # 1) Fast path
    pages = _parse_with_pypdf(pdf_path)

    # 2) Decide if pypdf got enough text
    if pages:
        avg_chars = sum(len(p["text"]) for p in pages) / len(pages)
        if avg_chars >= PYPDF_MIN_CHARS_PER_PAGE:
            return pages

    # 3) How many pages does this PDF have? Skip Docling if too many.
    try:
        n_pages = len(PdfReader(str(pdf_path)).pages)
    except Exception:
        n_pages = 0

    if n_pages > MAX_PAGES_FOR_DOCLING:
        print(f"[parse] {pdf_path.name}: {n_pages} pages > {MAX_PAGES_FOR_DOCLING}; "
              f"skipping Docling/OCR (kept {len(pages)} pypdf pages)")
        return pages

    # 4) Slow path: Docling + OCR (only for small PDFs)
    try:
        docling_pages = _parse_with_docling(pdf_path)
        if docling_pages:
            return docling_pages
        return pages
    except BaseException as e:
        print(f"[parse] Docling failed for {pdf_path.name} "
              f"({type(e).__name__}); keeping pypdf result ({len(pages)} pages)")
        return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    rows = []
    for page in pages:
        for chunk_idx, chunk_text in enumerate(text_splitter.split_text(page["text"])):
            rows.append({
                "chunk_id":  str(uuid.uuid4()),
                "doc_id":    page["doc_id"],
                "chunk_idx": chunk_idx,
                "text":      chunk_text,
                "page":      page["page"],
                "source":    page["source"],
            })
    return rows


def stage_chunk():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[chunk] found {len(pdf_paths)} PDFs")

    for shard_idx, pdf_batch in enumerate(shard(pdf_paths, DOCS_PER_SHARD)):
        output_file = CHUNKS_DIR / f"shard_{shard_idx:05d}.parquet"
        if output_file.exists():
            print(f"[chunk] skip {output_file.name}")
            continue

        # Submit each PDF as its own future so one OOM crash doesn't kill the batch.
        # If a worker dies, recreate the pool and retry the remaining PDFs.
        all_pages = []
        remaining = list(pdf_batch)
        while remaining:
            with ProcessPoolExecutor(max_workers=PARSE_WORKERS) as executor:
                futures = {executor.submit(parse_pdf, p): p for p in remaining}
                done_paths = set()
                try:
                    for fut in as_completed(futures):
                        pdf = futures[fut]
                        try:
                            all_pages.extend(fut.result())
                        except Exception as e:
                            print(f"[parse] FAILED {pdf.name}: {type(e).__name__}; skipping")
                        done_paths.add(pdf)
                except BrokenProcessPool:
                    print("[parse] worker pool died; restarting on remaining PDFs")
                remaining = [p for p in remaining if p not in done_paths]

        chunks = chunk_pages(all_pages)
        pd.DataFrame(chunks).to_parquet(output_file, compression="zstd", index=False)
        print(f"[chunk] wrote {output_file.name} -> {len(chunks):,} chunks")

# ===================================================================
# STAGE 3: EMBED (Matryoshka 1024 -> 256)
# ===================================================================
def stage_embed():
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[embed] loading {DENSE_MODEL_NAME} on {DEVICE}")
    model = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)
    if DEVICE == "cuda":
        model = model.half()
    model.max_seq_length = CHUNK_SIZE

    for chunk_file in sorted(CHUNKS_DIR.glob("shard_*.parquet")):
        output_file = EMB_DIR / chunk_file.name.replace("shard_", "emb_")
        if output_file.exists():
            print(f"[embed] skip {output_file.name}")
            continue

        df = pd.read_parquet(chunk_file, columns=["chunk_id", "text"])
        vectors = model.encode(
            df["text"].tolist(),
            batch_size=GPU_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        vectors = vectors[:, :EMBEDDING_DIM]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors.astype(np.float32)

        pd.DataFrame({
            "chunk_id":  df["chunk_id"],
            "embedding": list(vectors),
        }).to_parquet(output_file, compression="zstd", index=False)
        print(f"[embed] wrote {output_file.name} -> {len(df):,} vectors")


# ===================================================================
# STAGE 4: BUILD INDEX (TurboVec for dense + SQLite for BM25 + metadata)
# ===================================================================
def init_sqlite() -> sqlite3.Connection:
    """Create SQLite DB with FTS5 for BM25 and a metadata table."""
    conn = sqlite3.connect(DB_FILE)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            rowid     INTEGER PRIMARY KEY,
            chunk_id  TEXT UNIQUE,
            doc_id    TEXT,
            chunk_idx INTEGER,
            page      INTEGER,
            source    TEXT,
            text      TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, content='chunks', content_rowid='rowid', tokenize='porter'
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
    """)
    return conn


def stage_build_index():
    """Load all chunks + embeddings, build TurboVec index + populate SQLite."""
    DATA_DIR.mkdir(exist_ok=True)
    if INDEX_FILE.exists() and DB_FILE.exists():
        print(f"[index] {INDEX_FILE.name} and {DB_FILE.name} exist; delete to rebuild")
        return

    conn = init_sqlite()
    index = IdMapIndex(dim=EMBEDDING_DIM, bit_width=QUANT_BITS)

    next_rowid = 1
    for chunk_file in sorted(CHUNKS_DIR.glob("shard_*.parquet")):
        emb_file = EMB_DIR / chunk_file.name.replace("shard_", "emb_")
        if not emb_file.exists():
            continue

        chunks_df = pd.read_parquet(chunk_file)
        emb_df    = pd.read_parquet(emb_file)
        df        = chunks_df.merge(emb_df, on="chunk_id")

        # Assign sequential integer rowids (TurboVec needs uint64 ids)
        rowids = np.arange(next_rowid, next_rowid + len(df), dtype=np.uint64)
        next_rowid += len(df)

        # Insert into SQLite (metadata + FTS index)
        conn.executemany("""
            INSERT INTO chunks (rowid, chunk_id, doc_id, chunk_idx, page, source, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (int(rid), r.chunk_id, r.doc_id, int(r.chunk_idx),
             int(r.page), r.source, r.text)
            for rid, r in zip(rowids, df.itertuples(index=False))
        ])
        conn.executemany(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
            [(int(rid), r.text) for rid, r in zip(rowids, df.itertuples(index=False))],
        )
        conn.commit()

        # Insert into TurboVec
        vectors = np.vstack(df["embedding"].values).astype(np.float32)
        index.add_with_ids(vectors, rowids)
        print(f"[index] {chunk_file.name}: +{len(df):,} chunks (total {next_rowid - 1:,})")

    index.write(str(INDEX_FILE))
    conn.close()
    print(f"\n[index] wrote {INDEX_FILE.name} ({INDEX_FILE.stat().st_size / 1e6:.1f} MB)")
    print(f"[index] wrote {DB_FILE.name}    ({DB_FILE.stat().st_size / 1e6:.1f} MB)")


# ===================================================================
# STAGE 5: HYBRID SEARCH (TurboVec dense + SQLite BM25 + RRF + rerank)
# ===================================================================
_cache = {}

def load_search_components():
    if _cache:
        return _cache
    print(f"[search] loading models + index on {DEVICE}")
    _cache["index"]    = IdMapIndex.load(str(INDEX_FILE))
    _cache["db"]       = sqlite3.connect(DB_FILE, check_same_thread=False)
    _cache["dense"]    = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)
    _cache["reranker"] = CrossEncoder(RERANKER_NAME, device=DEVICE)
    return _cache


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """RRF: combine multiple ranked id lists into one fused ranking."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, rowid in enumerate(ranked):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank + 1)
    return [rid for rid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def search(query: str, top_k: int = 5, candidates: int = 50, verbose: bool = True):
    c = load_search_components()

    # ---------- Dense branch (TurboVec) ----------
    prompt = f"Represent this sentence for searching relevant passages: {query}"
    qv = c["dense"].encode(prompt, normalize_embeddings=True, convert_to_numpy=True)
    if qv.ndim == 2:
        qv = qv[0]     
    qv = qv[:EMBEDDING_DIM]
    qv = (qv / np.linalg.norm(qv)).astype(np.float32)
    print(f"qv type:  {type(qv)}")
    print(f"qv shape: {qv.shape if hasattr(qv, 'shape') else len(qv)}")
    print(f"qv dtype: {qv.dtype if hasattr(qv, 'dtype') else 'N/A'}")
    print(f"index dim expected: {EMBEDDING_DIM}")

    qv_batch = qv.reshape(1, -1)                              # shape (1, 384)
    _, dense_ids = c["index"].search(qv_batch, k=candidates)
    dense_ids = [int(i) for i in dense_ids[0]] 


    # ---------- Sparse branch (SQLite FTS5 BM25) ----------
    # FTS5 query: escape quotes, treat as natural-language match
    fts_query = " ".join(w for w in query.split() if w.isalnum())
    bm25_rows = c["db"].execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (fts_query, candidates),
    ).fetchall()
    bm25_ids = [r[0] for r in bm25_rows]

    # ---------- Fuse with RRF ----------
    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:20]
    if not fused_ids:
        print("no results")
        return []

    # ---------- Fetch payload from SQLite ----------
    placeholders = ",".join("?" * len(fused_ids))
    rows = c["db"].execute(
        f"SELECT rowid, text, source, page, doc_id FROM chunks WHERE rowid IN ({placeholders})",
        fused_ids,
    ).fetchall()
    by_id = {r[0]: r for r in rows}
    hits = [by_id[rid] for rid in fused_ids if rid in by_id]

    # ---------- Cross-encoder rerank ----------
    pairs  = [(query, h[1]) for h in hits]
    scores = c["reranker"].predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_k]

    # Pretty-print retrieved chunks with source + page (only in CLI search mode)
    if verbose:
        print(f"\nQ: {query}\n")
        for i, (hit, score) in enumerate(ranked, 1):
            _, text, source, page, doc_id = hit
            snippet = text[:280].strip()
            if len(text) > 280:
                snippet += "..."
            print(f"#{i}  score={float(score):.3f}  {source} (page {page})")
            print(f"    {snippet}\n")

    return ranked

# ===================================================================
# STAGE 6: ANSWER GENERATION (LLM with inline citations)
# ===================================================================
def answer(query: str, top_k: int = 5):
    """Retrieve + generate an answer with [source, page] citations."""
    ranked = search(query, top_k=top_k, verbose=False)
    if not ranked:
        print("no results")
        return

    # Build a context block with numbered citations
    context_parts = []
    citations = []
    for i, (hit, _) in enumerate(ranked, 1):
        _, text, source, page, doc_id = hit
        context_parts.append(f"[{i}] {text}")
        citations.append(f"[{i}] {Path(source).name}, page {page}")

    context = "\n\n".join(context_parts)
    sources = "\n".join(citations)

    prompt = (
        f"Answer the question using ONLY the context below. "
        f"Cite sources inline with [number] tags. If the context does not "
        f"contain the answer, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer_text = response.choices[0].message.content
    except Exception as e:
        # Fallback: just show context + cite (no LLM)
        print(f"[warn] LLM call failed ({e}); showing top chunks only")
        answer_text = "(no LLM configured - showing retrieved context above)"

    print(f"\nQ: {query}\n")
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer_text)
    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    print(sources)


# ===================================================================
# CLI
# ===================================================================
def ingest():
    stage_chunk()
    stage_embed()
    stage_build_index()
    print("\n✅ ingest complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "ingest":
        ingest()
    elif args[0] == "search":
        query = " ".join(args[1:]) or "test query"
        search(query)
    elif args[0] == "answer":
        query = " ".join(args[1:]) or "test query"
        answer(query)
    else:
        print('usage: python pipeline.py ingest')
        print('       python pipeline.py search  "your question"   # retrieved chunks only')
        print('       python pipeline.py answer  "your question"   # LLM answer + citations')
