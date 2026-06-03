"""
Streamlit dashboard for the RAG API.

Two-process architecture: this UI calls the FastAPI backend over HTTP.

Run alongside rag_api.py:
    # Terminal 1
    uvicorn rag_api:app --port 8000

    # Terminal 2
    streamlit run dashboard.py

Open http://localhost:8501 in your browser.
"""

from __future__ import annotations

import json
import time
from typing import Iterator

import requests
import streamlit as st


API_URL = "http://localhost:8000"


# ===================================================================
# PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
# SESSION STATE
# ===================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ===================================================================
# SIDEBAR
# ===================================================================
st.sidebar.title("⚙️  Settings")

# --- API health ---
api_slot = st.sidebar.empty()
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    api_slot.success(f"✓ API online\n\n`{health.get('model', 'unknown model')}`")
except Exception:
    api_slot.error(
        "✗ API offline\n\nStart the backend first:\n"
        "```\nuvicorn rag_api:app --port 8000\n```"
    )
    st.stop()

# --- Query options ---
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode",
    ["💬 Answer (streaming)", "🔍 Search (raw chunks)"],
    help="Answer = LLM-synthesized response.  Search = raw retrieved chunks.",
)
top_k = st.sidebar.slider("Top K results", 1, 10, 5)

# --- Recent queries ---
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Recent queries")
if st.session_state.history:
    for past in reversed(st.session_state.history[-10:]):
        if st.sidebar.button(
            past[:50] + ("…" if len(past) > 50 else ""),
            key=f"hist_{hash(past)}",
            use_container_width=True,
        ):
            st.session_state.last_query = past
            st.rerun()
else:
    st.sidebar.caption("No queries yet.")

if st.sidebar.button("Clear history", use_container_width=True):
    st.session_state.history.clear()
    st.rerun()


# ===================================================================
# MAIN AREA
# ===================================================================
st.title("📚  RAG Chat")
st.caption("Hybrid retrieval (dense + BM25) → reranked → LLM answer with sources.")

query = st.text_input(
    "Your question",
    value=st.session_state.last_query,
    placeholder="e.g. Who founded Soka Gakkai and when?",
    key="query_input",
)

ask_clicked = st.button("Ask", type="primary")


# ===================================================================
# HELPERS
# ===================================================================
# Replace requests.iter_lines() with iter_content + manual parse
def parse_sse(response) -> Iterator[dict]:
    buffer = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        buffer += chunk
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.split("\n"):
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue


def render_answer_mode(query: str, top_k: int) -> None:
    """Streaming answer: tokens render live, sources appear at the end."""
    answer_slot = st.empty()
    meta_slot   = st.empty()
    sources_slot = st.empty()

    meta_slot.caption("⏳  Searching documents...")
    t0 = time.time()

    try:
        response = requests.get(
            f"{API_URL}/answer/stream",
            params={"q": query, "top_k": top_k},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return

    # AFTER (fast):
    sources_holder = {"sources": [], "first_token_at": None}

    def token_generator():
        """Yield tokens for st.write_stream; capture sources via closure."""
        for msg in parse_sse(response):
            if msg.get("type") == "token":
                if sources_holder["first_token_at"] is None:
                    sources_holder["first_token_at"] = time.time() - t0
                yield msg["text"]
            elif msg.get("type") == "sources":
                sources_holder["sources"] = msg.get("sources", [])

    # st.write_stream is Streamlit's optimized streaming renderer
    answer_slot.write_stream(token_generator())

    sources = sources_holder["sources"]
    first_token_at = sources_holder["first_token_at"]

    total = time.time() - t0
    if first_token_at is not None:
        meta_slot.caption(
            f"✨  First token: **{first_token_at:.2f}s**  ·  "
            f"Total: **{total:.2f}s**  ·  "
            f"Sources: **{len(sources)}**"
        )
    else:
        meta_slot.caption(f"Done in {total:.2f}s")

    if sources:
        with sources_slot.expander("📖  Show sources", expanded=False):
            for s in sources:
                st.markdown(f"- **{s}**")


def render_search_mode(query: str, top_k: int) -> None:
    """Raw retrieval: show ranked chunks with metadata."""
    meta_slot = st.empty()
    meta_slot.caption("⏳  Searching...")
    t0 = time.time()

    try:
        response = requests.get(
            f"{API_URL}/search",
            params={"q": query, "top_k": top_k},
            timeout=30,
        )
        response.raise_for_status()
        hits = response.json()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return

    meta_slot.caption(
        f"📊  {len(hits)} results in **{time.time() - t0:.2f}s**"
    )

    if not hits:
        st.warning("No results found.")
        return

    for i, hit in enumerate(hits, 1):
        with st.expander(
            f"**#{i}**  ·  score=**{hit['score']:.2f}**  ·  "
            f"{hit['source']} (page {hit['page']})",
            expanded=(i == 1),
        ):
            st.write(hit["text"])


# ===================================================================
# DISPATCH
# ===================================================================
if ask_clicked and query.strip():
    # Track in history (deduplicated)
    if not st.session_state.history or st.session_state.history[-1] != query:
        st.session_state.history.append(query)
    st.session_state.last_query = query

    if mode.startswith("💬"):
        render_answer_mode(query, top_k)
    else:
        render_search_mode(query, top_k)
elif ask_clicked and not query.strip():
    st.warning("Please enter a question first.")
