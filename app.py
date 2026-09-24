
from __future__ import annotations

import hashlib
import html
import os
import re
import time
from datetime import datetime

import chromadb
import numpy as np
import streamlit as st
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from rank_bm25 import BM25Okapi

try:
    import pymupdf
except ImportError:  # older PyMuPDF releases
    import fitz as pymupdf

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
load_dotenv()

DB_PATH = "./chroma_db"
DEFAULT_GEN_MODEL = "gemini-3.6-flash"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"

CHUNK_WORDS = 220        # words per chunk
CHUNK_OVERLAP = 40       # words shared between consecutive chunks
EMBED_BATCH = 64         # texts per embedding request
RRF_K = 60               # Reciprocal Rank Fusion constant
SHORT_QUESTION_WORDS = 8  # questions this short (or shorter) get expanded
RETRYABLE = {429, 500, 502, 503, 504}

DEPTH_OPTIONS = ["Simple first, then technical", "Straight to technical"]
FORMAT_OPTIONS = ["Summary", "Table", "Outline"]
FORMAT_HINTS = {
    "Summary": "Write a clear, well-organised prose summary.",
    "Table": "Present the answer as a markdown table where that helps, with a short intro line.",
    "Outline": "Present the answer as a structured bullet-point outline with nested points.",
}
DEPTH_HINTS = {
    "Simple first, then technical": "Start with a plain-language explanation, then add the technical detail.",
    "Straight to technical": "Be direct and technical; assume the reader knows the basics.",
}

st.set_page_config(page_title="Smart AI Research Assistant", page_icon="🔬", layout="wide")

CSS = """
<style>
.hero{padding:1.6rem 1.9rem;border-radius:18px;margin-bottom:1.1rem;color:#fff;
      background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
      box-shadow:0 10px 30px rgba(79,70,229,.25);}
.hero-title{font-size:2rem;font-weight:800;line-height:1.2;color:#fff;}
.hero-sub{margin-top:.4rem;font-size:1.02rem;opacity:.92;color:#fff;}
.src-head{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;}
.badge{display:inline-flex;align-items:center;justify-content:center;min-width:1.7rem;height:1.7rem;
       border-radius:999px;background:#4f46e5;color:#fff;font-weight:700;font-size:.85rem;}
.src-file{font-weight:600;}
.chip{padding:.1rem .6rem;border-radius:999px;font-size:.75rem;background:rgba(127,127,127,.16);}
.chip.sem{background:rgba(79,70,229,.2);}
.chip.kw{background:rgba(219,39,119,.2);}
.rel{margin-left:auto;width:120px;height:6px;border-radius:99px;background:rgba(127,127,127,.22);overflow:hidden;}
.rel>span{display:block;height:100%;background:linear-gradient(90deg,#4f46e5,#db2777);}
.excerpt{white-space:pre-wrap;font-size:.9rem;line-height:1.6;opacity:.92;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("history", [])           # list of result dicts
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("library_version", 0)    # bumped when the index changes
st.session_state.setdefault("flash", None)           # (level, message) shown after a rerun


# ----------------------------------------------------------------------------
# API plumbing: key, client, retries, friendly errors
# ----------------------------------------------------------------------------
def resolve_api_key() -> str:
    typed = (st.session_state.get("api_key_input") or "").strip()
    if typed:
        return typed
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    return make_client(resolve_api_key())


def with_retry(fn, tries: int = 4):
    """Call fn(), retrying rate-limit and server errors with exponential backoff."""
    for attempt in range(tries):
        try:
            return fn()
        except errors.APIError as exc:
            if attempt == tries - 1 or exc.code not in RETRYABLE:
                raise
            time.sleep(2 ** attempt)


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, errors.APIError):
        code = exc.code
        detail = getattr(exc, "message", None) or str(exc)
        if code == 404:
            return ("The selected Gemini model isn't available any more. Open **⚙️ Models** in the "
                    "sidebar and pick another one.")
        if code == 429:
            return "Gemini rate limit reached. Wait a few seconds and try again."
        if code in (401, 403) or "API key" in detail:
            return "Gemini rejected the API key. Check it in the sidebar (**🔑 API key**)."
        return f"Gemini API error {code}: {detail}"
    return f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False, ttl=3600)
def discover_models(api_key: str) -> tuple[list[str], list[str]]:
    """Ask the API which models this key can use, so retired names never bite us again."""
    client = genai.Client(api_key=api_key)
    skip = ("image", "tts", "audio", "live", "imagen", "veo", "robotics", "computer-use", "embed")
    gen, emb = [], []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        actions = set(getattr(model, "supported_actions", None) or [])
        if "embed" in name:
            if not actions or "embedContent" in actions:
                emb.append(name)
        elif "generateContent" in actions and not any(word in name for word in skip):
            gen.append(name)
    return sorted(set(gen), reverse=True), sorted(set(emb), reverse=True)


def pick_default(options: list[str], preferred: str, hint: str = "") -> str:
    if preferred in options:
        return preferred
    if hint:
        for name in options:
            if hint in name and "lite" not in name:
                return name
    return options[0]


# ----------------------------------------------------------------------------
# Gemini calls
# ----------------------------------------------------------------------------
def generate_text(prompt: str, temperature: float = 0.3) -> str:
    client = get_client()
    response = with_retry(lambda: client.models.generate_content(
        model=st.session_state.gen_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    ))
    return (response.text or "").strip()


def stream_text(prompt: str, temperature: float = 0.3):
    """Yield answer text as it arrives. Retries only if nothing has been sent yet."""
    client = get_client()
    attempt = 0
    while True:
        yielded = False
        try:
            stream = client.models.generate_content_stream(
                model=st.session_state.gen_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            for chunk in stream:
                text = chunk.text
                if text:
                    yielded = True
                    yield text
            return
        except errors.APIError as exc:
            attempt += 1
            if yielded or attempt >= 4 or exc.code not in RETRYABLE:
                raise
            time.sleep(2 ** attempt)


def _embed_batch(client: genai.Client, model: str, batch: list[str], task: str) -> list[list[float]]:
    try:
        response = with_retry(lambda: client.models.embed_content(
            model=model, contents=batch, config=types.EmbedContentConfig(task_type=task)))
    except errors.ClientError as exc:
        if exc.code != 400:
            raise
        # Some embedding models don't accept task_type - retry without it.
        response = with_retry(lambda: client.models.embed_content(model=model, contents=batch))
    return [item.values for item in response.embeddings]


def embed_texts(texts: list[str], task: str, model: str) -> np.ndarray:
    """Embed texts in batches and L2-normalise so distance ordering == cosine ordering."""
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend(_embed_batch(client, model, texts[start:start + EMBED_BATCH], task))
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ----------------------------------------------------------------------------
# Vector store (ChromaDB) and keyword index (BM25)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_chroma():
    return chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))


def collection_name(embed_model: str) -> str:
    # One collection per embedding model: vectors from different models are incompatible.
    slug = re.sub(r"[^A-Za-z0-9]+", "_", embed_model).strip("_")[:80] or "default"
    return f"docs_{slug}"


def get_collection(embed_model: str):
    return get_chroma().get_or_create_collection(collection_name(embed_model))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def extract_chunks(data: bytes) -> list[dict]:
    """Split a PDF into overlapping word chunks, keeping page numbers for citations."""
    chunks: list[dict] = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        for page_no, page in enumerate(doc, start=1):
            words = page.get_text().split()
            for start in range(0, len(words), step):
                piece = words[start:start + CHUNK_WORDS]
                if start > 0 and len(piece) <= CHUNK_OVERLAP:
                    break  # tail is already fully covered by the previous chunk
                if len(piece) >= 5:
                    chunks.append({"text": " ".join(piece), "page": page_no})
    finally:
        doc.close()
    return chunks


def ingest_file(name: str, data: bytes, embed_model: str, on_progress) -> str:
    """Index one PDF. Returns 'indexed', 'skipped' (already there) or 'empty' (no text)."""
    file_hash = hashlib.sha1(data).hexdigest()
    chunks = extract_chunks(data)
    if not chunks:
        return "empty"
    col = get_collection(embed_model)
    existing = len(col.get(where={"file_hash": file_hash}, include=[])["ids"])
    if existing >= len(chunks):
        return "skipped"
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors = embed_texts([c["text"] for c in batch], "RETRIEVAL_DOCUMENT", embed_model)
        col.upsert(
            ids=[f"{file_hash[:16]}-{start + i}" for i in range(len(batch))],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"file": name, "file_hash": file_hash, "page": c["page"], "chunk": start + i}
                       for i, c in enumerate(batch)],
        )
        on_progress(min(start + len(batch), len(chunks)) / len(chunks))
    return "indexed"


def library_summary(embed_model: str) -> dict:
    """{file_hash: {file, chunks, pages}} for everything indexed."""
    data = get_collection(embed_model).get(include=["metadatas"])
    docs: dict = {}
    for meta in data["metadatas"] or []:
        key = meta.get("file_hash") or meta.get("file", "unknown")
        entry = docs.setdefault(key, {"file": meta.get("file", "unknown"), "chunks": 0, "pages": set()})
        entry["chunks"] += 1
        entry["pages"].add(meta.get("page"))
    return docs


def get_bm25(embed_model: str) -> dict:
    """BM25 index over the whole collection, rebuilt only when the library changes."""
    col = get_collection(embed_model)
    stamp = (embed_model, col.count(), st.session_state.library_version)
    cache = st.session_state.get("bm25_cache")
    if cache and cache["stamp"] == stamp:
        return cache
    data = col.get(include=["documents", "metadatas"])
    docs = data["documents"] or []
    cache = {
        "stamp": stamp,
        "ids": data["ids"] or [],
        "docs": docs,
        "metas": data["metadatas"] or [],
        "bm25": BM25Okapi([tokenize(d) or ["_"] for d in docs]) if docs else None,
    }
    st.session_state.bm25_cache = cache
    return cache


# ----------------------------------------------------------------------------
# Retrieval: semantic + keyword, fused with Reciprocal Rank Fusion
# ----------------------------------------------------------------------------
def hybrid_search(query: str, top_k: int, files: list[str], embed_model: str):
    col = get_collection(embed_model)
    total = col.count()
    if total == 0:
        return [], "empty"

    pool = min(total, max(top_k * 4, 20))
    fused: dict[str, dict] = {}

    def add(cid: str, text: str, meta: dict, rank: int, via: str):
        entry = fused.setdefault(cid, {"id": cid, "text": text, "meta": meta, "score": 0.0, "via": set()})
        entry["score"] += 1.0 / (RRF_K + rank + 1)
        entry["via"].add(via)

    mode = "hybrid"
    try:  # semantic
        qvec = embed_texts([query], "RETRIEVAL_QUERY", embed_model)
        kwargs = {"query_embeddings": qvec.tolist(), "n_results": pool, "include": ["documents", "metadatas"]}
        if files:
            kwargs["where"] = {"file": {"$in": files}}
        res = col.query(**kwargs)
        for rank, (cid, text, meta) in enumerate(zip(res["ids"][0], res["documents"][0], res["metadatas"][0])):
            add(cid, text, meta, rank, "semantic")
    except Exception:  # embeddings unavailable -> keep going with keywords only
        mode = "keyword only"

    cache = get_bm25(embed_model)  # keyword
    if cache["bm25"] is not None:
        scores = cache["bm25"].get_scores(tokenize(query))
        rank = 0
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] <= 0:
                break
            meta = cache["metas"][idx]
            if files and meta.get("file") not in files:
                continue
            add(cache["ids"][idx], cache["docs"][idx], meta, rank, "keyword")
            rank += 1
            if rank >= pool:
                break

    hits = sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:top_k]
    if hits:
        best = hits[0]["score"]
        for hit in hits:
            hit["relevance"] = hit["score"] / best
    return hits, mode


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
def expand_query(raw: str) -> str:
    """Rewrite short/vague questions into one clear research question."""
    if len(raw.split()) > SHORT_QUESTION_WORDS:
        return raw
    recent = [h["question"] for h in st.session_state.history[-3:]]
    prompt = f"""You expand short or vague research questions into fuller, well-scoped questions,
using context about the user and the recent conversation.

User's field: {st.session_state.get('profile_field') or 'not specified'}
Recent questions: {recent}
Short question: "{raw}"

Rewrite it as ONE clear, specific research question. Return ONLY the rewritten question."""
    try:
        text = generate_text(prompt, temperature=0.2).strip().strip("\"'")
    except Exception:
        return raw
    return text.splitlines()[0].strip() if text else raw


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (file: {h['meta'].get('file')}, page {h['meta'].get('page')})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def answer_prompt(question: str, context: str, fmt: str) -> str:
    return f"""You are a careful research assistant. Answer the question using ONLY the numbered excerpts below.

Rules:
- Cite evidence inline with the excerpt number in square brackets, like [1] or [2][4].
- If the excerpts do not contain the answer, say so plainly. Never invent facts.
- Reader's field of study: {st.session_state.get('profile_field') or 'not specified'}.
- Explanation style: {DEPTH_HINTS[st.session_state.profile_depth]}
- Format: {FORMAT_HINTS[fmt]}

Excerpts:
{context}

Question: {question}"""


def directions_prompt(context: str, answer: str) -> str:
    return f"""Given these excerpts and the answer already given, write a short "what to explore next" note
in markdown with exactly three bold headings, each followed by 1-2 concise bullets:
**Open questions**, **Where sources disagree** (say "No clear disagreement found" if none),
**Related topics worth exploring**. Do not repeat the answer.

Excerpts:
{context}

Answer given:
{answer}"""


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_pipeline(question: str, fmt: str):
    """Expand -> retrieve -> stream answer -> directions. Returns a result dict or None."""
    started = time.time()
    embed_model = st.session_state.embed_model
    try:
        with st.status("Working on it…", expanded=True) as status:
            expanded = question
            if st.session_state.use_expansion:
                st.write("🧠 Interpreting your question")
                expanded = expand_query(question)
            st.write("🔎 Searching your library (semantic + keyword)")
            hits, mode = hybrid_search(expanded, st.session_state.top_k,
                                       st.session_state.get("file_filter", []), embed_model)
            if not hits:
                status.update(label="No matching passages found", state="error", expanded=False)
                st.warning("I couldn't find relevant passages. Try rephrasing, or clear the document filter.")
                return None
            status.update(label=f"Found {len(hits)} passages ({mode})", state="complete", expanded=False)

        context = build_context(hits)
        answer = st.write_stream(stream_text(answer_prompt(expanded, context, fmt)))
        if not isinstance(answer, str):
            answer = "".join(str(part) for part in answer)
        if not answer.strip():
            answer = "The model returned an empty response (it may have been blocked). Try rephrasing."

        with st.spinner("Looking for what to explore next…"):
            try:
                directions = generate_text(directions_prompt(context, answer))
            except Exception:
                directions = "_Couldn't generate suggestions this time._"
    except Exception as exc:
        st.error(friendly_error(exc))
        return None

    return {
        "id": str(time.time_ns()),
        "question": question,
        "expanded": expanded,
        "fmt": fmt,
        "mode": mode,
        "answer": answer,
        "directions": directions or "_No suggestions generated._",
        "seconds": time.time() - started,
        "time": datetime.now().strftime("%H:%M"),
        "sources": [
            {"n": i, "file": h["meta"].get("file", "unknown"), "page": h["meta"].get("page", "?"),
             "text": h["text"], "relevance": h.get("relevance", 0.0), "via": sorted(h["via"])}
            for i, h in enumerate(hits, start=1)
        ],
    }


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def to_markdown(res: dict) -> str:
    sources = "\n".join(f"{s['n']}. {s['file']}, p.{s['page']}" for s in res["sources"])
    return (f"# {res['expanded']}\n\n{res['answer']}\n\n## Worth exploring next\n\n{res['directions']}\n\n"
            f"## Sources\n\n{sources}\n")


def render_sources(sources: list[dict]):
    for s in sources:
        chips = "".join(f'<span class="chip {"sem" if v == "semantic" else "kw"}">{v}</span>' for v in s["via"])
        with st.container(border=True):
            st.markdown(
                f'<div class="src-head"><span class="badge">{s["n"]}</span>'
                f'<span class="src-file">{html.escape(str(s["file"]))}</span>'
                f'<span class="chip">p. {html.escape(str(s["page"]))}</span>{chips}'
                f'<span class="rel"><span style="width:{int(s["relevance"] * 100)}%"></span></span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Show excerpt"):
                st.markdown(f'<div class="excerpt">{html.escape(s["text"])}</div>', unsafe_allow_html=True)


def render_result(res: dict):
    if res["expanded"] != res["question"]:
        st.caption(f"🔍 Interpreted as: *{res['expanded']}*")
    tab_answer, tab_next, tab_sources = st.tabs(["✅ Answer", "🧭 Worth exploring next",
                                                 f"📎 Sources ({len(res['sources'])})"])
    with tab_answer:
        st.markdown(res["answer"])
        st.caption(f"⏱ {res['seconds']:.1f}s · {len(res['sources'])} passages · "
                   f"{res['mode']} search · {res['fmt']} format")
        st.download_button("⬇️ Download as Markdown", to_markdown(res),
                           file_name="research-answer.md", mime="text/markdown", key=f"dl_{res['id']}")
    with tab_next:
        st.markdown(res["directions"])
    with tab_sources:
        render_sources(res["sources"])


def render_history_entry(res: dict):
    st.markdown(res["answer"])
    st.markdown("**Worth exploring next**")
    st.markdown(res["directions"])
    st.markdown("**Sources**")
    st.markdown("\n".join(f"{s['n']}. {s['file']}, p.{s['page']}" for s in res["sources"]))
    st.download_button("⬇️ Download as Markdown", to_markdown(res),
                       file_name="research-answer.md", mime="text/markdown", key=f"hdl_{res['id']}")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
api_key = resolve_api_key()

with st.sidebar:
    st.markdown("### 🎓 Your profile")
    st.text_input("Your field of study", key="profile_field", placeholder="e.g. Machine learning")
    st.selectbox("Preferred explanation depth", DEPTH_OPTIONS, key="profile_depth")

    with st.expander("⚙️ Models"):
        gen_options, embed_options = [DEFAULT_GEN_MODEL], [DEFAULT_EMBED_MODEL]
        if api_key:
            try:
                found_gen, found_embed = discover_models(api_key)
                gen_options = found_gen or gen_options
                embed_options = found_embed or embed_options
            except Exception as exc:
                st.warning(f"Couldn't list models ({friendly_error(exc)}). Using defaults.")
        if st.session_state.get("gen_model") not in gen_options:
            st.session_state.gen_model = pick_default(gen_options, DEFAULT_GEN_MODEL, "flash")
        if st.session_state.get("embed_model") not in embed_options:
            st.session_state.embed_model = pick_default(embed_options, DEFAULT_EMBED_MODEL, "embedding")
        st.selectbox("Answer model", gen_options, key="gen_model")
        st.selectbox("Embedding model", embed_options, key="embed_model",
                     help="Each embedding model keeps its own library, so changing it means re-indexing.")

    st.markdown("### 🔎 Search")
    st.slider("Passages to retrieve", 3, 12, 6, key="top_k")
    st.toggle("Expand short questions", value=True, key="use_expansion")

    library = library_summary(st.session_state.embed_model)
    file_names = sorted({d["file"] for d in library.values()})
    st.session_state.file_filter = [f for f in st.session_state.get("file_filter", []) if f in file_names]
    st.multiselect("Limit to documents", file_names, key="file_filter",
                   placeholder="All documents", help="Leave empty to search everything.")

    with st.expander("🔑 API key", expanded=not api_key):
        st.text_input("Gemini API key", type="password", key="api_key_input",
                      help="Or set GEMINI_API_KEY in a .env file.")
    st.caption("Answers are generated only from your uploaded documents.")

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div class="hero-title">🔬 Smart AI Research Assistant</div>'
    '<div class="hero-sub">Upload papers, ask questions, and get answers with numbered citations '
    'plus suggestions for where to dig next.</div></div>',
    unsafe_allow_html=True,
)

flash = st.session_state.pop("flash", None)
if flash:
    {"success": st.success, "warning": st.warning, "error": st.error}.get(flash[0], st.info)(flash[1])

if not api_key:
    st.info("👈 Add your Gemini API key in the sidebar (or put `GEMINI_API_KEY=...` in a `.env` file) to begin.")
    st.stop()

total_chunks = sum(d["chunks"] for d in library.values())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", len(library))
m2.metric("Indexed passages", total_chunks)
m3.metric("Questions asked", len(st.session_state.history))
m4.metric("Model", st.session_state.gen_model)

tab_ask, tab_lib, tab_hist = st.tabs(["💬 Ask", "📚 Library", "🕘 History"])

# ---- Ask ---------------------------------------------------------------------
with tab_ask:
    with st.form("ask_form"):
        question = st.text_area("Ask a question about your documents", height=90,
                                placeholder="e.g. What methods did the authors use to evaluate the model?")
        col_fmt, col_btn = st.columns([4, 1])
        with col_fmt:
            fmt = st.radio("Answer format", FORMAT_OPTIONS, horizontal=True)
        with col_btn:
            submitted = st.form_submit_button("Ask ✨", type="primary")

    new_result = None
    if submitted:
        if not question.strip():
            st.warning("Type a question first.")
        elif total_chunks == 0:
            st.warning("Your library is empty. Upload PDFs in the **📚 Library** tab first.")
        else:
            new_result = run_pipeline(question.strip(), fmt)

    if new_result:
        st.session_state.history.append(new_result)
        st.session_state.last_result = new_result
        st.rerun()

    if st.session_state.last_result:
        st.divider()
        st.markdown(f"##### Q: {st.session_state.last_result['question']}")
        render_result(st.session_state.last_result)
    elif total_chunks == 0:
        st.info("📄 Start by uploading PDFs in the **📚 Library** tab.")

# ---- Library -----------------------------------------------------------------
with tab_lib:
    uploads = st.file_uploader("Upload research documents (PDF)", type=["pdf"],
                               accept_multiple_files=True, key="uploader")
    if uploads and st.button("Process documents", type="primary"):
        results = {"indexed": 0, "skipped": 0, "empty": [], "failed": []}
        progress = st.progress(0.0, text="Starting…")
        try:
            for n, file in enumerate(uploads):
                progress.progress(n / len(uploads), text=f"Indexing {file.name}…")
                try:
                    outcome = ingest_file(
                        file.name, file.getvalue(), st.session_state.embed_model,
                        lambda frac, n=n, name=file.name: progress.progress(
                            (n + frac) / len(uploads), text=f"Indexing {name}…"),
                    )
                except Exception as exc:
                    results["failed"].append(f"{file.name} ({friendly_error(exc)})")
                    continue
                if outcome == "empty":
                    results["empty"].append(file.name)
                else:
                    results[outcome] += 1
        finally:
            progress.empty()
        st.session_state.library_version += 1
        parts = [f"Indexed {results['indexed']} new file(s)"]
        if results["skipped"]:
            parts.append(f"{results['skipped']} already in library")
        if results["empty"]:
            parts.append("no extractable text in " + ", ".join(results["empty"]) + " (scanned PDF?)")
        level = "warning" if results["empty"] or results["failed"] else "success"
        if results["failed"]:
            parts.append("failed: " + "; ".join(results["failed"]))
        st.session_state.flash = (level, " · ".join(parts) + ".")
        st.rerun()

    st.markdown("#### Your library")
    if not library:
        st.caption("Nothing indexed yet.")
    for file_hash, info in sorted(library.items(), key=lambda kv: kv[1]["file"].lower()):
        with st.container(border=True):
            c_name, c_stats, c_del = st.columns([5, 3, 1])
            c_name.markdown(f"**📄 {info['file']}**")
            c_stats.caption(f"{info['chunks']} passages · {len(info['pages'])} pages")
            if c_del.button("🗑", key=f"del_{file_hash}", help="Remove this document"):
                get_collection(st.session_state.embed_model).delete(where={"file_hash": file_hash})
                st.session_state.library_version += 1
                st.session_state.flash = ("success", f"Removed {info['file']}.")
                st.rerun()

    if library:
        with st.expander("Danger zone"):
            confirm = st.checkbox("I want to delete the entire library for this embedding model")
            if st.button("Clear library", disabled=not confirm):
                get_chroma().delete_collection(collection_name(st.session_state.embed_model))
                st.session_state.library_version += 1
                st.session_state.flash = ("success", "Library cleared.")
                st.rerun()

# ---- History -----------------------------------------------------------------
with tab_hist:
    if not st.session_state.history:
        st.caption("Your questions will show up here.")
    else:
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
        for res in reversed(st.session_state.history):
            with st.expander(f"{res['time']} · {res['question']}"):
                render_history_entry(res)