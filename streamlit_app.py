"""
streamlit_app.py — interview demo dashboard.

Not a replacement for the Gradio app (app.py) or the FastAPI surface
(app/main.py) — this is a small, focused view built to demonstrate the
pipeline's differentiators: a confidence score broken into its parts,
clickable citations, and a side-by-side hybrid-vs-dense-only retrieval
comparison (Project 6 spec, Phase 5.2). Same palette as app.py (Gradio),
so the two look like one product.

Two tabs: "PDF Upload" (upload/select a document, ask, compare retrieval)
and "Golden Set" (the 50-question eval results). Run with:

    streamlit run streamlit_app.py
"""

import html
import os
import tempfile
from datetime import datetime

import streamlit as st

from app.config import Config
from app.document_loader import _SUPPORTED_EXTENSIONS
from app.rag_pipeline import build_vector_store, load_vector_store, make_qa_chain, dense_only_retrieve
from app.citations import verify_citations
from app.evaluation import evaluate_rag_response
from app.session_store import (
    save_history,
    load_history,
    delete_history,
    save_feedback,
    get_feedback_stats,
)
from app.report_formatting import (
    format_retrieval_eval,
    format_ragas_eval,
    format_citation_verification,
    format_answer_html,
    format_bar_html,
    format_source_detail_html,
    format_comparison_html,
)
from app.golden_eval_summary import load_golden_eval_summary as _load_golden_eval_summary

STORE_PATH = "artifacts/vector_store"
DOC_ID = "Attention.pdf"
GOLDEN_RESULTS_PATH = "artifacts/eval/golden_eval_results.jsonl"

# Outline icons (Feather-style: stroke-based, no fill) -- used next to section
# labels instead of emoji.
_ICON_ATTRS = 'width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
_ICONS = {
    "document": f'<svg {_ICON_ATTRS}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "message": f'<svg {_ICON_ATTRS}><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>',
    "check": f'<svg {_ICON_ATTRS}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "activity": f'<svg {_ICON_ATTRS}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "bookmark": f'<svg {_ICON_ATTRS}><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
    "layers": f'<svg {_ICON_ATTRS}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "target": f'<svg {_ICON_ATTRS}><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "grid": f'<svg {_ICON_ATTRS}><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "thumbsup": f'<svg {_ICON_ATTRS}><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>',
}

st.set_page_config(page_title="RAG Pipeline Dashboard", layout="wide")

# Palette matches app.py (Gradio): #FFD3AC (light peach) / #FFB5AB (rose) /
# #E39A7B (terracotta) / #DBB06B (gold), cream page background.
_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

_CSS = """
<style>
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; }

.stApp {
    background:
        radial-gradient(640px circle at 12% 8%, rgba(219,176,107,0.22), transparent 60%),
        radial-gradient(720px circle at 88% 6%, rgba(227,154,123,0.20), transparent 60%),
        radial-gradient(680px circle at 78% 92%, rgba(255,181,171,0.20), transparent 60%),
        radial-gradient(600px circle at 8% 88%, rgba(219,176,107,0.16), transparent 60%),
        linear-gradient(160deg, #FFF8F2 0%, #FFF0E6 50%, #FFE8DA 100%) !important;
    background-attachment: fixed !important;
}
.block-container { max-width: 1200px; padding-top: 1.5rem; }

.app-header {
    background: linear-gradient(135deg, #DBB06B 0%, #E39A7B 55%, #FFB5AB 100%);
    border-radius: 18px;
    padding: 30px 40px 26px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(219,176,107,0.28);
    text-align: center;
}
.app-header h1 {
    font-family: 'Fraunces', Georgia, serif;
    color: #FFFFFF;
    font-size: 2.4em;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    text-shadow: 0 2px 8px rgba(100,40,10,0.18);
}
.app-header p {
    color: rgba(255,255,255,0.88);
    font-size: 0.92em;
    margin: 0;
    letter-spacing: 0.01em;
}

.section-label {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: linear-gradient(135deg, #E39A7B 0%, #DBB06B 100%);
    color: #FFFFFF;
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-radius: 8px;
    padding: 6px 14px;
    margin: 4px 0 12px 0;
    box-shadow: 0 2px 8px rgba(227,154,123,0.22);
}
.section-label svg { flex-shrink: 0; }

/* Field-level labels (document/question/history/eval boxes) -- same red
   Gradio's default component labels render in (#EA580C, Tailwind orange-600),
   left unstyled there since it was never overridden -- kept deliberate here
   so the two surfaces read as the same product at this smaller label tier
   too, distinct from the bigger gradient .section-label pills above. */
.field-label {
    display: inline-block;
    background: #EA580C;
    color: #FFFFFF;
    font-size: 0.68em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-radius: 6px;
    padding: 4px 10px;
    margin: 0 0 8px 0;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 14px !important;
    border: 1px solid #FFD3AC !important;
    box-shadow: 0 2px 14px rgba(219,176,107,0.10);
}

/* Buttons */
.stButton > button, [data-testid^="stBaseButton"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(130deg, #DBB06B 0%, #E39A7B 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 3px 12px rgba(219,176,107,0.32) !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(130deg, #E39A7B 0%, #DBB06B 100%) !important;
    box-shadow: 0 6px 20px rgba(227,154,123,0.42) !important;
}
.stButton > button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: #FFFFFF !important;
    color: #C9713F !important;
    border: 1.5px solid #FFB5AB !important;
}
.stButton > button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
    background: #FFF5EE !important;
    border-color: #E39A7B !important;
    color: #A5502A !important;
}

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #FFD3AC !important;
    border-radius: 10px !important;
    color: #3D1A06 !important;
}

/* Progress bars (confidence) */
.stProgress > div > div { background: #FFE3CC !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, #DBB06B 0%, #E39A7B 100%) !important; }

/* Metrics */
[data-testid="stMetricValue"] { color: #7A4020 !important; }
[data-testid="stMetricLabel"] { color: #A06030 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #FFD3AC; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    color: #A06030;
    font-weight: 600;
    font-size: 0.94em;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] { color: #E39A7B !important; border-bottom-color: #E39A7B !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #FFF5EE !important;
    border: 2px dashed #FFB5AB !important;
    border-radius: 12px !important;
}

/* Body text */
p, li, .stMarkdown, label { color: #3D1A06; }
.stCaption, [data-testid="stCaptionContainer"] { color: #A06030 !important; }
hr { border-color: #FFD3AC !important; }

/* Answer card */
.answer-card {
    background: #FFFEF8;
    border: 1px solid #FFD3AC;
    border-left: 4px solid #DBB06B;
    border-radius: 10px;
    padding: 22px 26px;
    font-size: 1.06em;
    line-height: 1.85;
    color: #3D1A06;
}
.answer-card p { margin: 0 0 14px 0; font-size: 1em; line-height: inherit; }
.answer-card p:last-child { margin-bottom: 0; }
.answer-card .cite {
    color: #C9713F;
    font-weight: 700;
    font-size: 1.15em;
    padding: 0 1px;
}
.answer-card .analogy {
    display: block;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px dashed #FFD3AC;
    color: #7A4020;
    font-style: italic;
}

/* Snippet text in Sources / comparison */
.snippet {
    font-size: 0.88em;
    line-height: 1.7;
    color: #5A3010;
}
.meta-line {
    font-size: 0.78em;
    color: #A06030;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 4px;
}
</style>
"""


@st.cache_resource
def load_default_pipeline():
    try:
        vs, chunks = load_vector_store(STORE_PATH, doc_id=DOC_ID)
    except Exception:
        if Config.QDRANT_URL:
            # Qdrant configured but unreachable -- fall back to local FAISS
            # rather than let a demo fail on a flaky cloud dependency.
            Config.QDRANT_URL = ""
            vs, chunks = load_vector_store(STORE_PATH, doc_id=DOC_ID)
        else:
            raise
    chain = make_qa_chain(vs, doc_id=DOC_ID, all_chunks=chunks)
    return vs, chain


def ingest_uploaded_file(uploaded_file) -> str:
    """Builds a vector store + QA chain for a newly uploaded file and adds it
    to st.session_state.loaded_docs. Returns the doc name. Mirrors app.py's
    load_document() upload flow (same build_vector_store/make_qa_chain calls)."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {supported}")

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    store_path = os.path.join(tmp_dir, "vector_store")
    vs, chunks = build_vector_store(file_path, store_path, doc_id=name)
    chain = make_qa_chain(vs, doc_id=name, all_chunks=chunks)

    st.session_state.loaded_docs[name] = {"vs": vs, "chain": chain}
    return name


@st.cache_data
def load_golden_eval_summary():
    return _load_golden_eval_summary(GOLDEN_RESULTS_PATH)


def _section_label(text: str, icon: str = None):
    icon_html = _ICONS.get(icon, "") if icon else ""
    st.markdown(f'<div class="section-label">{icon_html}{text}</div>', unsafe_allow_html=True)


def _field_label(text: str):
    st.markdown(f'<div class="field-label">{text}</div>', unsafe_allow_html=True)


def _bar_row(label: str, correct: int, total: int):
    pct = (correct / total) if total else 0.0
    st.markdown(format_bar_html(label, f"{correct}/{total} · {pct:.0%}", pct), unsafe_allow_html=True)


def render_sources_panel(sources: list, selected_key: str):
    cols = st.columns(len(sources)) if len(sources) <= 6 else [st] * len(sources)
    for col, s in zip(cols, sources):
        chunk_num = s["chunk"]
        is_selected = st.session_state.get(selected_key) == chunk_num
        with col:
            if st.button(
                f"Source {chunk_num}",
                key=f"{selected_key}_btn_{chunk_num}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state[selected_key] = None if is_selected else chunk_num
                # Buttons render top-to-bottom as st.button() is called, so
                # without an immediate rerun, buttons ABOVE this one in the row
                # keep their stale highlight for this pass -- only a fresh
                # rerun redraws the whole row consistently from the new state.
                st.rerun()

    active = st.session_state.get(selected_key)
    if active is not None:
        match = next((s for s in sources if s["chunk"] == active), None)
        if match:
            with st.container(border=True):
                st.markdown(format_source_detail_html(match), unsafe_allow_html=True)


def render_comparison(hybrid_sources: list, dense_docs: list):
    hybrid_html, dense_html = format_comparison_html(hybrid_sources, dense_docs)
    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.markdown("**Hybrid** (BM25 + dense, RRF-fused)")
            st.markdown(hybrid_html, unsafe_allow_html=True)

    with col_b:
        with st.container(border=True):
            st.markdown("**Dense-only** (no BM25)")
            st.markdown(dense_html, unsafe_allow_html=True)


def render_ask_tab():
    if "loaded_docs" not in st.session_state:
        st.session_state.loaded_docs = {}

    _section_label("STEP 1 — DOCUMENT", "document")
    upload_col, select_col = st.columns([2, 1])
    with upload_col:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        uploaded_file = st.file_uploader(f"Upload a document ({supported})", label_visibility="collapsed")
        if uploaded_file is not None and uploaded_file.name not in st.session_state.loaded_docs:
            with st.spinner(f"Indexing '{uploaded_file.name}' (calls the embedding API — real cost)..."):
                try:
                    name = ingest_uploaded_file(uploaded_file)
                    st.session_state.active_doc = name
                    # Selectbox below reads this key directly (see STEP 1 select
                    # widget) -- setting session_state.active_doc alone isn't
                    # enough, since a selectbox's own keyed state, once it
                    # exists, takes priority over a freshly-computed `index=`.
                    st.session_state["active_doc_select"] = name
                except Exception as e:
                    st.error(f"Failed to index '{uploaded_file.name}': {e}")

    try:
        default_vs, default_chain = load_default_pipeline()
        st.session_state.loaded_docs.setdefault(DOC_ID, {"vs": default_vs, "chain": default_chain})
    except Exception as e:
        st.error(
            f"Could not load the default vector store at '{STORE_PATH}': {e}\n\n"
            "Run 'python run_me_once.py' to build it first. You can still upload your own document above."
        )

    if not st.session_state.loaded_docs:
        return

    if st.session_state.get("active_doc") not in st.session_state.loaded_docs:
        st.session_state.active_doc = next(iter(st.session_state.loaded_docs))
        st.session_state["active_doc_select"] = st.session_state.active_doc

    doc_names = list(st.session_state.loaded_docs.keys())
    if st.session_state.get("active_doc_select") not in doc_names:
        st.session_state["active_doc_select"] = st.session_state.active_doc

    with select_col:
        # key= (not index=) is what makes this reliably reflect a just-ingested
        # doc -- see the ingest block above, which sets this same key directly.
        active_doc = st.selectbox("Active document", doc_names, key="active_doc_select")
    st.session_state.active_doc = active_doc
    vs = st.session_state.loaded_docs[active_doc]["vs"]
    chain = st.session_state.loaded_docs[active_doc]["chain"]

    if "history_by_doc" not in st.session_state:
        st.session_state.history_by_doc = {}
    if active_doc not in st.session_state.history_by_doc:
        # Restored once per doc per session -- Redis-backed, shared with the
        # Gradio app's own history for the same doc name (app/session_store.py).
        st.session_state.history_by_doc[active_doc] = load_history(active_doc)

    st.write("")
    _section_label("STEP 2 — ASK", "message")

    _field_label("CONVERSATION HISTORY")
    history_text = st.session_state.history_by_doc.get(active_doc, "")
    with st.container(border=True):
        if history_text:
            st.markdown(
                f'<div class="snippet" style="white-space:pre-wrap;">{html.escape(history_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No conversation yet.")
    if st.button("Clear Conversation", key="clear_history_btn"):
        st.session_state.history_by_doc[active_doc] = ""
        delete_history(active_doc)
        st.rerun()

    st.write("")
    question = st.text_input("Question", label_visibility="collapsed", placeholder="Ask a question about the active document...")
    st.caption("Calls the configured LLM once per question (real API cost, same as normal use).")
    ask = st.button("Ask", type="primary", icon=":material/send:")

    if ask and question.strip():
        with st.spinner("Retrieving and generating..."):
            result = chain.invoke({"query": question})
            dense_docs, dense_grade = dense_only_retrieve(vs, question)
        answer_text = result.get("result", "")
        sources_for_eval = result.get("sources") or []

        timestamp = datetime.now().strftime("%H:%M:%S")
        separator = ". " * 30
        new_entry = f"[{timestamp}]  You: {question}\n\nAssistant: {answer_text}\n\n{separator}\n\n"
        updated_history = st.session_state.history_by_doc.get(active_doc, "") + new_entry
        st.session_state.history_by_doc[active_doc] = updated_history
        save_history(active_doc, updated_history)

        st.session_state["last_result"] = result
        st.session_state["last_dense_docs"] = dense_docs
        st.session_state["last_dense_grade"] = dense_grade
        st.session_state["selected_source"] = None
        st.session_state["last_answered_doc"] = active_doc

        st.session_state["last_eval_question"] = question
        st.session_state["last_eval_answer"] = answer_text
        st.session_state["last_eval_contexts"] = [s.get("content", "") for s in sources_for_eval]
        st.session_state["last_eval_sources"] = sources_for_eval
        st.session_state["feedback_status"] = ""
        st.session_state["phase1_report"] = ""
        st.session_state["phase2_report"] = ""
        st.session_state["phase4_report"] = ""
        # The CONVERSATION HISTORY box above was already drawn this pass with
        # the pre-answer text -- only a rerun redraws it with this entry included
        # (same reasoning as the Source-button highlight fix elsewhere in this file).
        st.rerun()

    if st.session_state.get("last_answered_doc") != active_doc:
        # Switched documents since the last answer -- don't show a stale answer
        # against a source-chunk set that no longer matches the active document.
        return

    result = st.session_state.get("last_result")
    if result is None:
        return

    st.write("")
    _section_label("ANSWER", "check")
    st.markdown(
        f'<div class="answer-card">{format_answer_html(result.get("result", ""))}</div>',
        unsafe_allow_html=True,
    )

    confidence = result.get("confidence") or {}
    if confidence:
        st.write("")
        _section_label("CONFIDENCE", "activity")
        with st.container(border=True):
            st.metric("Composite", f"{confidence.get('composite', 0):.2f}")
            c1, c2, c3 = st.columns(3)
            for col, label, key in [
                (c1, "Retrieval", "retrieval"),
                (c2, "Citation coverage", "citation_coverage"),
                (c3, "Completeness", "completeness"),
            ]:
                with col:
                    st.caption(label)
                    st.progress(min(max(confidence.get(key, 0), 0.0), 1.0))
                    st.write(f"{confidence.get(key, 0):.2f}")

    sources = result.get("sources") or []
    if sources:
        st.write("")
        _section_label("SOURCES", "bookmark")
        st.caption("Click a source to view the chunk it was cited from.")
        render_sources_panel(sources, "selected_source")

    dense_docs = st.session_state.get("last_dense_docs")
    if dense_docs is not None and sources:
        st.write("")
        _section_label("RETRIEVAL COMPARISON — HYBRID VS. DENSE-ONLY", "layers")
        render_comparison(sources, dense_docs)

    st.write("")
    _section_label("RATE THE LAST ANSWER", "thumbsup")
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
    with fb_col1:
        if st.button("Thumbs Up", key="thumbs_up_btn", use_container_width=True):
            save_feedback(active_doc, st.session_state.get("last_eval_question", ""), st.session_state.get("last_eval_answer", ""), "up")
            fb = get_feedback_stats(active_doc)
            st.session_state["feedback_status"] = f"Recorded: thumbs up  ({fb['up']} up / {fb['down']} down total)"
    with fb_col2:
        if st.button("Thumbs Down", key="thumbs_down_btn", use_container_width=True):
            save_feedback(active_doc, st.session_state.get("last_eval_question", ""), st.session_state.get("last_eval_answer", ""), "down")
            fb = get_feedback_stats(active_doc)
            st.session_state["feedback_status"] = f"Recorded: thumbs down  ({fb['up']} up / {fb['down']} down total)"
    with fb_col3:
        _field_label("FEEDBACK STATUS")
        st.write(st.session_state.get("feedback_status") or " ")

    st.write("")
    _section_label("STEP 3 — EVALUATION", "grid")
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(
        ["Phase 1 — Retrieval Eval", "Phase 2 — RAGAS Eval", "Phase 4 — Citation Verification"]
    )

    with eval_tab1:
        st.markdown(
            "Generates synthetic questions from document chunks — measures "
            "**Precision@K · Recall@K · MRR · Coverage**.  \n"
            "Takes ~20 seconds. No extra API keys needed."
        )
        if st.button("Run Retrieval Evaluation", key="phase1_btn"):
            if not hasattr(chain, "run_retrieval_eval"):
                st.session_state["phase1_report"] = "Retrieval eval not available on this chain."
            else:
                with st.spinner("Generating synthetic questions and scoring retrieval..."):
                    eval_result = chain.run_retrieval_eval(n_questions=8, k=Config.RETRIEVAL_K)
                st.session_state["phase1_report"] = format_retrieval_eval(eval_result, active_doc)
        if st.session_state.get("phase1_report"):
            _field_label("RETRIEVAL METRICS")
            st.code(st.session_state["phase1_report"], language=None)

    with eval_tab2:
        st.markdown(
            "Scores the last answer on **Faithfulness · Answer Relevancy · Context Precision** "
            "via LLM-as-judge.  \nRequires `RAGAS_EVAL=true` in `.env`. Uses your configured Groq "
            "LLM — takes 10-20 seconds."
        )
        if st.button("Evaluate Last Answer (RAGAS)", key="phase2_btn"):
            q = st.session_state.get("last_eval_question", "")
            if not q:
                st.session_state["phase2_report"] = "Ask a question first, then click Evaluate."
            else:
                a = st.session_state.get("last_eval_answer", "")
                ctxs = st.session_state.get("last_eval_contexts", [])
                with st.spinner("Running RAGAS evaluation..."):
                    scores = evaluate_rag_response(q, a, ctxs)
                st.session_state["phase2_report"] = format_ragas_eval(scores, q)
        if st.session_state.get("phase2_report"):
            _field_label("RAGAS SCORES")
            st.code(st.session_state["phase2_report"], language=None)

    with eval_tab3:
        st.markdown(
            "Checks whether each cited superscript (¹²³) in the last answer is actually "
            "supported by the source chunk it points to, and flags claims with no citation at all.  \n"
            "Uses your configured LLM — one extra call per cited sentence, takes a few seconds."
        )
        if st.button("Verify Citations (Last Answer)", key="phase4_btn"):
            q = st.session_state.get("last_eval_question", "")
            srcs = st.session_state.get("last_eval_sources", [])
            if not q:
                st.session_state["phase4_report"] = "Ask a question first, then click Verify."
            elif not srcs:
                st.session_state["phase4_report"] = "No source chunks were returned for the last answer — nothing to verify."
            else:
                a = st.session_state.get("last_eval_answer", "")
                with st.spinner("Verifying citations..."):
                    citation_result = verify_citations(a, srcs)
                st.session_state["phase4_report"] = format_citation_verification(citation_result, q)
        if st.session_state.get("phase4_report"):
            _field_label("CITATION VERIFICATION")
            st.code(st.session_state["phase4_report"], language=None)


def render_golden_tab():
    summary = load_golden_eval_summary()
    if summary is None:
        st.info("No full golden eval run found. Run: python run_golden_eval.py")
        return

    correct, total = summary["overall"]
    _section_label("OVERALL", "target")
    with st.container(border=True):
        st.metric("Correctness", f"{correct / total:.1%}", f"{correct}/{total} questions")
        st.caption(f"Last full run: {summary['timestamp']}")

    st.write("")
    _section_label("BY CATEGORY", "grid")
    with st.container(border=True):
        for cat in ["lookup", "multi_hop", "no_answer", "ambiguous"]:
            if cat in summary["by_category"]:
                c, n = summary["by_category"][cat]
                _bar_row(cat.replace("_", " "), c, n)


def main():
    st.markdown(_FONTS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-header">
          <h1>RAG Pipeline Dashboard</h1>
          <p>Hybrid BM25 + dense retrieval, cross-encoder reranking, CRAG grading,
          citation verification, and a composite confidence score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_ask, tab_golden = st.tabs(["PDF Upload", "Golden Set"])
    with tab_ask:
        render_ask_tab()
    with tab_golden:
        render_golden_tab()


if __name__ == "__main__":
    main()
