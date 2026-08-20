
import logging

# Configured before any other import so INFO logs emitted at import time
# (e.g. rag_pipeline's LangSmith setup log) aren't silently dropped, and
# force=True wins even if a dependency (transformers/huggingface_hub/etc.)
# already called basicConfig() as a side effect, which would otherwise make
# this a no-op and leave log level/format inconsistent across modules.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

import gradio as gr  # noqa: E402
import html  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime  # noqa: E402
from functools import partial  # noqa: E402
from app.rag_pipeline import (  # noqa: E402
    build_vector_store,
    make_qa_chain,
    check_input_guard,
    redact_pii,
    evaluate_rag_response,
    dense_only_retrieve,
    _SUPPORTED_EXTENSIONS,
    Config,
)
from app.citations import verify_citations  # noqa: E402
from app.session_store import (  # noqa: E402
    save_history as _save_history,
    load_history as _load_history,
    delete_history as _delete_history,
    save_feedback as _save_feedback,
    get_feedback_stats as _get_feedback_stats,
)
from app.report_formatting import (  # noqa: E402
    format_retrieval_eval,
    format_ragas_eval,
    format_citation_verification,
    format_answer_html,
    format_bar_html,
    format_source_detail_html,
    format_comparison_html,
)
from app.golden_eval_summary import load_golden_eval_summary  # noqa: E402

MAX_SOURCE_BUTTONS = 6
_BLANK_ANSWER_HTML = '<div class="answer-card"><p style="color:#C8A080;">No answer yet.</p></div>'


def strip_emojis(text):
    """Remove emojis and non-standard unicode symbols from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002700-\U000027bf"
        "\U0001f900-\U0001f9ff"
        "\U00002600-\U000026ff"
        "\U00002b50-\U00002b55"
        "\U0000fe0f"
        "\U0000200d"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


# ---------- Global State ----------
loaded_docs = {}          # {name: {"chain": QAChainWrapper, "vs": vector_store}}
current_doc_name = None
question_count = 0
last_eval_data = {"question": "", "answer": "", "contexts": [], "sources": []}   # for RAGAS + citation verification buttons


def format_stats():
    doc_info = f"Active: {current_doc_name}" if current_doc_name else "No document loaded"
    cache_info = ""
    feedback_info = ""
    if current_doc_name and current_doc_name in loaded_docs:
        chain = loaded_docs[current_doc_name]["chain"]
        size = chain.cache_size()
        backend = chain._cache.backend if hasattr(chain, "_cache") else "memory"
        cache_info = f"  |  Cache [{backend}]: {size} entries"
        fb = _get_feedback_stats(current_doc_name)
        if fb["total"] > 0:
            feedback_info = (
                f"  |  Feedback: {fb['up']} up / {fb['down']} down"
                f"  (Success rate: {fb['rate']}%)"
            )
    return (
        f"Questions asked: {question_count}  |  {doc_info}"
        f"  |  Docs loaded: {len(loaded_docs)}{cache_info}{feedback_info}"
    )


def _no_source_buttons():
    return [gr.update(visible=False) for _ in range(MAX_SOURCE_BUTTONS)]


def _source_buttons_for(sources, selected=None):
    updates = []
    for i in range(MAX_SOURCE_BUTTONS):
        if i < len(sources):
            chunk_num = sources[i]["chunk"]
            updates.append(gr.update(
                value=f"Source {chunk_num}",
                visible=True,
                variant="primary" if selected == chunk_num else "secondary",
            ))
        else:
            updates.append(gr.update(visible=False))
    return updates


def _reset_answer_panels():
    """(answer_html, meta_line_html, confidence_html, sources_state,
    selected_source_state, 6x source buttons, source_detail_html,
    comparison_hybrid_html, comparison_dense_html) -- used whenever the
    active document changes so a stale answer/source set from a different
    document can't linger on screen."""
    return (
        _BLANK_ANSWER_HTML, "", "", [], None,
        *_no_source_buttons(),
        "", "", "",
    )


def load_document(doc_file):
    global loaded_docs, current_doc_name

    if doc_file is None:
        return ("No file uploaded.", gr.update(), format_stats(), "", *_reset_answer_panels())

    name = os.path.basename(doc_file.name)
    ext = os.path.splitext(name)[1].lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        return (
            f"Unsupported file type '{ext}'. Supported: {supported}",
            gr.update(), format_stats(), "", *_reset_answer_panels(),
        )

    try:
        tmp = tempfile.mkdtemp()
        store_path = os.path.join(tmp, "vector_store")

        vector_store, all_chunks = build_vector_store(doc_file.name, store_path, doc_id=name)
        qa_chain = make_qa_chain(vector_store, doc_id=name, all_chunks=all_chunks)

        loaded_docs[name] = {"chain": qa_chain, "vs": vector_store}
        current_doc_name = name

        choices = list(loaded_docs.keys())
        status = f"'{name}' loaded. {len(choices)} document(s) ready."
        restored_history = _load_history(name)
        if restored_history:
            status += " (conversation history restored)"

        return (
            status, gr.update(choices=choices, value=name), format_stats(), restored_history,
            *_reset_answer_panels(),
        )

    except Exception as e:
        return (f"Failed to load document: {str(e)}", gr.update(), format_stats(), "", *_reset_answer_panels())


def switch_document(selected_name):
    global current_doc_name
    if selected_name and selected_name in loaded_docs:
        current_doc_name = selected_name
        history = _load_history(selected_name)
        return (f"Switched to: '{selected_name}'", history, *_reset_answer_panels())
    return ("Document not found.", "", *_reset_answer_panels())


def clear_cache():
    """Clear the semantic cache for the active document."""
    if current_doc_name and current_doc_name in loaded_docs:
        loaded_docs[current_doc_name]["chain"].clear_cache()
        return f"Cache cleared for '{current_doc_name}'.", format_stats()
    return "No document loaded.", format_stats()


def ask_question(question, history_text):
    """Generator — streams answer tokens and updates UI in real time.

    Yields a fixed-shape tuple matching _ASK_OUTPUTS every time (Gradio
    requires consistent output arity across a generator's yields); slots not
    relevant to a given branch (e.g. confidence/sources on an early-return)
    are left untouched via gr.update() no-ops.
    """
    global question_count, last_eval_data

    NOOP = gr.update()
    NOOP_BTNS = [gr.update() for _ in range(MAX_SOURCE_BUTTONS)]

    if not question.strip():
        yield "", history_text, NOOP, NOOP, NOOP, NOOP, NOOP, *NOOP_BTNS, NOOP, NOOP, NOOP, format_stats()
        return

    # Input Guard
    is_safe, reason = check_input_guard(question)
    if not is_safe:
        yield (
            "", history_text,
            f'<div class="answer-card"><p>[Input Guard] {html.escape(reason)}</p></div>',
            '<div class="meta-line">Request blocked before retrieval.</div>',
            NOOP, NOOP, NOOP, *NOOP_BTNS, NOOP, NOOP, NOOP, format_stats(),
        )
        return

    if current_doc_name is None or current_doc_name not in loaded_docs:
        yield (
            "", history_text,
            '<div class="answer-card"><p>Please load a document first using Step 1.</p></div>',
            "", NOOP, NOOP, NOOP, *NOOP_BTNS, NOOP, NOOP, NOOP, format_stats(),
        )
        return

    qa_chain = loaded_docs[current_doc_name]["chain"]
    vector_store = loaded_docs[current_doc_name]["vs"]
    recent_history = history_text[-500:] if len(history_text) > 500 else history_text

    t_start = time.time()
    yield (
        "", history_text,
        '<div class="answer-card"><p style="color:#C8A080;">Retrieving and reranking...</p></div>',
        "", NOOP, NOOP, NOOP, *NOOP_BTNS, NOOP, NOOP, NOOP, format_stats(),
    )

    answer = ""
    sources = []
    answer_html_val = _BLANK_ANSWER_HTML
    meta_html_val = ""
    confidence_html_val = ""
    btn_updates = _no_source_buttons()
    detail_html = ""
    hybrid_html = ""
    dense_html = ""
    t_retrieval_done = None

    try:
        final_update = None

        for update in qa_chain.stream({"query": question, "history": recent_history}):
            if not update["done"]:
                if t_retrieval_done is None:
                    t_retrieval_done = time.time()  # first token = retrieval done
                answer = strip_emojis(update["result"])
                yield (
                    "", history_text, f'<div class="answer-card">{format_answer_html(answer)}</div>',
                    '<div class="meta-line">Generating...</div>',
                    NOOP, NOOP, NOOP, *NOOP_BTNS, NOOP, NOOP, NOOP, format_stats(),
                )
            else:
                final_update = update

        if final_update is None:
            yield (
                "", history_text, '<div class="answer-card"><p>No response generated.</p></div>',
                "", NOOP, NOOP, NOOP, *NOOP_BTNS, NOOP, NOOP, NOOP, format_stats(),
            )
            return

        raw_answer = final_update["result"]
        grade = final_update.get("grade", "")
        query_type = final_update.get("query_type", "")
        sources = final_update.get("sources", [])
        cache_hit = final_update.get("cache_hit", False)
        confidence = final_update.get("confidence", {})

        # Output Guard — redact PII from the generated answer
        clean_answer, redacted_types = redact_pii(raw_answer)
        answer = strip_emojis(clean_answer)
        answer_html_val = f'<div class="answer-card">{format_answer_html(answer)}</div>'

        # Meta line — CRAG grade / query type / cache hit / PII / latency.
        # Streamlit's dashboard doesn't surface these, but they're kept here
        # since they're the concrete evidence behind the header tagline's own
        # claims (CRAG grading, semantic cache) — dropping them would make
        # the tagline nothing you can actually see working.
        labels = []
        if cache_hit:
            labels.append("CACHE HIT")
        if grade:
            labels.append(f"CRAG: {grade}")
        if query_type and query_type != "N/A":
            labels.append(f"Type: {query_type}")
        if redacted_types:
            labels.append(f"PII redacted: {', '.join(sorted(set(redacted_types)))}")

        t_end = time.time()
        total_ms = int((t_end - t_start) * 1000)
        if cache_hit:
            labels.append(f"Served from semantic cache: {total_ms}ms")
        else:
            ttft_ms = int((t_retrieval_done - t_start) * 1000) if t_retrieval_done else 0
            generation_ms = total_ms - ttft_ms
            generated_tokens = (final_update or {}).get("generated_tokens", 0)
            tokens_per_sec = round(generated_tokens / (generation_ms / 1000), 1) if generation_ms > 0 and generated_tokens else 0.0
            labels.append(f"TTFT {ttft_ms}ms · Generation {generation_ms}ms · {tokens_per_sec} tok/s")

        meta_html_val = (
            f'<div class="meta-line">{" &middot; ".join(html.escape(label) for label in labels)}</div>'
            if labels else ""
        )

        # Confidence
        if confidence:
            composite = confidence.get("composite", 0)
            bars = "".join([
                format_bar_html("Retrieval", f"{confidence.get('retrieval', 0):.2f}", confidence.get("retrieval", 0)),
                format_bar_html("Citation coverage", f"{confidence.get('citation_coverage', 0):.2f}", confidence.get("citation_coverage", 0)),
                format_bar_html("Completeness", f"{confidence.get('completeness', 0):.2f}", confidence.get("completeness", 0)),
            ])
            confidence_html_val = f'<div class="confidence-composite">{composite:.2f}</div>{bars}'
        else:
            confidence_html_val = '<div class="snippet" style="color:#C8A080;">No confidence data.</div>'

        # Sources + hybrid-vs-dense-only comparison
        if sources:
            btn_updates = _source_buttons_for(sources, selected=None)
            try:
                dense_docs, _dense_grade = dense_only_retrieve(vector_store, question)
            except Exception:
                dense_docs = []
            hybrid_html, dense_html = format_comparison_html(sources, dense_docs)
        elif grade == "INCORRECT":
            detail_html = '<div class="snippet" style="color:#C8A080;">No relevant chunks found in document.</div>'
        else:
            detail_html = '<div class="snippet" style="color:#C8A080;">No source chunks returned.</div>'

    except Exception as e:
        answer = f"Error: {str(e)}"
        answer_html_val = f'<div class="answer-card"><p>{html.escape(answer)}</p></div>'
        sources = []

    question_count += 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    separator = ". " * 30
    clean_question = strip_emojis(question)
    new_entry = (
        f"[{timestamp}]  You: {clean_question}\n\n"
        f"Assistant: {answer}\n\n"
        f"{separator}\n\n"
    )
    updated_history = history_text + new_entry

    # Store for RAGAS evaluation button + citation verification button
    if sources:
        last_eval_data["question"] = question
        last_eval_data["answer"] = answer
        last_eval_data["contexts"] = [s["content"] for s in sources]
        last_eval_data["sources"] = sources

    # Persist conversation history to Redis
    _save_history(current_doc_name, updated_history)

    yield (
        "", updated_history, answer_html_val, meta_html_val,
        confidence_html_val, sources, None,
        *btn_updates,
        detail_html, hybrid_html, dense_html,
        format_stats(),
    )


def select_source(idx, sources, selected):
    """idx is the 1-based position of the clicked button among the currently
    visible source buttons, which matches the sources list's own order."""
    if not sources or idx > len(sources):
        return (selected, gr.update(), *[gr.update() for _ in range(MAX_SOURCE_BUTTONS)])

    chunk_num = sources[idx - 1]["chunk"]
    new_selected = None if selected == chunk_num else chunk_num

    if new_selected is None:
        detail = ""
    else:
        match = next((s for s in sources if s["chunk"] == new_selected), None)
        detail = format_source_detail_html(match) if match else ""

    return (new_selected, detail, *_source_buttons_for(sources, selected=new_selected))


def clear_history():
    _delete_history(current_doc_name)
    return ("", *_reset_answer_panels())


def run_phase1_eval():
    """
    Phase 1 Retrieval Evaluation.
    Generates a synthetic test set from the active document's chunks,
    then computes Precision@K, Recall@K, MRR, and Coverage.
    Expects ~10-20 seconds (one LLM call per question generated).
    """
    if current_doc_name is None or current_doc_name not in loaded_docs:
        return "Load a document first."

    chain = loaded_docs[current_doc_name]["chain"]
    if not hasattr(chain, "run_retrieval_eval"):
        return "Retrieval eval not available on this chain."

    result = chain.run_retrieval_eval(n_questions=8, k=Config.RETRIEVAL_K)
    return format_retrieval_eval(result, current_doc_name)


def thumbs_up():
    q = last_eval_data.get("question", "")
    a = last_eval_data.get("answer", "")
    if not q:
        return "Ask a question first.", format_stats()
    _save_feedback(current_doc_name, q, a, "up")
    fb = _get_feedback_stats(current_doc_name)
    return f"Recorded: thumbs up  ({fb['up']} up / {fb['down']} down total)", format_stats()


def thumbs_down():
    q = last_eval_data.get("question", "")
    a = last_eval_data.get("answer", "")
    if not q:
        return "Ask a question first.", format_stats()
    _save_feedback(current_doc_name, q, a, "down")
    fb = _get_feedback_stats(current_doc_name)
    return f"Recorded: thumbs down  ({fb['up']} up / {fb['down']} down total)", format_stats()


def run_evaluation():
    """Run RAGAS on the last question/answer/contexts and format a score report."""
    q = last_eval_data.get("question", "")
    a = last_eval_data.get("answer", "")
    ctxs = last_eval_data.get("contexts", [])

    if not q:
        return "Ask a question first, then click Evaluate."

    scores = evaluate_rag_response(q, a, ctxs)
    return format_ragas_eval(scores, q)


def run_citation_verification():
    """
    Parse the last answer's superscript citations and check each one against
    its cited source chunk's full text via LLM-as-judge. Flags unsupported
    citations instead of trusting that a citation number means the claim is
    actually grounded.
    """
    q = last_eval_data.get("question", "")
    a = last_eval_data.get("answer", "")
    sources = last_eval_data.get("sources", [])

    if not q:
        return "Ask a question first, then click Verify."
    if not sources:
        return "No source chunks were returned for the last answer — nothing to verify."

    result = verify_citations(a, sources)
    return format_citation_verification(result, q)


# ---------- UI — Theme & CSS ----------
_supported_ext_list = sorted(_SUPPORTED_EXTENSIONS)
_file_types_display = ", ".join(_supported_ext_list)

# Outline icons (Feather-style: stroke-based, no fill) -- same set as
# streamlit_app.py, used next to section labels instead of emoji.
_ICON_ATTRS = 'width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
_ICONS = {
    "document": f'<svg {_ICON_ATTRS}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "message": f'<svg {_ICON_ATTRS}><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>',
    "check": f'<svg {_ICON_ATTRS}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "activity": f'<svg {_ICON_ATTRS}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "bookmark": f'<svg {_ICON_ATTRS}><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
    "layers": f'<svg {_ICON_ATTRS}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "target": f'<svg {_ICON_ATTRS}><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "thumbsup": f'<svg {_ICON_ATTRS}><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>',
    "grid": f'<svg {_ICON_ATTRS}><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
}


def _section_header(text: str, icon: str = None):
    icon_html = _ICONS.get(icon, "") if icon else ""
    return gr.HTML(f'<div class="section-header">{icon_html}{text}</div>')


_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

# Palette: #FFD3AC (light peach) · #FFB5AB (rose) · #E39A7B (terracotta) · #DBB06B (gold)
_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.stone,
).set(
    # Page
    body_background_fill="#FFF8F2",
    body_background_fill_dark="#FFF8F2",
    body_text_color="#3D1A06",
    body_text_color_subdued="#8B5030",
    # Blocks / cards
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#FFD3AC",
    block_border_color_dark="#FFD3AC",
    block_label_text_color="#A06030",
    block_label_text_color_dark="#A06030",
    block_title_text_color="#7A4020",
    block_title_text_color_dark="#7A4020",
    # Inputs
    input_background_fill="#FFF5EE",
    input_background_fill_dark="#FFF5EE",
    input_border_color="#FFD3AC",
    input_border_color_dark="#FFD3AC",
    input_border_color_focus="#E39A7B",
    input_border_color_focus_dark="#E39A7B",
    # Buttons — primary
    button_primary_background_fill="#DBB06B",
    button_primary_background_fill_hover="#E39A7B",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="transparent",
    # Buttons — secondary
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#FFD3AC",
    button_secondary_text_color="#E39A7B",
    button_secondary_border_color="#FFB5AB",
    button_secondary_border_color_hover="#E39A7B",
    # Fills
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#FFF5EE",
    # Borders & accent
    border_color_accent="#E39A7B",
    border_color_primary="#FFD3AC",
    color_accent="#E39A7B",
    color_accent_soft="rgba(227,154,123,0.18)",
    # Shadows
    shadow_drop="0 2px 14px rgba(219,176,107,0.14)",
    shadow_drop_lg="0 6px 28px rgba(227,154,123,0.18)",
    # Links
    link_text_color="#E39A7B",
    link_text_color_hover="#DBB06B",
    link_text_color_visited="#A06030",
)

_CSS = """
/* ── Fonts ── */
gradio-app, .gradio-container, body, button, input, textarea, select {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* ── Page ── */
gradio-app, .gradio-container {
    background:
        radial-gradient(640px circle at 12% 8%, rgba(219,176,107,0.22), transparent 60%),
        radial-gradient(720px circle at 88% 6%, rgba(227,154,123,0.20), transparent 60%),
        radial-gradient(680px circle at 78% 92%, rgba(255,181,171,0.20), transparent 60%),
        radial-gradient(600px circle at 8% 88%, rgba(219,176,107,0.16), transparent 60%),
        linear-gradient(160deg, #FFF8F2 0%, #FFF0E6 50%, #FFE8DA 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}
.gradio-container {
    max-width: 1360px !important;
    margin: 0 auto !important;
    padding: 28px 28px !important;
}

/* ── Header band ── */
.app-header {
    background: linear-gradient(135deg, #DBB06B 0%, #E39A7B 55%, #FFB5AB 100%) !important;
    border-radius: 18px !important;
    padding: 32px 40px 28px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 8px 32px rgba(219,176,107,0.28) !important;
    text-align: center !important;
}
.app-header h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: #FFFFFF !important;
    font-size: 2.4em !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    margin: 0 0 8px 0 !important;
    text-shadow: 0 2px 8px rgba(100,40,10,0.18) !important;
}
.app-header p {
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.9em !important;
    margin: 0 !important;
    letter-spacing: 0.02em !important;
}

/* ── Section headers ── */
.section-header {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    background: linear-gradient(135deg, #E39A7B 0%, #DBB06B 100%) !important;
    color: #FFFFFF !important;
    font-size: 0.74em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 7px 16px !important;
    margin-bottom: 4px !important;
    box-shadow: 0 2px 8px rgba(227,154,123,0.22) !important;
}
.section-header svg { flex-shrink: 0 !important; }

/* ── Stats bar ── */
#stats-bar textarea {
    color: #7A4020 !important;
    font-size: 0.82em !important;
    font-family: 'Courier New', monospace !important;
    background: #FFF5EE !important;
    border-color: #FFD3AC !important;
    border-radius: 8px !important;
}

/* ── Field labels ── */
label > span:first-child {
    color: #A06030 !important;
    font-size: 0.76em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}

/* ── All textareas & text inputs ── */
textarea, input[type=text] {
    color: #3D1A06 !important;
    caret-color: #E39A7B !important;
    line-height: 1.68 !important;
    border-radius: 10px !important;
}
textarea::placeholder, input[type=text]::placeholder {
    color: #C8A080 !important;
}

/* ── Question input ── */
#question-input textarea {
    font-size: 1.0em !important;
    background: #FFFFFF !important;
    border: 2px solid #FFD3AC !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
#question-input textarea:focus {
    border-color: #E39A7B !important;
    box-shadow: 0 0 0 4px rgba(227,154,123,0.14) !important;
}

/* ── Chat history ── */
#chat-history textarea {
    font-size: 0.91em !important;
    line-height: 1.78 !important;
    color: #4A2410 !important;
    background: #FFFBF8 !important;
    border-color: #FFD3AC !important;
}

/* ── Answer card ── */
.answer-card {
    background: #FFFEF8 !important;
    border: 1px solid #FFD3AC !important;
    border-left: 4px solid #DBB06B !important;
    border-radius: 10px !important;
    padding: 22px 26px !important;
    font-size: 1.06em !important;
    line-height: 1.85 !important;
    color: #3D1A06 !important;
}
.answer-card p { margin: 0 0 14px 0 !important; font-size: 1em !important; line-height: inherit !important; }
.answer-card p:last-child { margin-bottom: 0 !important; }
.answer-card .cite {
    color: #C9713F !important;
    font-weight: 700 !important;
    font-size: 1.15em !important;
    padding: 0 1px !important;
}
.answer-card .analogy {
    display: block !important;
    margin-top: 14px !important;
    padding-top: 14px !important;
    border-top: 1px dashed #FFD3AC !important;
    color: #7A4020 !important;
    font-style: italic !important;
}

/* ── Snippet / meta-line (sources, comparison, golden set) ── */
.snippet {
    font-size: 0.88em !important;
    line-height: 1.7 !important;
    color: #5A3010 !important;
}
.meta-line {
    font-size: 0.78em !important;
    color: #A06030 !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 4px !important;
}

/* ── Card blocks (confidence, source detail, comparison, golden set) ── */
.card-block {
    background: #FFFFFF !important;
    border: 1px solid #FFD3AC !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 2px 14px rgba(219,176,107,0.10) !important;
}
.confidence-composite {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.2em !important;
    font-weight: 700 !important;
    color: #7A4020 !important;
    line-height: 1 !important;
    margin-bottom: 14px !important;
}

/* ── Load status ── */
#load-status textarea {
    color: #6B3820 !important;
    font-size: 0.88em !important;
    background: #FFF5EE !important;
    border-color: #FFD3AC !important;
}

/* ── Feedback status ── */
#feedback-status textarea {
    color: #6B3820 !important;
    font-size: 0.88em !important;
    background: #FFF5EE !important;
    border-color: #FFD3AC !important;
}

/* ── Eval output boxes ── */
#eval-box textarea, #phase1-box textarea {
    color: #5A3010 !important;
    font-size: 0.83em !important;
    font-family: 'Courier New', monospace !important;
    line-height: 1.62 !important;
    background: #FFF5EE !important;
    border-color: #FFD3AC !important;
}

/* ── Primary button ── */
button.primary {
    background: linear-gradient(130deg, #DBB06B 0%, #E39A7B 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 12px rgba(219,176,107,0.32) !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.02em !important;
}
button.primary:hover {
    background: linear-gradient(130deg, #E39A7B 0%, #DBB06B 100%) !important;
    box-shadow: 0 6px 20px rgba(227,154,123,0.42) !important;
    transform: translateY(-2px) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* ── Secondary button ── */
button.secondary {
    background: #FFFFFF !important;
    color: #E39A7B !important;
    border: 1.5px solid #FFB5AB !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
button.secondary:hover {
    border-color: #E39A7B !important;
    color: #DBB06B !important;
    background: #FFF5EE !important;
    box-shadow: 0 3px 10px rgba(227,154,123,0.18) !important;
}

/* ── Stop / thumbs-down button ── */
button.stop {
    background: #FFFFFF !important;
    color: #E39A7B !important;
    border: 1.5px solid #FFB5AB !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
button.stop:hover {
    background: #FFF0EA !important;
    border-color: #E39A7B !important;
    box-shadow: 0 3px 10px rgba(227,154,123,0.2) !important;
}

/* ── Tabs ──
   `.tab-nav > button` doesn't exist in this Gradio version's DOM (verified
   via getComputedStyle/matches() -- the real structure is
   [role=tablist] > .tab-container > button[role=tab], no "tab-nav" class
   anywhere), so that selector never matched and every unselected tab fell
   through to Gradio's own default (var(--body-text-color), which resolves
   to the neutral-hue's near-white shade here) -- functionally invisible.
   ARIA role attributes are stable across Gradio versions; class names aren't. */
.gradio-container [role="tablist"] {
    border-bottom: 2px solid #FFD3AC !important;
    background: transparent !important;
    margin-bottom: 16px !important;
}
.gradio-container [role="tablist"] button[role="tab"] {
    color: #A06030 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
    transition: color 0.15s !important;
    margin-bottom: -2px !important;
}
.gradio-container [role="tablist"] button[role="tab"][aria-selected="true"] {
    color: #E39A7B !important;
    border-bottom-color: #E39A7B !important;
}
.gradio-container [role="tablist"] button[role="tab"]:hover { color: #DBB06B !important; }

/* ── File upload ── */
.upload-button-container > button {
    background: #FFF5EE !important;
    border: 2px dashed #FFB5AB !important;
    color: #C8A080 !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
.upload-button-container > button:hover {
    border-color: #E39A7B !important;
    color: #E39A7B !important;
    background: #FFF0EA !important;
}

/* ── Dropdown ── */
.wrap-inner {
    background: #FFFFFF !important;
    border-color: #FFD3AC !important;
    color: #3D1A06 !important;
}
/* Selected value text — covers all Gradio 5 dropdown internals */
.wrap-inner *, .wrap-inner input, .wrap-inner span,
.wrap-inner .value, .wrap-inner .token,
.multiselect span, .multiselect input,
.svelte-select span, .svelte-select input {
    color: #3D1A06 !important;
    background: transparent !important;
}
.wrap-inner input::placeholder {
    color: #C8A080 !important;
}
.options { background: #FFFAF5 !important; border-color: #FFD3AC !important; }
li.item { color: #3D1A06 !important; }
li.item:hover, li.item.active {
    background: #FFD3AC !important;
    color: #7A4020 !important;
}
li.item.selected {
    background: rgba(219,176,107,0.18) !important;
    color: #7A4020 !important;
}

/* ── Horizontal rules ── */
hr {
    border: none !important;
    border-top: 1.5px solid #FFD3AC !important;
    margin: 22px 0 !important;
}

/* ── Markdown / prose ── */
.prose p { color: #8B5030 !important; font-size: 0.9em !important; }
.prose strong { color: #7A4020 !important; }
.prose a { color: #E39A7B !important; }
.prose a:hover { color: #DBB06B !important; }
.prose code {
    background: #FFD3AC !important;
    color: #7A4020 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.88em !important;
}

/* ── Block border & shadow override ── */
.block {
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(219,176,107,0.1) !important;
}

/* ── Footer text ── */
.footer-text p, .footer-text a {
    color: #C8A080 !important;
    font-size: 0.82em !important;
}
.footer-text a:hover { color: #E39A7B !important; }
"""

# ================================================================
# UI LAYOUT
# ================================================================
with gr.Blocks(
    title="RAG Chatbot — Chat with Any Document",
) as demo:

    # ── Header ──────────────────────────────────────────────────
    with gr.Column(elem_classes="app-header"):
        gr.Markdown("# RAG Chatbot — Chat with Any Document")
        gr.Markdown(
            f"Supports **{_file_types_display}** &nbsp;·&nbsp; "
            "Hybrid BM25+FAISS &nbsp;·&nbsp; Cross-encoder reranking &nbsp;·&nbsp; "
            "CRAG grading &nbsp;·&nbsp; Semantic cache &nbsp;·&nbsp; PII redaction"
        )

    with gr.Tabs():
        with gr.Tab("PDF Upload"):
            stats_bar = gr.Textbox(
                value=format_stats(),
                label="Session Stats",
                interactive=False,
                lines=1,
                elem_id="stats-bar",
            )

            # ── Step 1 — Document Upload ─────────────────────────
            _section_header("STEP 1 — UPLOAD DOCUMENT", "document")

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    doc_input = gr.File(
                        label="Choose a document to upload",
                        file_types=_supported_ext_list,
                    )
                with gr.Column(scale=4):
                    load_status = gr.Textbox(
                        label="Load Status",
                        value="No document loaded yet.",
                        interactive=False,
                        lines=5,
                        elem_id="load-status",
                    )

            with gr.Row():
                load_btn = gr.Button("Load Document", variant="primary", scale=2, size="lg")
                doc_selector = gr.Dropdown(
                    label="Switch Active Document",
                    choices=[],
                    interactive=True,
                    scale=3,
                )

            gr.HTML("<hr/>")

            # ── Step 2 — Chat ────────────────────────────────────
            _section_header("STEP 2 — ASK QUESTIONS", "message")

            conversation = gr.Textbox(
                label="Conversation History",
                value="",
                interactive=False,
                lines=13,
                max_lines=13,
                elem_id="chat-history",
            )

            with gr.Row(equal_height=True):
                question_box = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the loaded document...",
                    lines=2,
                    scale=6,
                    elem_id="question-input",
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1, size="lg", min_width=90)

            with gr.Row():
                clear_btn = gr.Button("Clear Conversation", size="sm", variant="secondary", scale=1)
                clear_cache_btn = gr.Button("Clear Semantic Cache", size="sm", variant="secondary", scale=1)

            gr.HTML("<hr/>")

            # ── Answer ────────────────────────────────────────────
            _section_header("ANSWER", "check")
            answer_html = gr.HTML(value=_BLANK_ANSWER_HTML)
            meta_line_html = gr.HTML(value="")

            gr.HTML("<div style='height:8px'></div>")

            # ── Confidence ────────────────────────────────────────
            _section_header("CONFIDENCE", "activity")
            confidence_html = gr.HTML(value="", elem_classes="card-block")

            gr.HTML("<div style='height:8px'></div>")

            # ── Sources (clickable citations) ───────────────────
            _section_header("SOURCES", "bookmark")
            gr.Markdown("Click a source to view the chunk it was cited from.")
            with gr.Row():
                src_btn_1 = gr.Button("Source 1", variant="secondary", visible=False)
                src_btn_2 = gr.Button("Source 2", variant="secondary", visible=False)
                src_btn_3 = gr.Button("Source 3", variant="secondary", visible=False)
                src_btn_4 = gr.Button("Source 4", variant="secondary", visible=False)
                src_btn_5 = gr.Button("Source 5", variant="secondary", visible=False)
                src_btn_6 = gr.Button("Source 6", variant="secondary", visible=False)
            source_detail_html = gr.HTML(value="", elem_classes="card-block")

            gr.HTML("<div style='height:8px'></div>")

            # ── Retrieval comparison ─────────────────────────────
            _section_header("RETRIEVAL COMPARISON — HYBRID VS. DENSE-ONLY", "layers")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Hybrid** (BM25 + dense, RRF-fused)")
                    comparison_hybrid_html = gr.HTML(value="", elem_classes="card-block")
                with gr.Column():
                    gr.Markdown("**Dense-only** (no BM25)")
                    comparison_dense_html = gr.HTML(value="", elem_classes="card-block")

            sources_state = gr.State([])
            selected_source_state = gr.State(None)

            gr.HTML("<hr/>")

            # ── Feedback ──────────────────────────────────────────
            _section_header("RATE THE LAST ANSWER", "thumbsup")

            with gr.Row():
                thumbs_up_btn = gr.Button("Thumbs Up", variant="primary", size="sm", scale=1)
                thumbs_down_btn = gr.Button("Thumbs Down", variant="stop", size="sm", scale=1)
                feedback_status = gr.Textbox(
                    label="Feedback Status",
                    interactive=False,
                    scale=5,
                    lines=1,
                    elem_id="feedback-status",
                )

            gr.HTML("<hr/>")

            # ── Step 3 — Evaluation ───────────────────────────────
            _section_header("STEP 3 — EVALUATION", "grid")

            with gr.Tabs():
                with gr.Tab("Phase 1 — Retrieval Eval"):
                    gr.Markdown(
                        "Generates synthetic questions from document chunks — measures **Precision@K · Recall@K · MRR · Coverage**.  \n"
                        "Takes ~20 seconds. No extra API keys needed."
                    )
                    with gr.Row():
                        phase1_btn = gr.Button("Run Retrieval Evaluation", variant="secondary", scale=1)
                        phase1_box = gr.Textbox(
                            label="Retrieval Metrics",
                            value="",
                            interactive=False,
                            lines=10,
                            scale=3,
                            elem_id="phase1-box",
                        )

                with gr.Tab("Phase 2 — RAGAS Eval"):
                    gr.Markdown(
                        "Scores the last answer on **Faithfulness · Answer Relevancy · Context Precision** via LLM-as-judge.  \n"
                        "Requires `RAGAS_EVAL=true` in `.env`. Uses your configured Groq LLM — takes 10-20 seconds."
                    )
                    with gr.Row():
                        eval_btn = gr.Button("Evaluate Last Answer (RAGAS)", variant="secondary", scale=1)
                        eval_box = gr.Textbox(
                            label="RAGAS Scores",
                            value="",
                            interactive=False,
                            lines=10,
                            scale=3,
                            elem_id="eval-box",
                        )

                with gr.Tab("Phase 4 — Citation Verification"):
                    gr.Markdown(
                        "Checks whether each cited superscript (¹²³) in the last answer is actually "
                        "supported by the source chunk it points to, and flags claims with no citation at all.  \n"
                        "Uses your configured LLM — one extra call per cited sentence, takes a few seconds."
                    )
                    with gr.Row():
                        verify_btn = gr.Button("Verify Citations (Last Answer)", variant="secondary", scale=1)
                        citation_box = gr.Textbox(
                            label="Citation Verification",
                            value="",
                            interactive=False,
                            lines=10,
                            scale=3,
                            elem_id="citation-box",
                        )

            gr.Markdown(
                "Built with [LangChain](https://langchain.com) · "
                "[FAISS](https://github.com/facebookresearch/faiss) · "
                "[BM25](https://github.com/dorianbrown/rank_bm25) · "
                "[Gradio](https://gradio.app)",
                elem_classes="footer-text",
            )

        with gr.Tab("Golden Set"):
            _summary = load_golden_eval_summary()
            if _summary is None:
                gr.Markdown("No full golden eval run found. Run: `python run_golden_eval.py`")
            else:
                _correct, _total = _summary["overall"]
                _section_header("OVERALL", "target")
                gr.HTML(
                    f'<div class="card-block"><div class="confidence-composite">{_correct / _total:.1%}</div>'
                    f'<div class="snippet">{_correct}/{_total} questions &middot; last full run: {_summary["timestamp"]}</div></div>'
                )
                gr.HTML("<div style='height:12px'></div>")
                _section_header("BY CATEGORY", "grid")
                _cat_bars = "".join(
                    format_bar_html(cat.replace("_", " "), f"{c}/{n} · {c / n:.0%}", c / n)
                    for cat in ["lookup", "multi_hop", "no_answer", "ambiguous"]
                    if cat in _summary["by_category"]
                    for c, n in [_summary["by_category"][cat]]
                )
                gr.HTML(f'<div class="card-block">{_cat_bars}</div>')

    # ── Event wiring ─────────────────────────────────────────────
    _panel_outputs = [
        answer_html, meta_line_html, confidence_html, sources_state, selected_source_state,
        src_btn_1, src_btn_2, src_btn_3, src_btn_4, src_btn_5, src_btn_6,
        source_detail_html, comparison_hybrid_html, comparison_dense_html,
    ]

    load_btn.click(
        fn=load_document,
        inputs=doc_input,
        outputs=[load_status, doc_selector, stats_bar, conversation, *_panel_outputs],
    )
    doc_selector.change(
        fn=switch_document,
        inputs=doc_selector,
        outputs=[load_status, conversation, *_panel_outputs],
    )
    _ask_outputs = [question_box, conversation, *_panel_outputs, stats_bar]
    ask_btn.click(fn=ask_question, inputs=[question_box, conversation], outputs=_ask_outputs)
    question_box.submit(fn=ask_question, inputs=[question_box, conversation], outputs=_ask_outputs)

    for _i, _btn in enumerate([src_btn_1, src_btn_2, src_btn_3, src_btn_4, src_btn_5, src_btn_6], start=1):
        _btn.click(
            fn=partial(select_source, _i),
            inputs=[sources_state, selected_source_state],
            outputs=[selected_source_state, source_detail_html, src_btn_1, src_btn_2, src_btn_3, src_btn_4, src_btn_5, src_btn_6],
        )

    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[conversation, *_panel_outputs],
    )
    clear_cache_btn.click(
        fn=clear_cache,
        inputs=[],
        outputs=[load_status, stats_bar],
    )
    eval_btn.click(fn=run_evaluation, inputs=[], outputs=[eval_box])
    phase1_btn.click(fn=run_phase1_eval, inputs=[], outputs=[phase1_box])
    verify_btn.click(fn=run_citation_verification, inputs=[], outputs=[citation_box])
    thumbs_up_btn.click(fn=thumbs_up, inputs=[], outputs=[feedback_status, stats_bar])
    thumbs_down_btn.click(fn=thumbs_down, inputs=[], outputs=[feedback_status, stats_bar])


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=_THEME, css=_CSS, head=_HEAD)
