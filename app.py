
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
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime  # noqa: E402
from app.rag_pipeline import (  # noqa: E402
    build_vector_store,
    make_qa_chain,
    check_input_guard,
    redact_pii,
    evaluate_rag_response,
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
)


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
loaded_docs = {}          # {name: qa_chain}
current_doc_name = None
question_count = 0
last_eval_data = {"question": "", "answer": "", "contexts": [], "sources": []}   # for RAGAS + citation verification buttons


def format_stats():
    doc_info = f"Active: {current_doc_name}" if current_doc_name else "No document loaded"
    cache_info = ""
    feedback_info = ""
    if current_doc_name and current_doc_name in loaded_docs:
        chain = loaded_docs[current_doc_name]
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


def load_document(doc_file):
    global loaded_docs, current_doc_name

    if doc_file is None:
        return "No file uploaded.", gr.update(), format_stats(), ""

    name = os.path.basename(doc_file.name)
    ext = os.path.splitext(name)[1].lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        return (
            f"Unsupported file type '{ext}'. Supported: {supported}",
            gr.update(),
            format_stats(),
            "",
        )

    try:
        tmp = tempfile.mkdtemp()
        store_path = os.path.join(tmp, "vector_store")

        vector_store, all_chunks = build_vector_store(doc_file.name, store_path, doc_id=name)
        qa_chain = make_qa_chain(vector_store, doc_id=name, all_chunks=all_chunks)

        loaded_docs[name] = qa_chain
        current_doc_name = name

        choices = list(loaded_docs.keys())
        status = f"'{name}' loaded. {len(choices)} document(s) ready."
        restored_history = _load_history(name)
        if restored_history:
            status += " (conversation history restored)"

        return status, gr.update(choices=choices, value=name), format_stats(), restored_history

    except Exception as e:
        return f"Failed to load document: {str(e)}", gr.update(), format_stats(), ""


def switch_document(selected_name):
    global current_doc_name
    if selected_name and selected_name in loaded_docs:
        current_doc_name = selected_name
        history = _load_history(selected_name)
        return f"Switched to: '{selected_name}'", history
    return "Document not found.", ""


def clear_cache():
    """Clear the semantic cache for the active document."""
    if current_doc_name and current_doc_name in loaded_docs:
        loaded_docs[current_doc_name].clear_cache()
        return f"Cache cleared for '{current_doc_name}'.", format_stats()
    return "No document loaded.", format_stats()


def ask_question(question, history_text):
    """Generator — streams answer tokens and updates UI in real time."""
    global question_count, last_eval_data

    if not question.strip():
        yield "", history_text, "", "", format_stats()
        return

    # Input Guard
    is_safe, reason = check_input_guard(question)
    if not is_safe:
        blocked_msg = f"[Input Guard] {reason}"
        yield (
            "",
            history_text,
            blocked_msg,
            "[Input Guard] Request blocked before retrieval.",
            format_stats(),
        )
        return

    if current_doc_name is None or current_doc_name not in loaded_docs:
        yield "", history_text, "Please load a document first using Step 1.", "", format_stats()
        return

    qa_chain = loaded_docs[current_doc_name]
    recent_history = history_text[-500:] if len(history_text) > 500 else history_text

    # Show immediate "thinking" state
    t_start = time.time()
    yield "", history_text, "", "Retrieving and reranking...", format_stats()

    answer = ""
    sources_text = ""
    t_retrieval_done = None

    try:
        final_update = None

        for update in qa_chain.stream({"query": question, "history": recent_history}):
            if not update["done"]:
                if t_retrieval_done is None:
                    t_retrieval_done = time.time()  # first token = retrieval done
                answer = strip_emojis(update["result"])
                yield "", history_text, answer, "Generating...", format_stats()
            else:
                final_update = update

        if final_update is None:
            yield "", history_text, "No response generated.", "", format_stats()
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

        # Build sources panel header with all metadata labels
        labels = []
        if cache_hit:
            labels.append("CACHE HIT")
        if grade:
            labels.append(f"CRAG: {grade}")
        if query_type and query_type != "N/A":
            labels.append(f"Type: {query_type}")
        if confidence.get("composite") is not None:
            labels.append(f"Confidence: {confidence['composite']:.2f}")
        if redacted_types:
            labels.append(f"PII Redacted: {', '.join(set(redacted_types))}")

        header = f"[{' | '.join(labels)}]\n\n" if labels else ""

        if sources:
            sources_text = f"{header}Source chunks from '{current_doc_name}':\n\n"
            for s in sources:
                clean_content = strip_emojis(s["content"])
                score_str = f"  -  Confidence: {s['score']}" if s.get("score") != "N/A" else ""
                method = s.get("method", "")
                method_str = f"  -  [{method.upper()}]" if method else ""
                sources_text += (
                    f"[Source {s['chunk']}  -  Page {s['page']}{score_str}{method_str}]\n"
                    f"{clean_content}...\n\n{'- ' * 25}\n\n"
                )
        elif grade == "INCORRECT":
            sources_text = f"{header}No relevant chunks found in document."
        else:
            sources_text = f"{header}No source chunks returned." if header else "No source chunks returned."

        # Phase 3 — append latency breakdown to sources panel
        t_end = time.time()
        total_ms = int((t_end - t_start) * 1000)

        if cache_hit:
            # Semantic cache hit — served without a new LLM call, so TTFT/tokens/sec
            # don't apply. Say so explicitly instead of showing misleading zeros.
            sources_text += (
                f"\n\n{'- ' * 25}\n"
                f"Latency — Served from semantic cache: {total_ms}ms (no new generation)"
            )
        else:
            ttft_ms = int((t_retrieval_done - t_start) * 1000) if t_retrieval_done else 0
            generation_ms = total_ms - ttft_ms

            generated_tokens = (final_update or {}).get("generated_tokens", 0)
            cached_tokens = (final_update or {}).get("cached_tokens", 0)
            tokens_per_sec = round(generated_tokens / (generation_ms / 1000), 1) if generation_ms > 0 and generated_tokens else 0.0
            cache_str = f"  |  Prompt cache: {cached_tokens} tok" if cached_tokens else ""

            sources_text += (
                f"\n\n{'- ' * 25}\n"
                f"Latency — TTFT (incl. retrieval+rerank): {ttft_ms}ms  |  "
                f"Generation: {generation_ms}ms  |  Total: {total_ms}ms\n"
                f"{tokens_per_sec} tok/s  |  {generated_tokens} tokens generated{cache_str}"
            )

    except Exception as e:
        answer = f"Error: {str(e)}"
        sources_text = ""

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
    if final_update and final_update.get("sources"):
        last_eval_data["question"] = question
        last_eval_data["answer"] = answer
        last_eval_data["contexts"] = [s["content"] for s in final_update["sources"]]
        last_eval_data["sources"] = final_update["sources"]

    # Persist conversation history to Redis
    _save_history(current_doc_name, updated_history)

    yield "", updated_history, answer, sources_text, format_stats()


def clear_history():
    _delete_history(current_doc_name)
    return "", "", ""


def run_phase1_eval():
    """
    Phase 1 Retrieval Evaluation.
    Generates a synthetic test set from the active document's chunks,
    then computes Precision@K, Recall@K, MRR, and Coverage.
    Expects ~10-20 seconds (one LLM call per question generated).
    """
    if current_doc_name is None or current_doc_name not in loaded_docs:
        return "Load a document first."

    chain = loaded_docs[current_doc_name]
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

/* ── Answer box ── */
#answer-box textarea {
    color: #3D1A06 !important;
    font-size: 0.95em !important;
    line-height: 1.78 !important;
    background: #FFFEF8 !important;
    border-color: #DBB06B !important;
    border-left: 3px solid #DBB06B !important;
}

/* ── Sources box ── */
#sources-box textarea {
    color: #7A4020 !important;
    font-size: 0.81em !important;
    font-family: 'Courier New', monospace !important;
    line-height: 1.58 !important;
    background: #FFF5EE !important;
    border-color: #FFD3AC !important;
    border-left: 3px solid #FFB5AB !important;
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

/* ── Tabs ── */
.tabs > .tab-nav {
    border-bottom: 2px solid #FFD3AC !important;
    background: transparent !important;
    margin-bottom: 16px !important;
}
.tabs > .tab-nav > button {
    color: #C8A080 !important;
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
.tabs > .tab-nav > button.selected {
    color: #E39A7B !important;
    border-bottom-color: #E39A7B !important;
}
.tabs > .tab-nav > button:hover { color: #DBB06B !important; }

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

    stats_bar = gr.Textbox(
        value=format_stats(),
        label="Session Stats",
        interactive=False,
        lines=1,
        elem_id="stats-bar",
    )

    # ── Step 1 — Document Upload ─────────────────────────────────
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

    # ── Step 2 — Chat ────────────────────────────────────────────
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

    # ── Answer & Sources ─────────────────────────────────────────
    _section_header("LAST ANSWER & SOURCE CHUNKS", "check")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            last_answer_box = gr.Textbox(
                label="Generated Answer  [streams in real time]",
                value="",
                interactive=True,
                lines=10,
                elem_id="answer-box",
            )
        with gr.Column(scale=1):
            sources_box = gr.Textbox(
                label="Retrieved Chunks  [CRAG grade · Query type · Confidence · Latency]",
                value="",
                interactive=False,
                lines=10,
                elem_id="sources-box",
            )

    gr.HTML("<hr/>")

    # ── Feedback ─────────────────────────────────────────────────
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

    # ── Step 3 — Evaluation ──────────────────────────────────────
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

    # ── Event wiring ─────────────────────────────────────────────
    load_btn.click(
        fn=load_document,
        inputs=doc_input,
        outputs=[load_status, doc_selector, stats_bar, conversation],
    )
    doc_selector.change(
        fn=switch_document,
        inputs=doc_selector,
        outputs=[load_status, conversation],
    )
    ask_btn.click(
        fn=ask_question,
        inputs=[question_box, conversation],
        outputs=[question_box, conversation, last_answer_box, sources_box, stats_bar],
    )
    question_box.submit(
        fn=ask_question,
        inputs=[question_box, conversation],
        outputs=[question_box, conversation, last_answer_box, sources_box, stats_bar],
    )
    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[conversation, last_answer_box, sources_box],
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
