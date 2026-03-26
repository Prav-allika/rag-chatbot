import gradio as gr
import os
import re
import tempfile
from datetime import datetime
from app.rag_pipeline import build_vector_store, make_qa_chain


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
loaded_pdfs = {}
current_pdf_name = None
question_count = 0


def format_stats():
    pdf_info = f"Active: {current_pdf_name}" if current_pdf_name else "No PDF loaded"
    return f"Questions asked: {question_count}  |  {pdf_info}  |  PDFs loaded: {len(loaded_pdfs)}"


def load_pdf(pdf_file):
    global loaded_pdfs, current_pdf_name

    if pdf_file is None:
        return "No file uploaded.", gr.update(), format_stats()

    try:
        tmp = tempfile.mkdtemp()
        store_path = os.path.join(tmp, "vector_store")

        vector_store = build_vector_store(pdf_file.name, store_path)
        qa_chain = make_qa_chain(vector_store)

        name = os.path.basename(pdf_file.name)
        loaded_pdfs[name] = qa_chain
        current_pdf_name = name

        choices = list(loaded_pdfs.keys())
        status = f"'{name}' loaded. {len(choices)} PDF(s) ready."

        return status, gr.update(choices=choices, value=name), format_stats()

    except Exception as e:
        return f"Failed to load PDF: {str(e)}", gr.update(), format_stats()


def switch_pdf(selected_name):
    global current_pdf_name
    if selected_name and selected_name in loaded_pdfs:
        current_pdf_name = selected_name
        return f"Switched to: '{selected_name}'"
    return "PDF not found."


def ask_question(question, history_text):
    global question_count

    if not question.strip():
        return "", history_text, "", "", format_stats()

    if current_pdf_name is None or current_pdf_name not in loaded_pdfs:
        answer = "Please load a PDF first using Step 1."
        sources_text = ""
        last_ans = answer
    else:
        qa_chain = loaded_pdfs[current_pdf_name]
        recent_history = (
            history_text[-500:] if len(history_text) > 500 else history_text
        )

        try:
            result = qa_chain.invoke({"query": question, "history": recent_history})
            answer = strip_emojis(result["result"])
            sources = result.get("sources", [])

            if sources:
                sources_text = f"Source chunks from '{current_pdf_name}':\n\n"
                for s in sources:
                    clean_content = strip_emojis(s["content"])
                    sources_text += f"[Chunk {s['chunk']}  -  Page {s['page']}]\n{clean_content}...\n\n{'- ' * 25}\n\n"
            else:
                sources_text = "No source chunks returned."

        except Exception as e:
            answer = f"Error: {str(e)}"
            sources_text = ""

        last_ans = answer

    question_count += 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    separator = ". " * 30
    clean_question = strip_emojis(question)
    new_entry = f"[{timestamp}]  You: {clean_question}\n\nAssistant: {answer}\n\n{separator}\n\n"
    updated_history = history_text + new_entry

    return "", updated_history, last_ans, sources_text, format_stats()


def clear_history():
    return "", "", ""


# ---------- UI ----------
with gr.Blocks(title="RAG Chatbot - Chat with Any PDF") as demo:
    gr.Markdown("# RAG Chatbot - Chat with Any PDF")
    gr.Markdown(
        "Upload one or more PDFs, switch between them, and ask questions. "
        "Powered by LangChain, FAISS, and Groq LLaMA 3.3."
    )

    stats_bar = gr.Textbox(
        value=format_stats(),
        label="Session Stats",
        interactive=False,
        lines=1,
    )

    gr.Markdown("---")

    gr.Markdown("### Step 1: Upload PDFs")
    gr.Markdown(
        "You can load multiple PDFs and switch between them using the dropdown."
    )

    with gr.Row():
        pdf_input = gr.File(label="PDF Document", file_types=[".pdf"])
        load_status = gr.Textbox(
            label="Status",
            value="No PDF loaded yet.",
            interactive=False,
            lines=2,
        )

    with gr.Row():
        load_btn = gr.Button("Load PDF", variant="primary", scale=1)
        pdf_selector = gr.Dropdown(
            label="Switch active PDF",
            choices=[],
            interactive=True,
            scale=2,
        )

    gr.Markdown("---")

    gr.Markdown("### Step 2: Ask questions")

    conversation = gr.Textbox(
        label="Conversation history",
        value="",
        interactive=False,
        lines=14,
        max_lines=14,
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="Type your question and press Enter or click Ask...",
            lines=2,
            scale=5,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    clear_btn = gr.Button("Clear conversation", size="sm")

    gr.Markdown("---")

    gr.Markdown("### Last Answer & Sources")

    with gr.Row():
        with gr.Column(scale=1):
            last_answer_box = gr.Textbox(
                label="Last Answer - click inside to select all and copy",
                value="",
                interactive=True,
                lines=8,
            )
        with gr.Column(scale=1):
            sources_box = gr.Textbox(
                label="Source Chunks - paragraphs used to generate the answer",
                value="",
                interactive=False,
                lines=8,
            )

    gr.Markdown("---")
    gr.Markdown(
        "Built with [LangChain](https://langchain.com) · "
        "[FAISS](https://github.com/facebookresearch/faiss) · "
        "[Gradio](https://gradio.app)"
    )

    load_btn.click(
        fn=load_pdf,
        inputs=pdf_input,
        outputs=[load_status, pdf_selector, stats_bar],
    )

    pdf_selector.change(
        fn=switch_pdf,
        inputs=pdf_selector,
        outputs=load_status,
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

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
