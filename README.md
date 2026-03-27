# RAG Chatbot — Chat with Any PDF

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, FAISS, Groq LLaMA 3.3, and Gradio.

**Live Demo**: [huggingface.co/spaces/Prav04/rag-chatbot](https://huggingface.co/spaces/Prav04/rag-chatbot)

---

## Features

- **Upload any PDF** — not locked to one document
- **Multi-PDF support** — load multiple PDFs and switch between them mid-session
- **Fast responses** — powered by Groq LLaMA 3.3 (70B), ~1 second per answer
- **Chat history** — full conversation log with timestamps
- **Follow-up questions** — context-aware answers using recent conversation history
- **Source chunks** — see exactly which paragraphs from the PDF were used to generate the answer
- **Session stats** — live question counter and active PDF tracker
- **Copy last answer** — click inside the answer box to select and copy

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq LLaMA 3.3 70B (or OpenAI GPT-3.5) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| RAG Framework | LangChain (LCEL pattern) |
| UI | Gradio |
| API | FastAPI |

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/Prav-allika/rag-chatbot.git
cd rag-chatbot
```

**2. Create a virtual environment**
```bash
conda create -n rag-chatbot python=3.10 -y
conda activate rag-chatbot
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — 14,400 requests/day free.

**5. Run the app**
```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## LLM Options

The app supports three LLM backends — set in `.env`:

| Option | Speed | Cost | Setup |
|---|---|---|---|
| Groq (recommended) | ~1 second | Free (14k req/day) | `GROQ_API_KEY=...` |
| OpenAI | ~2 seconds | ~$0.002/query | `OPENAI_API_KEY=...` |
| Local FLAN-T5 | ~15 seconds | Free | No key needed |

---

## Project Structure

```
rag-chatbot/
├── app.py                  # Gradio UI with all features
├── app/
│   ├── rag_pipeline.py     # Embeddings, FAISS, QA chain, Groq LLM
│   └── main.py             # FastAPI backend
├── run_me_once.py          # Build vector store from PDF
├── .env.example            # Environment variable template
└── requirements.txt
```

---

## How It Works

```
User uploads PDF
      |
      v
PDF split into chunks (500 chars, 50 overlap)
      |
      v
Chunks embedded using sentence-transformers
      |
      v
Embeddings stored in FAISS vector store
      |
      v
User asks a question
      |
      v
Question embedded → top 3 similar chunks retrieved
      |
      v
Chunks + question sent to Groq LLaMA 3.3
      |
      v
Answer + source chunks returned to UI
```

---

## Deployment on HuggingFace Spaces

1. Fork this repo
2. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in Settings → Variables and secrets

---

## Author

**Pravalli** — ML Engineer  
[GitHub](https://github.com/Prav-allika) · [HuggingFace](https://huggingface.co/Prav04)