---
title: Rag Chatbot
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
---

# RAG Chatbot — Chat with Any Document

A production-grade Retrieval-Augmented Generation chatbot. Upload any document and ask questions about it. Built with LangChain, Groq/OpenAI LLMs, FAISS, Qdrant, and Gradio.

**Live Demo**: [huggingface.co/spaces/Prav04/rag-chatbot](https://huggingface.co/spaces/Prav04/rag-chatbot)

---

## Features

**Document support**
- PDF (text, table-aware, and scanned/image pages via OCR — adaptive per page)
- DOCX, HTML, TXT, Markdown
- Multiple documents per session — load several and switch between them

**Retrieval pipeline**
- Hybrid BM25 + FAISS dense retrieval with Reciprocal Rank Fusion
- Contextual Retrieval (optional) — LLM-generated situating context prepended to each chunk before embedding, improves recall on chunks that read as ambiguous in isolation
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- CRAG grading — out-of-scope questions are blocked before reaching the LLM
- Query condensation — follow-up questions rewritten as standalone before retrieval
- Query decomposition — complex multi-part questions split into focused sub-queries
- HyDE (Hypothetical Document Embeddings) — optional, improves conceptual recall
- Query routing — FACTUAL / CONCEPTUAL / COMPARATIVE prompts selected automatically (CONCEPTUAL may add one original, clearly-labeled analogy beyond the source text)

**Vector store**
- Qdrant Cloud when internet is available (persistent, collection-based)
- FAISS local fallback when offline (no setup needed)

**Caching**
- Semantic cache backed by Redis (persistent across restarts)
- In-memory LRU fallback when Redis is unavailable
- Cosine similarity threshold — similar questions served from cache instantly

**Safety and quality**
- Input guard — blocks prompt injection, jailbreaks, and harmful content
- PII redaction — removes emails, phone numbers, SSNs, credit card numbers from answers
- Strict prompting — LLM prohibited from using general knowledge; refuses off-topic questions

**Evaluation**
- Phase 1 — Retrieval Evaluation: synthetic question generation, Precision@K, Recall@K, MRR, Coverage. `evaluate_config.py` runs this against a saved, fixed test set so you can compare configs (models, embeddings, contextual retrieval on/off) on equal footing instead of regenerating questions each run.
- Phase 2 — RAGAS: Faithfulness, Answer Relevancy, Context Precision (LLM-as-judge via your configured Groq or OpenAI model)
- Phase 3 — Human Feedback: thumbs up/down recorded to Redis with success rate tracking

**Observability**
- LangSmith tracing — all LangChain calls auto-traced when `LANGCHAIN_API_KEY` is set
- Latency breakdown per answer — TTFT (retrieval+rerank), generation time, tokens/sec, and OpenAI prompt-cache hit tokens when applicable; cache-served answers are labeled separately since no new generation occurs

---

## Tech Stack

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| LLM              | Groq (recommended, fast + free tier) / OpenAI (gpt-5-mini tested) / local FLAN-T5 |
| Embeddings       | OpenAI text-embedding-3-small (when `OPENAI_API_KEY` set) / sentence-transformers/all-MiniLM-L6-v2 (local fallback) |
| Reranker         | cross-encoder/ms-marco-MiniLM-L-6-v2            |
| Vector Store     | Qdrant Cloud (online) / FAISS (offline fallback)|
| Sparse Retrieval | BM25 (rank-bm25)                                |
| RAG Framework    | LangChain                                       |
| Cache            | Redis / in-memory LRU                           |
| PDF Extraction   | pdfplumber (text + tables) / pytesseract (OCR)  |
| UI               | Gradio                                          |
| Evaluation       | RAGAS                                           |
| Observability    | LangSmith                                       |

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

OCR support requires Tesseract (for scanned PDF pages):
```bash
brew install tesseract        # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

**4. Set up environment variables**
```bash
cp env.example .env
```

Open `.env` and fill in at minimum:
```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — 14,400 requests/day, no credit card.

**5. Run the app**
```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## LLM Options

Set in `.env`. Priority order: Groq > OpenAI > local FLAN-T5.

| Option          | Speed       | Cost             | Config key        |
|-----------------|-------------|------------------|-------------------|
| Groq (default)  | ~1 second   | Free (14k/day)   | `GROQ_API_KEY`    |
| OpenAI (gpt-5-mini) | ~2 seconds | ~$0.001-0.002/query ($0.25/1M input, $2/1M output tokens) | `OPENAI_API_KEY` |
| Local FLAN-T5   | ~15 seconds | Free, no key     | (no key needed)   |

Check current model availability before deploying — both providers deprecate models periodically ([Groq](https://console.groq.com/docs/deprecations), [OpenAI](https://developers.openai.com/api/docs/deprecations)). `gpt-5.x` models don't accept the `temperature` parameter (handled automatically in `get_llm()`).

---

## Vector Store Options

Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` to use Qdrant Cloud.
Leave them empty to use FAISS local storage (default, no setup needed).

The app detects internet connectivity at startup and picks the backend automatically.

Get a free Qdrant cluster (no credit card) at [cloud.qdrant.io](https://cloud.qdrant.io).

---

## Project Structure

```
rag-chatbot/
├── app.py                      # Gradio UI — event wiring, streaming, feedback
├── app/
│   ├── __init__.py             # Package re-exports
│   ├── config.py               # All env vars with typed defaults (Config class)
│   ├── guards.py               # Input guard, PII redaction
│   ├── document_loader.py      # Adaptive PDF extraction, FAISS/Qdrant backends
│   ├── evaluation.py           # Phase 1 retrieval eval + RAGAS Phase 2
│   └── rag_pipeline.py         # Embeddings, LLM, reranker, cache, QA chain
├── run_me_once.py              # CLI to pre-build a vector store from a file
├── evaluate_config.py          # CLI to compare configs (models/embeddings) on a fixed test set
├── env.example                 # All configurable settings with descriptions
└── requirements.txt
```

---

## How It Works

```
User uploads document
         |
         v
Adaptive extraction per page:
  scanned page  -> OCR (pytesseract)
  table page    -> pdfplumber table mode
  text page     -> pdfplumber text mode
         |
         v
Text split into 500-char chunks (50 overlap)
         |
         v
(optional) Contextual Retrieval: LLM writes a situating
sentence per chunk, prepended before embedding
         |
         v
Chunks embedded (OpenAI text-embedding-3-small, or
sentence-transformers locally)
         |
         v
Embeddings stored in Qdrant Cloud (or FAISS offline)
  - unchanged doc + same embedding config -> reused, skips re-embedding
BM25 sparse index built in memory
         |
         v
User asks a question
         |
         v
Follow-up condensed to standalone question
Complex questions decomposed into sub-queries
         |
         v
Hybrid retrieval: BM25 + FAISS/Qdrant dense -> RRF fusion
Cross-encoder reranker scores all candidates
CRAG grader: CORRECT / AMBIGUOUS / INCORRECT
  AMBIGUOUS or INCORRECT -> refuse immediately (no LLM call)
         |
         v
Query routed: FACTUAL / CONCEPTUAL / COMPARATIVE
Top chunks + question sent to the configured LLM (Groq/OpenAI)
         |
         v
Answer streamed token-by-token to UI
PII redacted from output
Stored in semantic cache (Redis)
```

---

## Configuration Reference

All settings are in `env.example`. Key knobs:

| Variable                   | Default                          | Description                               |
|----------------------------|----------------------------------|-------------------------------------------|
| `GROQ_API_KEY`             | —                                | Groq API key (recommended LLM backend)    |
| `OPENAI_MODEL`             | gpt-3.5-turbo (code default; use gpt-5-mini) | OpenAI chat model — set explicitly, don't rely on the code default |
| `OPENAI_EMBEDDING_MODEL`   | text-embedding-3-large           | OpenAI embedding model (used when `OPENAI_API_KEY` is set) |
| `QDRANT_URL`               | —                                | Qdrant Cloud URL (leave empty for FAISS)  |
| `QDRANT_API_KEY`           | —                                | Qdrant Cloud API key                      |
| `RETRIEVAL_K`              | 3                                | Final chunks passed to LLM after rerank   |
| `RETRIEVAL_K_INITIAL`      | 10                               | Candidates fetched before reranking       |
| `GRADE_CORRECT_THRESHOLD`  | -2.0                             | Min reranker score to answer              |
| `GRADE_AMBIGUOUS_THRESHOLD`| -5.0                             | Min score to even pass to LLM             |
| `SEMANTIC_CACHE_THRESHOLD` | 0.92                             | Cosine similarity for cache hit           |
| `CONDENSE_QUESTIONS`       | true                             | Rewrite follow-ups before retrieval       |
| `DECOMPOSE_QUERIES`        | true                             | Split complex questions into sub-queries  |
| `HYDE_ENABLED`             | false                            | Hypothetical document embedding           |
| `CONTEXTUAL_RETRIEVAL`     | false                            | LLM-generated context per chunk before embedding |
| `LLM_MAX_OUTPUT_TOKENS`    | 800                              | max_tokens cap for Groq/OpenAI generation |
| `RAGAS_EVAL`               | false                            | Enable Phase 2 RAGAS evaluation button    |
| `LANGCHAIN_TRACING_V2`     | false                            | Enable LangSmith tracing                  |
| `REDIS_URL`                | redis://localhost:6379           | Redis for semantic cache + history        |

Note: `OPENAI_MODEL`'s code-level default (`app/config.py`) is still `gpt-3.5-turbo` for backward compatibility, but that model is being retired by OpenAI on **October 23, 2026** — always set `OPENAI_MODEL` explicitly in `.env` for new setups.

---

## Deployment on HuggingFace Spaces

1. Fork this repo on GitHub
2. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) (SDK: Gradio)
3. Connect your GitHub repo to the Space
4. Add secrets in Space Settings > Variables and secrets:
   - `GROQ_API_KEY` or `OPENAI_API_KEY` (at least one required — set `OPENAI_MODEL`/`GROQ_MODEL` explicitly, see LLM Options above)
   - `QDRANT_URL` and `QDRANT_API_KEY` (optional — for persistent vector storage)
   - `REDIS_URL` (optional — for persistent cache and history)

---

## Author

**Pravalli** — AIML Engineer  
[GitHub](https://github.com/Prav-allika) · [HuggingFace](https://huggingface.co/Prav04)
