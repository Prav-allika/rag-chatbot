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
- PDF (text, table-aware, and scanned/image pages via OCR — adaptive per page); `[TABLE]` blocks and informally-captioned tables (no visible ruling lines) are detected and kept atomic, never split mid-table across chunks
- DOCX, HTML, TXT, Markdown
- Multiple documents per session — load several and switch between them
- Three switchable chunking strategies (`CHUNKING_STRATEGY`: fixed / structure / semantic) — see [Chunking Strategy Comparison](#chunking-strategy-comparison)
- Near-duplicate dedup — chunks with cosine similarity above `NEAR_DUP_THRESHOLD` to one already kept are dropped, on top of exact-hash dedup

**Retrieval pipeline**
- Hybrid BM25 + FAISS dense retrieval with configurably-weighted Reciprocal Rank Fusion (`RRF_DENSE_WEIGHT`)
- Contextual Retrieval (optional) — LLM-generated situating context prepended to each chunk before embedding, improves recall on chunks that read as ambiguous in isolation
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- CRAG grading — out-of-scope questions are blocked before reaching the LLM
- Query condensation — follow-up questions rewritten as standalone before retrieval
- Query decomposition — complex multi-part questions split into focused sub-queries, each reranked against itself and interleaved before the final cut (so one sub-topic's chunks can't crowd out another's)
- HyDE (Hypothetical Document Embeddings) — optional, improves conceptual recall
- Query routing — FACTUAL / CONCEPTUAL / COMPARATIVE prompts selected automatically (CONCEPTUAL may add one original, clearly-labeled analogy beyond the source text); all three permit connecting multiple explicitly-stated facts across source chunks (synthesis), while still prohibiting invented facts

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
- Composite confidence score — retrieval confidence (reranker score) + citation coverage + answer completeness, shown next to every answer (free, no extra LLM calls)

**Evaluation**
- Phase 1 — Retrieval Evaluation: synthetic question generation, Precision@K, Recall@K, MRR, Coverage. `evaluate_config.py` runs this against a saved, fixed test set so you can compare configs (models, embeddings, contextual retrieval on/off) on equal footing instead of regenerating questions each run.
- Phase 2 — RAGAS: Faithfulness, Answer Relevancy, Context Precision (LLM-as-judge via your configured Groq or OpenAI model)
- Phase 3 — Human Feedback: thumbs up/down recorded to Redis with success rate tracking
- Phase 4 — Citation Verification: parses each answer's superscript citations (¹²³) and checks, per cited sentence, whether the source chunk it points to actually supports the claim (LLM-as-judge) — flags unsupported citations and claims with no citation at all, instead of trusting that a citation number means the claim is grounded
- Golden Eval Suite — 50 hand-curated questions against `data/Attention.pdf` (`artifacts/eval/golden_testset.json`) covering straightforward lookups, multi-hop questions, no-answer-in-corpus, and ambiguous questions. `run_golden_eval.py` scores each on retrieval confidence, citation accuracy, and LLM-judged correctness (for no-answer questions, correctness means "did it refuse"). Strategy-neutral by construction — unlike the synthetic test set, it isn't generated from any one chunking strategy's own chunks

**Observability**
- LangSmith tracing — all LangChain calls auto-traced when `LANGCHAIN_API_KEY` is set
- Latency breakdown per answer — TTFT (retrieval+rerank), generation time, tokens/sec, and OpenAI prompt-cache hit tokens when applicable; cache-served answers are labeled separately since no new generation occurs

---

## Tech Stack

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| LLM              | OpenAI (gpt-5-mini tested, default) / Groq (fast + free tier fallback) / local FLAN-T5 |
| Embeddings       | OpenAI text-embedding-3-small (when `OPENAI_API_KEY` set) / sentence-transformers/all-MiniLM-L6-v2 (local fallback) |
| Reranker         | cross-encoder/ms-marco-MiniLM-L-6-v2            |
| Vector Store     | Qdrant Cloud (online) / FAISS (offline fallback)|
| Sparse Retrieval | BM25 (rank-bm25)                                |
| RAG Framework    | LangChain                                       |
| Cache            | Redis / in-memory LRU                           |
| PDF Extraction   | pdfplumber (text + tables) / pytesseract (OCR)  |
| UI               | Gradio (primary) / Streamlit (interview demo dashboard) |
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

Set in `.env`. Priority order: OpenAI > Groq > local FLAN-T5.

| Option          | Speed       | Cost             | Config key        |
|-----------------|-------------|------------------|-------------------|
| OpenAI (gpt-5-mini) | ~2 seconds | ~$0.001-0.002/query ($0.25/1M input, $2/1M output tokens) | `OPENAI_API_KEY` |
| Groq            | ~1 second   | Free (14k/day)   | `GROQ_API_KEY`    |
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
│   ├── document_loader.py      # Adaptive PDF extraction, chunking dispatch, FAISS/Qdrant backends
│   ├── chunking.py             # Switchable chunking strategies (fixed/structure/semantic), atomic tables
│   ├── citations.py            # Superscript citation parsing + LLM-as-judge verification
│   ├── confidence.py           # Composite confidence score (retrieval + citation coverage + completeness)
│   ├── evaluation.py           # Phase 1 retrieval eval + RAGAS Phase 2
│   ├── rag_pipeline.py         # Embeddings, LLM, reranker, hybrid retrieval, cache, QA chain
│   └── main.py                 # FastAPI app — multi-document ingest/list/ask
├── streamlit_app.py             # Interview demo dashboard — confidence breakdown, clickable citations, hybrid-vs-dense comparison, golden eval results
├── run_me_once.py              # CLI to pre-build a vector store from a file
├── evaluate_config.py          # CLI to compare configs (models/embeddings) on a fixed test set
├── compare_chunking_strategies.py  # CLI to compare chunking strategies (synthetic or golden test set)
├── run_golden_eval.py          # CLI to run the 50-question golden eval suite
├── artifacts/eval/golden_testset.json  # Hand-curated golden Q&A set
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
  table page    -> pdfplumber table mode (x_tolerance tuned to avoid word-fusion)
  text page     -> pdfplumber text mode (+ informal "Table N:" caption detection)
         |
         v
Chunked (fixed/structure/semantic — CHUNKING_STRATEGY)
  [TABLE] blocks kept atomic — never split mid-table
         |
         v
Exact-hash dedup, then near-duplicate dedup (cosine similarity > NEAR_DUP_THRESHOLD)
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
  -> each sub-query retrieved + reranked against ITSELF, then interleaved
     (avoids one sub-topic's chunks crowding out another's)
         |
         v
Hybrid retrieval: BM25 + FAISS/Qdrant dense -> weighted RRF fusion (RRF_DENSE_WEIGHT)
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
Composite confidence score computed (retrieval + citation coverage + completeness — free, no extra LLM calls)
PII redacted from output
Stored in semantic cache (Redis)
         |
         v
(on demand) Citation Verification — each superscript checked
against its source chunk by an LLM judge (Phase 4 tab)
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
| `CHUNKING_STRATEGY`        | fixed                            | `fixed` \| `structure` \| `semantic` — see [Chunking Strategy Comparison](#chunking-strategy-comparison) |
| `NEAR_DUP_DEDUP_ENABLED`   | true                              | Drop chunks with cosine similarity > `NEAR_DUP_THRESHOLD` to one already kept |
| `NEAR_DUP_THRESHOLD`       | 0.95                             | Similarity threshold for near-duplicate dedup |
| `CONFIDENCE_WEIGHT_RETRIEVAL` / `_COVERAGE` / `_COMPLETENESS` | 0.5 / 0.3 / 0.2 | Weights for the composite confidence score (auto-normalized) |
| `RRF_DENSE_WEIGHT`         | 0.5                              | Dense weight in RRF fusion (sparse = 1 - this); 0.5 = unweighted |
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

## Chunking Strategy Comparison

Three switchable chunking strategies (`app/chunking.py`), selected via `CHUNKING_STRATEGY` in `.env`:

| Strategy    | How it splits                                                                 |
|-------------|--------------------------------------------------------------------------------|
| `fixed`     | `RecursiveCharacterTextSplitter`, fixed size + overlap (default, baseline)     |
| `structure` | Splits on detected section headers first (Markdown ATX, numbered sections, ALL-CAPS lines), recursive-splits within oversized sections |
| `semantic`  | Splits on topic boundaries — cosine distance between consecutive sentence embeddings, cut wherever the jump exceeds a percentile threshold |

Every chunk is tagged with `chunking_strategy`, `chunk_index`, and `char_count` in its metadata regardless of which strategy produced it.

Compare all three on your own document with a fixed golden test set, reusing the same retrieval-eval metrics as `evaluate_config.py` (Precision@K, Recall@K, MRR, Coverage):

```bash
# 1. Once — generate and save a fixed question set from your document
python compare_chunking_strategies.py --file data/Attention.pdf --generate-testset --n 12

# 2. Rerun any time for an apples-to-apples comparison against the same test set
python compare_chunking_strategies.py --file data/Attention.pdf
```

Writes `artifacts/eval/chunking_comparison.md` (a markdown report with a per-metric winner) and appends a record to `artifacts/eval/chunking_comparison.jsonl`. `semantic` costs one extra embedding call per sentence at build time — noticeably slower to index than `fixed`/`structure`; the report includes build time so you can weigh that against any retrieval-quality gain.

Add `--golden` to compare against the hand-curated golden set instead — see below.

---

## Golden Eval Suite

`artifacts/eval/golden_testset.json` — 50 hand-written questions against `data/Attention.pdf`, covering:

| Category | Count | Tests |
|---|---|---|
| `lookup` | 20 | Single-fact questions answerable from one section |
| `multi_hop` | 15 | Require combining information from two sections |
| `no_answer` | 10 | Plausible-sounding questions the paper doesn't cover — correct behavior is to refuse |
| `ambiguous` | 5 | Underspecified questions — correct behavior is to hedge/ask for clarification, not confidently guess |

```bash
python run_golden_eval.py --n 5                  # cheap smoke test on 5 questions first
python run_golden_eval.py                        # full 50-question run
python run_golden_eval.py --category multi_hop    # re-test just one category (cheap targeted re-check)
python run_golden_eval.py --ragas                 # + RAGAS faithfulness (needs RAGAS_EVAL=true)
```

Each question is scored on retrieval confidence (free, from the composite confidence score), citation accuracy (LLM-as-judge, reusing the Phase 4 verifier), and LLM-judged correctness against the golden `expected_answer` — for `no_answer` questions, correctness means "did it correctly decline." Writes `artifacts/eval/golden_eval_results.jsonl` and a per-category `artifacts/eval/golden_eval_summary.md`. This is a real cost run — one invoke, one citation-verification pass, and one correctness judgment per question — start with `--n 5` before running the full set.

**Results** (gpt-5-mini, text-embedding-3-small, full 50-question run):

| Category | N | Correctness |
|---|---|---|
| Lookup | 20 | 90.0% |
| Multi-hop | 15 | 66.7% |
| No-answer | 10 | 100% |
| Ambiguous | 5 | 80.0% |
| **Overall** | **50** | **84.0%** |

A first baseline run scored 72.0% overall (multi-hop at 46.7% specifically) before four targeted fixes: a prompt rule that was blocking legitimate cross-passage synthesis, per-subquery retrieval reranking, a PDF text-extraction tolerance bug that was fusing words together (`"Table1:Maximumpathlengths"` → `"Table 1: Maximum path lengths"`), and keeping `[TABLE]` blocks atomic during chunking so a table's rows can't be split across chunk boundaries. Remaining failures are individually diagnosed rather than unexplained — mainly a geometric table-layout limitation (row-group labels that are vertically centered across several rows in the source PDF get attributed to the wrong row when flattened to text) and ordinary LLM answer-completeness variance, not retrieval or extraction bugs.

---

## API (FastAPI)

`app/main.py` exposes a multi-document REST API (separate from the Gradio UI, same underlying pipeline):

```bash
uvicorn app.main:app --reload
# Visit http://localhost:8000/docs for interactive OpenAPI docs
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check + count of loaded documents |
| `/documents` | GET | List indexed documents (doc_id, chunk count, load time) |
| `/documents/ingest` | POST | Upload + index a new document (multipart file) — returns its `doc_id` |
| `/ask` | POST | `{"question": "...", "doc_id": "..."}` — `doc_id` optional, defaults to the startup document. Returns the answer with `sources` and a composite `confidence` score |
| `/metrics` | GET | Basic metrics (documents loaded, doc_ids) |

At startup the API auto-ingests `data/Attention.pdf` (downloading it if missing) under `doc_id="Attention.pdf"`, so `/ask` works out of the box with no `doc_id` — existing integrations aren't affected by adding multi-document support.

---

## Interview Demo Dashboard (Streamlit)

`streamlit_app.py` — a separate, focused dashboard built to surface the pipeline's differentiators visually, rather than reusing the Gradio UI's document-management-heavy layout. Not a replacement for `app.py` or `app/main.py`; same underlying pipeline and same visual palette as the Gradio app.

```bash
streamlit run streamlit_app.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502). Two tabs:

- **PDF Upload** — upload or select a loaded document, ask a question, and see:
  - The generated answer with inline superscript citations
  - Composite confidence broken into its three parts (retrieval / citation coverage / completeness), each as its own bar — not just one number
  - Clickable "Source N" buttons that expand the exact chunk (page, rerank score) each citation points to
  - A side-by-side **hybrid vs. dense-only** retrieval comparison, so the value of BM25+RRF fusion is visible per-question rather than asserted
- **Golden Set** — the 50-question golden eval results (see [Golden Eval Suite](#golden-eval-suite)) rendered as a scorecard, read from the last full run in `artifacts/eval/golden_eval_results.jsonl`

Each uploaded document gets its own isolated Qdrant collection (or FAISS index offline) — switching the "Active document" dropdown switches the whole retrieval chain, not just the display.

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
