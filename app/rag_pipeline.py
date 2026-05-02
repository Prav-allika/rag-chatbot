"""
app/rag_pipeline.py — core RAG pipeline.

Orchestrates:
  1. LangSmith tracing setup
  2. Model loading (embeddings, LLM, reranker) — cached
  3. Semantic cache (Redis + in-memory fallback)
  4. Query pre-processing (condense, decompose, HyDE)
  5. Hybrid BM25+FAISS retrieval + RRF fusion
  6. Cross-encoder reranking + CRAG grading
  7. Query routing (FACTUAL / CONCEPTUAL / COMPARATIVE)
  8. Token streaming

Delegates to:
  app.config          — environment configuration (Config)
  app.guards          — check_input_guard, redact_pii
  app.document_loader — document loading, PDF pipeline, vector store
  app.evaluation      — retrieval eval (Phase 1) + RAGAS (Phase 2)
"""

import os
import re
import json
import hashlib
import logging
import threading

import numpy as np
from functools import lru_cache
from collections import OrderedDict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import Config
from app.guards import check_input_guard, redact_pii
from app.document_loader import build_vector_store, load_vector_store, _SUPPORTED_EXTENSIONS
from app.evaluation import (
    generate_retrieval_test_set,
    evaluate_retrieval_metrics,
    evaluate_rag_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LANGSMITH — auto-traces all LangChain calls when LANGCHAIN_API_KEY is set
# =============================================================================
try:
    from langsmith import traceable as _traceable
    if Config.LANGCHAIN_API_KEY:
        logger.info(f"LangSmith tracing enabled — project: {Config.LANGCHAIN_PROJECT}")
    else:
        def _traceable(**_):
            return lambda f: f
except ImportError:
    def _traceable(**_):
        return lambda f: f


# =============================================================================
# QUERY ROUTER
# =============================================================================
_QUERY_PROMPTS = {
    "FACTUAL": """Answer ONLY using the document context provided below.
STRICT RULES — violating any of these is an error:
  1. Do NOT use your general knowledge or training data.
  2. Do NOT fabricate connections between the question and unrelated content.
  3. Do NOT speculate or infer beyond what the context explicitly states.
  4. If the exact answer is not present in the context, respond with exactly:
     "This question is not covered in the loaded document. Please ask something about the document content."
  5. After each sentence drawn from the context, place a superscript number at the end of the sentence (e.g. ¹ for Source 1, ² for Source 2, ³ for Source 3). No brackets, no "Source" word — just the tiny number. Example: "Random forests outperform single decision trees on most benchmarks.¹"
{history_section}
Context:
{context}

Question: {question}
Answer:""",

    "CONCEPTUAL": """Explain using ONLY the document context provided below.
STRICT RULES — violating any of these is an error:
  1. Do NOT use your general knowledge or training data.
  2. Do NOT fabricate connections between the question and unrelated content.
  3. Do NOT speculate or infer beyond what the context explicitly states.
  4. If the concept is not explained in the context, respond with exactly:
     "This question is not covered in the loaded document. Please ask something about the document content."
  5. After each sentence drawn from the context, place a superscript number at the end of the sentence (e.g. ¹ for Source 1, ² for Source 2, ³ for Source 3). No brackets, no "Source" word — just the tiny number. Example: "Random forests outperform single decision trees on most benchmarks.¹"
{history_section}
Context:
{context}

Question: {question}
Explanation:""",

    "COMPARATIVE": """Compare using ONLY the document context provided below.
STRICT RULES — violating any of these is an error:
  1. Do NOT use your general knowledge or training data.
  2. Do NOT fabricate connections between the question and unrelated content.
  3. Do NOT speculate or infer beyond what the context explicitly states.
  4. If the comparison cannot be drawn from the context, respond with exactly:
     "This question is not covered in the loaded document. Please ask something about the document content."
  5. After each sentence drawn from the context, place a superscript number at the end of the sentence (e.g. ¹ for Source 1, ² for Source 2, ³ for Source 3). No brackets, no "Source" word — just the tiny number. Example: "Random forests outperform single decision trees on most benchmarks.¹"
{history_section}
Context:
{context}

Question: {question}
Comparison:""",
}


def classify_query(question: str) -> str:
    """Classify query type via keyword rules: FACTUAL | CONCEPTUAL | COMPARATIVE."""
    q = question.lower()
    if any(w in q for w in [
        "compare", "difference", "vs ", "versus", "contrast",
        "better than", "similar to",
    ]):
        return "COMPARATIVE"
    if any(w in q for w in [
        "explain", "how does", "how do", "how is", "why", "describe",
        "what is", "what are", "define", "concept", "tell me about",
        "elaborate", "summarize", "overview", "types of",
    ]):
        return "CONCEPTUAL"
    return "FACTUAL"


# =============================================================================
# QUERY DECOMPOSER
# =============================================================================
def _is_complex_query(question: str) -> bool:
    """True if the question likely has multiple distinct aspects."""
    if question.count("?") > 1:
        return True
    if len(question.split()) > 25:
        return True
    patterns = [
        r"\band explain\b", r"\band describe\b", r"\band also\b",
        r"\badditionally\b", r"\bfurthermore\b", r"\bas well as\b",
        r"compare .{3,} and", r"difference between .{3,} and",
    ]
    q = question.lower()
    return any(re.search(p, q) for p in patterns)


def decompose_query(question: str) -> list:
    """
    Break a complex question into 2-3 focused sub-questions.
    Returns original question unchanged if decomposition fails or is unnecessary.
    """
    template = """Break the following complex question into 2-3 simpler, self-contained sub-questions that together cover the original question.
If the question is already simple, return just: 1. {question}

Question: {question}

Sub-questions (numbered list):"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = prompt | get_llm() | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip()
        sub_queries = []
        for line in result.split("\n"):
            cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", line.strip()).strip()
            if len(cleaned) > 10:
                sub_queries.append(cleaned)
        valid = sub_queries[:3]
        if valid:
            logger.info(f"Decomposed into {len(valid)} sub-queries")
        return valid if valid else [question]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return [question]


# =============================================================================
# SEMANTIC CACHE  (Redis-backed with in-memory fallback)
# =============================================================================
class PersistentSemanticCache:
    """
    Semantic cache backed by Redis with in-memory OrderedDict fallback.

    Redis keys:  rag:cache:<doc_id>:<md5(question)>
    Value:       JSON { question, embedding (list), result (dict) }
    TTL:         REDIS_CACHE_TTL seconds (default 24 h)
    """

    def __init__(self, doc_id: str, threshold: float, max_size: int, ttl: int):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl = ttl
        self._ns = f"rag:cache:{doc_id}:"
        self._redis = None
        self._mem: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

        if Config.REDIS_URL:
            try:
                import redis as _redis
                client = _redis.from_url(Config.REDIS_URL, decode_responses=True)
                client.ping()
                self._redis = client
                logger.info(f"Redis cache ready  [{Config.REDIS_URL}]  ns={self._ns}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}) — using in-memory cache.")

    @staticmethod
    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (norm + 1e-10))

    def lookup(self, question: str):
        emb = np.array(get_embeddings().embed_query(question))

        if self._redis is not None:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(
                        cursor, match=f"{self._ns}*", count=200
                    )
                    for key in keys:
                        raw = self._redis.get(key)
                        if not raw:
                            continue
                        entry = json.loads(raw)
                        sim = self._sim(emb, np.array(entry["embedding"]))
                        if sim >= self.threshold:
                            logger.info(
                                f"Redis cache HIT (sim={sim:.3f}): "
                                f"'{entry['question'][:50]}'"
                            )
                            return entry["result"]
                    if cursor == 0:
                        break
                return None
            except Exception as e:
                logger.warning(f"Redis lookup error: {e}")

        with self._lock:
            for cached_q, (cached_emb, result) in self._mem.items():
                sim = self._sim(emb, cached_emb)
                if sim >= self.threshold:
                    logger.info(f"Memory cache HIT (sim={sim:.3f}): '{cached_q[:50]}'")
                    return result
        return None

    def store(self, question: str, result: dict) -> None:
        emb = np.array(get_embeddings().embed_query(question))

        if self._redis is not None:
            try:
                key = f"{self._ns}{hashlib.md5(question.encode()).hexdigest()}"
                payload = json.dumps({
                    "question": question,
                    "embedding": emb.tolist(),
                    "result": result,
                })
                self._redis.setex(key, self.ttl, payload)
                logger.info(
                    f"Redis cache STORE: '{question[:50]}' "
                    f"(TTL={self.ttl}s, ns={self._ns})"
                )
                return
            except Exception as e:
                logger.warning(f"Redis store error: {e}")

        with self._lock:
            if len(self._mem) >= self.max_size:
                self._mem.popitem(last=False)
            self._mem[question] = (emb, result)
        logger.info(f"Memory cache STORE: '{question[:50]}' (size={len(self._mem)})")

    def size(self) -> int:
        if self._redis is not None:
            try:
                return sum(1 for _ in self._redis.scan_iter(f"{self._ns}*"))
            except Exception:
                pass
        with self._lock:
            return len(self._mem)

    def clear(self) -> None:
        if self._redis is not None:
            try:
                keys = list(self._redis.scan_iter(f"{self._ns}*"))
                if keys:
                    self._redis.delete(*keys)
                logger.info(f"Redis cache cleared: {len(keys)} entries (ns={self._ns})")
                return
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        with self._lock:
            self._mem.clear()

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"


# =============================================================================
# CACHED MODEL LOADING
# =============================================================================
@lru_cache(maxsize=1)
def get_embeddings():
    """Return embeddings model (cached)."""
    try:
        if Config.use_openai():
            logger.info("Using OpenAI Embeddings")
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        else:
            logger.info(f"Using HuggingFace Embeddings: {Config.EMBEDDING_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise RuntimeError(f"Embeddings initialization failed: {e}")


@lru_cache(maxsize=1)
def get_llm():
    """Return LLM (cached). Priority: Groq > OpenAI > HuggingFace FLAN-T5."""
    try:
        if Config.use_groq():
            logger.info(f"Using Groq: {Config.GROQ_MODEL}")
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=Config.GROQ_MODEL,
                groq_api_key=Config.GROQ_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
            )
        elif Config.use_openai():
            logger.info(f"Using OpenAI: {Config.OPENAI_MODEL}")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=Config.OPENAI_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
            )
        else:
            logger.info(f"Using HuggingFace LLM: {Config.LLM_MODEL}")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline as hf_pipeline

            tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(Config.LLM_MODEL)
            pipe = hf_pipeline(
                task="text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                do_sample=False,
                temperature=0.7,
                device=-1,
            )
            return HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={
                    "max_new_tokens": 150,
                    "min_length": 20,
                    "early_stopping": True,
                },
            )
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise RuntimeError(f"LLM initialization failed: {e}")


@lru_cache(maxsize=1)
def get_reranker():
    """Return cross-encoder reranker (cached). Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {Config.RERANKER_MODEL}")
        return CrossEncoder(Config.RERANKER_MODEL)
    except Exception as e:
        logger.warning(f"Reranker unavailable: {e}. Falling back to raw retrieval order.")
        return None


# =============================================================================
# RERANKING + CRAG GRADING
# =============================================================================
@_traceable(name="rerank_and_grade", run_type="retriever")
def rerank_and_grade(query: str, docs: list, top_k: int) -> tuple:
    """
    Rerank docs with cross-encoder, then CRAG-grade retrieval quality.
    Returns (filtered_docs, grade): CORRECT | AMBIGUOUS | INCORRECT.
    """
    if not docs:
        return [], "INCORRECT"

    reranker = get_reranker()
    if reranker is None:
        return docs[:top_k], "CORRECT"

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    best_score = float(scored[0][0])

    if best_score >= Config.GRADE_CORRECT_THRESHOLD:
        grade = "CORRECT"
    elif best_score >= Config.GRADE_AMBIGUOUS_THRESHOLD:
        grade = "AMBIGUOUS"
    else:
        grade = "INCORRECT"

    filtered = []
    for score, doc in scored:
        if float(score) >= Config.GRADE_AMBIGUOUS_THRESHOLD:
            doc.metadata["rerank_score"] = round(float(score), 3)
            filtered.append(doc)

    logger.info(
        f"Reranker: best={best_score:.2f}, grade={grade}, "
        f"kept {len(filtered)}/{len(docs)} chunks"
    )
    return filtered[:top_k], grade


# =============================================================================
# QUESTION CONDENSATION
# =============================================================================
def condense_question(question: str, history: str) -> str:
    """Rewrite a follow-up question as a standalone question using chat history."""
    template = """Given the conversation history and a follow-up question, rewrite the follow-up as a complete standalone question. If already standalone, return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""
    prompt = PromptTemplate(template=template, input_variables=["history", "question"])
    chain = prompt | get_llm() | StrOutputParser()
    try:
        condensed = chain.invoke({"history": history, "question": question}).strip()
        if condensed and condensed != question:
            logger.info(f"Condensed: '{question[:50]}' -> '{condensed[:50]}'")
        return condensed or question
    except Exception as e:
        logger.warning(f"Question condensation failed: {e}")
        return question


# =============================================================================
# CHAT HISTORY PARSER  +  HyDE
# =============================================================================
def _parse_recent_history(history_text: str, n_turns: int = 2) -> str:
    """
    Extract the last n_turns Q&A exchanges from the conversation history string
    and format them for inclusion in the LLM answer prompt.

    History format written by app.py:
        [HH:MM:SS]  You: <question>\\n\\nAssistant: <answer>\\n\\n. . . (separator)

    Returns a formatted block, or empty string when history is absent.
    """
    if not history_text.strip():
        return ""

    separator = ". " * 30
    entries = [e.strip() for e in history_text.split(separator) if e.strip()]
    recent = entries[-n_turns:] if entries else []

    pairs = []
    for entry in recent:
        user_m = re.search(r"You:\s*(.+?)(?=\n\nAssistant:)", entry, re.DOTALL)
        asst_m = re.search(r"Assistant:\s*(.+?)$", entry, re.DOTALL)
        if user_m and asst_m:
            q = user_m.group(1).strip()
            a = asst_m.group(1).strip()[:250]
            pairs.append(f"User: {q}\nAssistant: {a}")

    if not pairs:
        return ""

    return (
        "Recent conversation (use for context, do not repeat these answers):\n"
        + "\n\n".join(pairs)
        + "\n\n"
    )


def _hyde_query(question: str) -> str:
    """
    HyDE — Hypothetical Document Embeddings.
    Generates a short passage that would answer the question, then embeds
    THAT for FAISS dense retrieval — bridges the query/document vocab gap.
    Returns hypothetical passage, or original question on failure.
    """
    template = """Write a concise passage (2-3 sentences) that would directly answer the following question if found in a document. Write only the passage, no preamble or labels.

Question: {question}

Passage:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = prompt | get_llm() | StrOutputParser()
    try:
        result = chain.invoke({"question": question}).strip()
        logger.info(f"HyDE generated hypothetical: '{result[:80]}...'")
        return result or question
    except Exception as e:
        logger.warning(f"HyDE failed ({e}), using original question")
        return question


# =============================================================================
# QA CHAIN
# =============================================================================
def make_qa_chain(vector_store, doc_id: str = "default", all_chunks: list = None):
    """
    Build the full production QA chain:
      1. Semantic cache lookup   (skip everything on hit)
      2. Question condensation   (rewrite follow-ups)
      3. Query decomposition     (split complex questions)
      4. Hybrid BM25+FAISS/Qdrant (RRF-fused retrieval)
      5. Cross-encoder reranking + CRAG grading
      6. Query routing           (FACTUAL / CONCEPTUAL / COMPARATIVE)
      7. Token streaming
      8. PII redaction           (applied at app layer via redact_pii)
      9. Semantic cache store    (save result for future hits)

    all_chunks — pass from build_vector_store() return value so BM25 and
                 retrieval eval work with any backend (FAISS or Qdrant).
                 If None, falls back to extracting from FAISS docstore.
    """
    try:
        retriever = vector_store.as_retriever(
            search_type=Config.SEARCH_TYPE,
            search_kwargs={"k": Config.RETRIEVAL_K_INITIAL},
        )

        # Resolve chunks for BM25 and retrieval evaluation
        if all_chunks is None:
            # FAISS-only fallback — Qdrant stores don't have this attribute
            try:
                _doc_ids = list(vector_store.index_to_docstore_id.values())
                all_chunks = [vector_store.docstore.search(did) for did in _doc_ids]
                all_chunks = [d for d in all_chunks if d is not None]
                logger.info(f"Extracted {len(all_chunks)} chunks from FAISS docstore")
            except AttributeError:
                logger.warning("Cannot extract chunks from this vector store — BM25 and retrieval eval disabled")
                all_chunks = []

        # BM25 sparse index
        bm25_index = None
        bm25_docs = []
        try:
            from rank_bm25 import BM25Okapi
            bm25_docs = all_chunks
            corpus = [doc.page_content.lower().split() for doc in bm25_docs]
            bm25_index = BM25Okapi(corpus)
            logger.info(f"BM25 index built: {len(bm25_docs)} docs")
        except ImportError:
            logger.warning("rank_bm25 not installed — dense-only. Run: pip install rank-bm25")
        except Exception as e:
            logger.warning(f"BM25 build failed: {e} — falling back to dense-only.")

        # One answer chain per query type
        answer_chains = {}
        for qtype, template in _QUERY_PROMPTS.items():
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question", "history_section"],
            )
            answer_chains[qtype] = prompt | get_llm() | StrOutputParser()

        # Per-document persistent cache (Redis when available, memory fallback)
        cache = PersistentSemanticCache(
            doc_id=doc_id,
            threshold=Config.SEMANTIC_CACHE_THRESHOLD,
            max_size=Config.SEMANTIC_CACHE_SIZE,
            ttl=Config.REDIS_CACHE_TTL,
        )

        class QAChainWrapper:
            def __init__(self, answer_chains, retriever, bm25_index, bm25_docs, cache, all_chunks):
                self._answer_chains = answer_chains
                self._retriever = retriever
                self._bm25 = bm25_index
                self._bm25_docs = bm25_docs
                self._cache = cache
                self._all_chunks = all_chunks

            # ----------------------------------------------------------------
            # Hybrid BM25 + FAISS retrieval with RRF  (+optional HyDE)
            # ----------------------------------------------------------------
            def _hybrid_retrieve(self, query: str) -> list:
                dense_query = _hyde_query(query) if Config.HYDE_ENABLED else query
                dense = self._retriever.invoke(dense_query)

                if self._bm25 is None or not self._bm25_docs:
                    return dense

                tokens = query.lower().split()
                scores = self._bm25.get_scores(tokens)
                top_idx = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:Config.RETRIEVAL_K_INITIAL]
                sparse = [self._bm25_docs[i] for i in top_idx]

                rrf_k = 60
                content_to_doc = {}
                rrf = {}
                for rank, doc in enumerate(dense):
                    k = doc.page_content
                    content_to_doc[k] = doc
                    rrf[k] = rrf.get(k, 0.0) + 1.0 / (rrf_k + rank)
                for rank, doc in enumerate(sparse):
                    k = doc.page_content
                    content_to_doc[k] = doc
                    rrf[k] = rrf.get(k, 0.0) + 1.0 / (rrf_k + rank)

                merged = sorted(rrf, key=lambda x: rrf[x], reverse=True)
                fused = [content_to_doc[k] for k in merged[:Config.RETRIEVAL_K_INITIAL]]
                logger.info(
                    f"Hybrid: {len(dense)} dense + {len(sparse)} BM25 "
                    f"-> {len(fused)} fused"
                )
                return fused

            # ----------------------------------------------------------------
            # Preprocessing: condense -> decompose -> retrieve -> rerank+grade
            # ----------------------------------------------------------------
            def _preprocess(self, inputs: dict) -> tuple:
                question = inputs.get("query") or inputs.get("question")
                history = inputs.get("history", "").strip()
                if not question:
                    raise ValueError("No question provided in inputs")

                standalone_q = question
                if history and Config.CONDENSE_QUESTIONS:
                    recent = history[-500:] if len(history) > 500 else history
                    standalone_q = condense_question(question, recent)

                if Config.DECOMPOSE_QUERIES and _is_complex_query(standalone_q):
                    sub_queries = decompose_query(standalone_q)
                else:
                    sub_queries = [standalone_q]

                seen: set = set()
                all_docs = []
                for sq in sub_queries:
                    for doc in self._hybrid_retrieve(sq):
                        key = doc.page_content[:80]
                        if key not in seen:
                            seen.add(key)
                            all_docs.append(doc)

                all_docs = all_docs[: Config.RETRIEVAL_K_INITIAL * 2]
                graded_docs, grade = rerank_and_grade(
                    standalone_q, all_docs, Config.RETRIEVAL_K
                )
                return standalone_q, graded_docs, grade

            # ----------------------------------------------------------------
            # Non-streaming invoke
            # ----------------------------------------------------------------
            def invoke(self, inputs: dict) -> dict:
                question = inputs.get("query") or inputs.get("question", "")

                cached = self._cache.lookup(question)
                if cached is not None:
                    return {**cached, "cache_hit": True}

                standalone_q, graded_docs, grade = self._preprocess(inputs)

                if grade in ("INCORRECT", "AMBIGUOUS") or not graded_docs:
                    return {
                        "result": (
                            "This question is not covered in the loaded document. "
                            "Please ask something about the document content."
                        ),
                        "sources": [],
                        "grade": grade if graded_docs else "INCORRECT",
                        "query_type": "N/A",
                        "cache_hit": False,
                    }

                query_type = classify_query(standalone_q)
                context = "\n\n".join(
                    f"[Source {i+1}]\n{doc.page_content}"
                    for i, doc in enumerate(graded_docs)
                )
                history_section = _parse_recent_history(
                    inputs.get("history", ""), Config.HISTORY_TURNS
                )
                result = self._answer_chains[query_type].invoke(
                    {"context": context, "question": standalone_q,
                     "history_section": history_section}
                )
                sources = [
                    {
                        "chunk": i + 1,
                        "content": doc.page_content[:300],
                        "page": doc.metadata.get("page", "N/A"),
                        "score": doc.metadata.get("rerank_score", "N/A"),
                        "method": doc.metadata.get("extract_method", ""),
                    }
                    for i, doc in enumerate(graded_docs)
                ]
                payload = {
                    "result": result,
                    "sources": sources,
                    "grade": grade,
                    "query_type": query_type,
                    "cache_hit": False,
                }
                self._cache.store(question, {k: v for k, v in payload.items() if k != "cache_hit"})
                return payload

            # ----------------------------------------------------------------
            # Streaming invoke
            # ----------------------------------------------------------------
            def stream(self, inputs: dict):
                """
                Generator yielding dicts:
                  {"chunk", "result", "sources", "grade", "query_type",
                   "cache_hit", "done"}
                Intermediate yields have done=False and empty sources.
                Final yield has done=True with full sources.
                """
                question = inputs.get("query") or inputs.get("question", "")

                cached = self._cache.lookup(question)
                if cached is not None:
                    yield {
                        "chunk": cached["result"],
                        "result": cached["result"],
                        "sources": cached.get("sources", []),
                        "grade": cached.get("grade", "CORRECT"),
                        "query_type": cached.get("query_type", ""),
                        "cache_hit": True,
                        "done": True,
                    }
                    return

                try:
                    standalone_q, graded_docs, grade = self._preprocess(inputs)
                except Exception as e:
                    yield {
                        "chunk": "", "result": str(e),
                        "sources": [], "grade": "ERROR",
                        "query_type": "N/A", "cache_hit": False, "done": True,
                    }
                    return

                if grade in ("INCORRECT", "AMBIGUOUS") or not graded_docs:
                    msg = (
                        "This question is not covered in the loaded document. "
                        "Please ask something about the document content."
                    )
                    yield {
                        "chunk": msg, "result": msg,
                        "sources": [], "grade": grade if graded_docs else "INCORRECT",
                        "query_type": "N/A", "cache_hit": False, "done": True,
                    }
                    return

                query_type = classify_query(standalone_q)
                context = "\n\n".join(
                    f"[Source {i+1}]\n{doc.page_content}"
                    for i, doc in enumerate(graded_docs)
                )
                history_section = _parse_recent_history(
                    inputs.get("history", ""), Config.HISTORY_TURNS
                )
                sources = [
                    {
                        "chunk": i + 1,
                        "content": doc.page_content[:300],
                        "page": doc.metadata.get("page", "N/A"),
                        "score": doc.metadata.get("rerank_score", "N/A"),
                        "method": doc.metadata.get("extract_method", ""),
                    }
                    for i, doc in enumerate(graded_docs)
                ]
                answer_chain = self._answer_chains[query_type]
                full_answer = ""
                chain_inputs = {
                    "context": context,
                    "question": standalone_q,
                    "history_section": history_section,
                }

                try:
                    for chunk in answer_chain.stream(chain_inputs):
                        full_answer += chunk
                        yield {
                            "chunk": chunk,
                            "result": full_answer,
                            "sources": [],
                            "grade": grade,
                            "query_type": query_type,
                            "cache_hit": False,
                            "done": False,
                        }
                except Exception:
                    # Fallback for models that don't support streaming
                    full_answer = answer_chain.invoke(chain_inputs)
                    yield {
                        "chunk": full_answer,
                        "result": full_answer,
                        "sources": [],
                        "grade": grade,
                        "query_type": query_type,
                        "cache_hit": False,
                        "done": False,
                    }

                self._cache.store(question, {
                    "result": full_answer,
                    "sources": sources,
                    "grade": grade,
                    "query_type": query_type,
                })

                yield {
                    "chunk": "",
                    "result": full_answer,
                    "sources": sources,
                    "grade": grade,
                    "query_type": query_type,
                    "cache_hit": False,
                    "done": True,
                }

            def run(self, question: str) -> str:
                return self.invoke({"query": question})["result"]

            def cache_size(self) -> int:
                return self._cache.size()

            def clear_cache(self) -> None:
                self._cache.clear()

            # ----------------------------------------------------------------
            # Phase 1 — Retrieval Evaluation
            # ----------------------------------------------------------------
            def run_retrieval_eval(self, n_questions: int = 8, k: int = 3) -> dict:
                """
                Generate a synthetic test set from document chunks, then compute
                Precision@K, Recall@K, MRR, and Coverage.
                ~10-20 seconds for n_questions=8 (one LLM call per question).
                """
                test_set = generate_retrieval_test_set(self._all_chunks, n_questions)
                if not test_set:
                    return {"error": "Could not generate test set from document chunks"}
                return evaluate_retrieval_metrics(self._hybrid_retrieve, test_set, k)

        wrapped = QAChainWrapper(answer_chains, retriever, bm25_index, bm25_docs, cache, all_chunks)
        logger.info(
            "QA chain ready: input guard | multi-format | hybrid BM25+FAISS | "
            "reranker | CRAG | query router | decomposer | condensation | "
            "semantic cache | output guard | streaming"
        )
        return wrapped

    except Exception as e:
        logger.error(f"Failed to create QA chain: {e}")
        raise RuntimeError(f"QA chain creation failed: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_vector_store_info(store_path: str, doc_id: str = "default") -> dict:
    """Get information about an existing vector store (FAISS or Qdrant)."""
    try:
        vs, chunks = load_vector_store(store_path, doc_id=doc_id)
        info = {"exists": True, "num_documents": len(chunks)}
        # FAISS has .index.ntotal; Qdrant does not
        if hasattr(vs, "index"):
            info["num_documents"] = vs.index.ntotal
        if store_path:
            info["path"] = store_path
        return info
    except FileNotFoundError:
        return {"exists": False}
    except Exception as e:
        return {"exists": True, "error": str(e)}
