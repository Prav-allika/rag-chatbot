"""
app/evaluation.py — offline evaluation utilities.

Phase 1 — Retrieval Evaluation:
  Generates a synthetic test set from document chunks (LLM writes one
  question per chunk), then measures Precision@K, Recall@K, MRR, Coverage.

Phase 2 — Generation Evaluation (RAGAS):
  Scores a single RAG response on Faithfulness, Answer Relevancy,
  and Context Precision using your configured LLM as the judge.

Neither function is called in the production answer path.
Enable RAGAS with RAGAS_EVAL=true in .env.
"""

import json
import logging
import os
import random

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import Config

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 1 — RETRIEVAL EVALUATION
# =============================================================================
def _chunk_relevance(source: str, retrieved: str, threshold: float = 0.45) -> bool:
    """True if retrieved chunk shares enough words with the ground-truth chunk."""
    src = set(source.lower().split())
    ret = set(retrieved.lower().split())
    if not src:
        return False
    return len(src & ret) / len(src) >= threshold


def generate_retrieval_test_set(chunks: list, n_questions: int = 10) -> list:
    """
    Use the LLM to generate one focused question per sampled chunk.
    Each question's answer is definitively in its source chunk,
    giving us ground-truth relevance labels without manual annotation.
    Returns list of { question, source_content }.
    """
    from app.rag_pipeline import get_llm   # imported here to avoid circular imports

    if not chunks:
        return []

    sample = random.sample(chunks, min(n_questions, len(chunks)))
    template = (
        "Read the following document excerpt and write ONE specific factual question "
        "whose answer is clearly contained in the excerpt. "
        "Return only the question — no preamble, no numbering.\n\n"
        "Excerpt:\n{chunk}\n\nQuestion:"
    )
    prompt = PromptTemplate(template=template, input_variables=["chunk"])
    chain = prompt | get_llm() | StrOutputParser()

    test_set = []
    for chunk in sample:
        try:
            q = chain.invoke({"chunk": chunk.page_content[:600]}).strip()
            if q and len(q) > 10:
                test_set.append({"question": q, "source_content": chunk.page_content})
        except Exception as e:
            logger.warning(f"Test set generation failed for a chunk: {e}")

    logger.info(f"Retrieval test set: {len(test_set)} questions generated")
    return test_set


def save_test_set(test_set: list, path: str) -> None:
    """
    Persist a generated test set to JSON so it can be reused across configs.

    generate_retrieval_test_set() randomly samples chunks and has the LLM
    write fresh questions every call — comparing two configs against two
    independently-generated test sets confounds "config changed" with
    "questions changed". Generate once, save, then reuse for every config
    you want to compare.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(test_set, f, indent=2)
    logger.info(f"Test set saved: {len(test_set)} question(s) -> {path}")


def load_test_set(path: str) -> list:
    """Load a test set previously written by save_test_set()."""
    with open(path) as f:
        return json.load(f)


def evaluate_retrieval_metrics(retrieve_fn, test_set: list, k: int = 3) -> dict:
    """
    Compute retrieval quality metrics against a synthetic test set.

    Precision@K  — of K retrieved chunks, how many are relevant?
    Recall@K     — was the source chunk found in the top K?
    MRR          — Mean Reciprocal Rank of first relevant result
    Coverage     — % of questions with at least 1 relevant chunk retrieved

    Relevance is determined by word-overlap >= 45% with the source chunk.
    """
    if not test_set:
        return {"error": "Empty test set — no questions to evaluate"}

    precisions, recalls, reciprocal_ranks, covered = [], [], [], []

    for item in test_set:
        question = item["question"]
        source   = item["source_content"]

        try:
            retrieved = retrieve_fn(question)[:k]
        except Exception as e:
            logger.warning(f"Retrieval failed during eval: {e}")
            continue

        flags = [_chunk_relevance(source, doc.page_content) for doc in retrieved]

        precisions.append(sum(flags) / k)
        recall = 1.0 if any(flags) else 0.0
        recalls.append(recall)
        covered.append(1 if any(flags) else 0)

        rr = 0.0
        for rank, flag in enumerate(flags, start=1):
            if flag:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(precisions)
    if n == 0:
        return {"error": "All retrievals failed — no metrics computed"}

    return {
        "precision_at_k": round(sum(precisions) / n, 3),
        "recall_at_k":    round(sum(recalls) / n, 3),
        "mrr":            round(sum(reciprocal_ranks) / n, 3),
        "coverage":       round(sum(covered) / n, 3),
        "n_questions":    n,
        "k":              k,
    }


def _ragas_llm():
    """
    Native ragas LLM (InstructorBaseRagasLLM) for the app's configured provider.
    ragas 0.4.x's `metrics.collections` API requires this — it rejects the
    LangChain wrapper (LangchainLLMWrapper) used by older ragas versions.

    OpenAI is checked first, matching get_llm()'s own priority (OpenAI >
    Groq) — this used to check Groq first regardless of whether OpenAI was
    also configured, which silently diverged from the rest of the app
    whenever both keys were set. That divergence surfaced as a genuine
    crash, not just an inconsistency: ragas's Groq path patches AsyncGroq
    via `instructor`, and that patching failed against the installed groq
    SDK ("'AsyncGroq' object has no attribute 'messages'") -- a real
    incompatibility between ragas and groq's client internals, sidestepped
    entirely by preferring the OpenAI path whenever it's available.

    gpt-5.x models need the same reasoning_effort/max_tokens handling as
    get_llm() (rag_pipeline.py) -- they spend max_tokens on hidden
    reasoning before any visible output, and ragas's `instructor`-based
    structured-output parsing has no budget left over: it raised
    instructor.v2.core.errors.IncompleteOutputException ("output is
    incomplete due to a max_tokens length limit") on every RAGAS call
    once the Groq/instructor crash above was fixed and this path actually
    started being exercised. reasoning_effort="minimal" avoids burning the
    budget on reasoning, and a higher max_tokens leaves room for RAGAS's
    multi-claim JSON output (longer than a typical chat answer).
    """
    from ragas.llms import llm_factory

    if Config.use_openai():
        from openai import AsyncOpenAI
        kwargs = {"max_tokens": max(Config.LLM_MAX_OUTPUT_TOKENS, 2000)}
        if Config.OPENAI_MODEL.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "minimal"
        return llm_factory(
            Config.OPENAI_MODEL, provider="openai",
            client=AsyncOpenAI(api_key=Config.OPENAI_API_KEY), **kwargs,
        )
    from groq import AsyncGroq
    return llm_factory(Config.GROQ_MODEL, provider="groq", client=AsyncGroq(api_key=Config.GROQ_API_KEY))


def _ragas_embeddings():
    """
    Native ragas embeddings for AnswerRelevancy. Only available when OpenAI is
    configured — our app has no other embeddings provider wired up.
    """
    if not Config.use_openai():
        return None
    from ragas.embeddings.base import embedding_factory
    from openai import AsyncOpenAI
    return embedding_factory(
        provider="openai",
        model=Config.OPENAI_EMBEDDING_MODEL,
        client=AsyncOpenAI(api_key=Config.OPENAI_API_KEY),
    )


# =============================================================================
# PHASE 2 — RAGAS GENERATION EVALUATION
# =============================================================================
def evaluate_rag_response(question: str, answer: str, contexts: list) -> dict:
    """
    Score a single RAG response with RAGAS metrics (no ground truth needed).

    Metrics:
      Faithfulness               — every claim is supported by retrieved chunks
      Answer Relevancy           — answer addresses the question asked (skipped
                                    if OpenAI isn't configured — no embeddings provider)
      Context Precision          — retrieved chunks are relevant and well-ranked

    Scores: 0.0 (poor) → 1.0 (perfect).
    Uses the same LLM configured for the app (Groq or OpenAI).
    Requires RAGAS_EVAL=true in .env.
    """
    if not Config.RAGAS_EVAL:
        return {"info": "Set RAGAS_EVAL=true in .env to enable RAGAS evaluation"}
    if not answer.strip() or not contexts:
        return {"error": "No answer or contexts available to evaluate"}

    import asyncio

    # RAGAS 0.4.x "collections" API — native provider clients, async ascore()
    try:
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        llm = _ragas_llm()
        embeddings = _ragas_embeddings()

        async def _run():
            scores = {}
            f = await Faithfulness(llm=llm).ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            scores["faithfulness"] = f.value

            p = await ContextPrecisionWithoutReference(llm=llm).ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            scores["context_precision"] = p.value

            if embeddings is not None:
                r = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
                    user_input=question, response=answer
                )
                scores["answer_relevancy"] = r.value
            return scores

        scores = asyncio.run(_run())
        return {k: round(float(v), 3) for k, v in scores.items()}

    except ImportError as e:
        # Catches "ragas isn't installed at all" AND "ragas is installed but
        # some submodule/transitive import inside it failed" -- those are very
        # different problems, and reporting both as "pip install ragas
        # datasets" hides a real version/environment issue behind a misleading
        # fix. Log + surface the actual exception instead of guessing.
        logger.warning(f"RAGAS collections-API import failed: {e!r}")
        return {"error": f"RAGAS import failed ({e}). Run: pip install -U ragas datasets"}
    except AttributeError as e:
        logger.info(f"RAGAS collections API unavailable on this version ({e!r}) — falling back to legacy API")

    # RAGAS 0.1.x fallback (LangChain-wrapper API, older ragas versions)
    try:
        from app.rag_pipeline import get_llm, get_embeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import evaluate as _ragas_eval
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        llm_w = LangchainLLMWrapper(get_llm())
        emb_w = LangchainEmbeddingsWrapper(get_embeddings())
        faithfulness.llm = llm_w
        answer_relevancy.llm = llm_w
        answer_relevancy.embeddings = emb_w
        context_precision.llm = llm_w

        result = _ragas_eval(
            Dataset.from_dict({
                "question": [question],
                "answer":   [answer],
                "contexts": [contexts],
            }),
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        return {k: round(float(v), 3) for k, v in result.items() if isinstance(v, float)}

    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}
