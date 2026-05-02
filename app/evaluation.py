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

import logging
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


# =============================================================================
# PHASE 2 — RAGAS GENERATION EVALUATION
# =============================================================================
def evaluate_rag_response(question: str, answer: str, contexts: list) -> dict:
    """
    Score a single RAG response with RAGAS metrics (no ground truth needed).

    Metrics:
      Faithfulness               — every claim is supported by retrieved chunks
      Answer Relevancy           — answer addresses the question asked
      Context Precision          — retrieved chunks are relevant and well-ranked

    Scores: 0.0 (poor) → 1.0 (perfect).
    Uses the same Groq LLM + HuggingFace embeddings already loaded.
    Requires RAGAS_EVAL=true in .env.
    """
    if not Config.RAGAS_EVAL:
        return {"info": "Set RAGAS_EVAL=true in .env to enable RAGAS evaluation"}
    if not answer.strip() or not contexts:
        return {"error": "No answer or contexts available to evaluate"}

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        return {"error": "Run: pip install ragas datasets"}

    from app.rag_pipeline import get_llm, get_embeddings

    llm_w = LangchainLLMWrapper(get_llm())
    emb_w = LangchainEmbeddingsWrapper(get_embeddings())

    # RAGAS 0.2+ / 0.4.x API
    try:
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas import evaluate as _ragas_eval
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        result = _ragas_eval(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[
                Faithfulness(llm=llm_w),
                AnswerRelevancy(llm=llm_w, embeddings=emb_w),
                ContextPrecisionWithoutReference(llm=llm_w),
            ],
        )
        scores = {}
        for k, v in result.items():
            try:
                scores[k] = round(float(v), 3)
            except (TypeError, ValueError):
                pass
        return scores

    except (ImportError, AttributeError):
        pass

    # RAGAS 0.1.x fallback
    try:
        from ragas import evaluate as _ragas_eval
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

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
