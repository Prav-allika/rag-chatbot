"""
app/confidence.py — composite answer confidence score.

Combines three signals into one score:
  retrieval confidence  — how relevant were the top retrieved chunks?
  citation coverage     — what fraction of claims have a citation attached?
  answer completeness   — did the answer address the (possibly multi-part) question?

Deliberately cheap and always-on (no extra LLM calls) — unlike app/citations.py's
full per-sentence verification (which IS an LLM call per sentence and is kept
on-demand-only for exactly that reason), this is meant to run inline on every
answer. citation_coverage reuses app.citations.parse_cited_sentences, which is
regex-only. For a precise, LLM-judged version of these same ideas, use the
Phase 2 (RAGAS) / Phase 4 (Citation Verification) on-demand tabs instead.
"""

import math

from app.citations import parse_cited_sentences, _REFUSAL_MESSAGE
from app.config import Config


def retrieval_confidence(graded_docs: list) -> float:
    """Mean sigmoid(rerank_score) across the chunks used to answer.
    Sigmoid smoothly maps the CRAG thresholds into [0,1] without hard-clipping
    high scores (real CORRECT-grade scores commonly run well above +5)."""
    if not graded_docs:
        return 0.0
    scores = [d.metadata.get("rerank_score") for d in graded_docs]
    scores = [s for s in scores if s is not None]
    if not scores:
        return 0.0
    sigmoids = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return round(sum(sigmoids) / len(sigmoids), 3)


def citation_coverage(answer: str) -> float:
    """Fraction of citable sentences that carry a citation. 1.0 if there are
    no citable sentences at all (e.g. a refusal message) — don't penalize refusals."""
    cited, uncited = parse_cited_sentences(answer)
    total = len(cited) + len(uncited)
    if total == 0:
        return 1.0
    return round(len(cited) / total, 3)


def answer_completeness(answer: str, grade: str, sub_query_count: int = 1) -> float:
    """Cheap heuristic proxy (no LLM call) — CORRECT/AMBIGUOUS/INCORRECT grade as a
    base score, with a small penalty if the question was decomposed into multiple
    sub-questions but the answer looks too short to have addressed them all.

    Retrieval can grade CORRECT (good chunks found) while the LLM still declines
    to answer from them (the strict prompt allows this) — that's a non-answer
    regardless of grade, so it scores 0 rather than inheriting CORRECT's 1.0."""
    if _REFUSAL_MESSAGE in answer:
        return 0.0
    base = {"CORRECT": 1.0, "AMBIGUOUS": 0.4}.get(grade, 0.0)
    if base == 0.0 or sub_query_count <= 1:
        return round(base, 3)

    words_per_subquery = len(answer.split()) / sub_query_count
    if words_per_subquery < 8:   # suspiciously short for a multi-part question
        base *= 0.6
    return round(base, 3)


def compute_confidence(
    answer: str,
    graded_docs: list,
    grade: str,
    sub_query_count: int = 1,
    weight_retrieval: float = None,
    weight_coverage: float = None,
    weight_completeness: float = None,
) -> dict:
    """
    Returns {"retrieval", "citation_coverage", "completeness", "composite"}.
    Weights default to Config.CONFIDENCE_WEIGHT_* and are normalized to sum to 1.
    """
    w_r = Config.CONFIDENCE_WEIGHT_RETRIEVAL if weight_retrieval is None else weight_retrieval
    w_c = Config.CONFIDENCE_WEIGHT_COVERAGE if weight_coverage is None else weight_coverage
    w_a = Config.CONFIDENCE_WEIGHT_COMPLETENESS if weight_completeness is None else weight_completeness
    total_w = w_r + w_c + w_a
    if total_w <= 0:
        w_r, w_c, w_a, total_w = 1.0, 0.0, 0.0, 1.0

    r = retrieval_confidence(graded_docs)
    c = citation_coverage(answer)
    a = answer_completeness(answer, grade, sub_query_count)

    composite = (w_r * r + w_c * c + w_a * a) / total_w
    return {
        "retrieval": r,
        "citation_coverage": c,
        "completeness": a,
        "composite": round(composite, 3),
    }
