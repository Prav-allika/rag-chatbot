"""
app/citations.py — post-generation citation verification.

The QA prompts (app/rag_pipeline.py:_QUERY_PROMPTS) instruct the LLM to place
a superscript number (¹, ², ³ ...) after every sentence drawn from context,
matching the "Source N" blocks built into the same prompt. Nothing checks,
after generation, whether the sentence attached to ¹ is actually supported
by Source 1's text — that's what this module does.

On-demand only (see app.py's "Phase 3 — Citation Verification" tab, mirroring
the existing RAGAS button) — not wired into the live answer path, since it
costs one LLM call per cited sentence.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_TO_DIGIT = {c: str(i) for i, c in enumerate(_SUPERSCRIPT_DIGITS)}

_SENTENCE_RE = re.compile(r"([^.!?\n]*[.!?])([⁰¹²³⁴⁵⁶⁷⁸⁹]*)")

_REFUSAL_MESSAGE = "This question is not covered in the loaded document."


def _superscripts_to_source_numbers(run: str) -> list:
    """'¹²' -> [1, 2] — each superscript character is read as its own citation,
    not concatenated into one multi-digit number (RETRIEVAL_K is small)."""
    return [int(_SUPERSCRIPT_TO_DIGIT[c]) for c in run if c in _SUPERSCRIPT_TO_DIGIT]


def parse_cited_sentences(answer: str) -> tuple:
    """
    Splits an answer into citable sentences and classifies each as cited or
    uncited. Skips the standard refusal message and any "Analogy:" line —
    the prompt explicitly allows the analogy to be original, uncited content.

    Returns (cited, uncited):
      cited   = [{"sentence": str, "cited_sources": [int, ...]}, ...]
      uncited = [str, ...]
    """
    cited, uncited = [], []

    for line in answer.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("Analogy:") or _REFUSAL_MESSAGE in stripped:
            continue

        for match in _SENTENCE_RE.finditer(line):
            sentence, superscripts = match.group(1).strip(), match.group(2)
            if not sentence:
                continue
            sources = _superscripts_to_source_numbers(superscripts)
            if sources:
                cited.append({"sentence": sentence, "cited_sources": sources})
            else:
                uncited.append(sentence)

    return cited, uncited


_VERIFY_PROMPT = """You are auditing a RAG system's citations. Decide whether the SOURCE TEXT actually supports the CLAIM.

CLAIM: {sentence}

SOURCE TEXT:
{source_text}

Reply with exactly one line, either:
SUPPORTED: <one short sentence why>
NOT SUPPORTED: <one short sentence why>"""


def _verify_one(sentence: str, cited_sources: list, content_by_chunk: dict) -> dict:
    from app.rag_pipeline import get_llm
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    source_text = "\n\n".join(
        content_by_chunk[n] for n in cited_sources if n in content_by_chunk
    )
    if not source_text.strip():
        return {
            "sentence": sentence, "cited_sources": cited_sources,
            "supported": False, "reason": "Cited source number not found among retrieved chunks.",
        }

    prompt = PromptTemplate(template=_VERIFY_PROMPT, input_variables=["sentence", "source_text"])
    chain = prompt | get_llm() | StrOutputParser()

    try:
        raw = chain.invoke({"sentence": sentence, "source_text": source_text}).strip()
        supported = raw.upper().startswith("SUPPORTED")
        reason = raw.split(":", 1)[1].strip() if ":" in raw else raw
        return {
            "sentence": sentence, "cited_sources": cited_sources,
            "supported": supported, "reason": reason,
        }
    except Exception as e:
        logger.warning(f"Citation verification failed for a sentence: {e}")
        return {
            "sentence": sentence, "cited_sources": cited_sources,
            "supported": False, "reason": f"Verification call failed: {e}",
        }


def verify_citations(answer: str, sources: list) -> dict:
    """
    Parses an answer's superscript citations and checks each one against the
    full (untruncated) text of its cited source chunk(s), via one LLM-as-judge
    call per cited sentence (parallelized).

    sources: the list returned by rag_pipeline's invoke()/stream(), each with
             "chunk" (1-indexed, matching the superscript) and "full_content".

    Returns:
      {
        "citations": [{"sentence", "cited_sources", "supported", "reason"}, ...],
        "uncited_sentences": [str, ...],
        "coverage": float,   # cited sentences / all citable sentences
        "accuracy": float,   # supported citations / total citations checked
      }
    """
    content_by_chunk = {s["chunk"]: s.get("full_content", s.get("content", "")) for s in sources}
    cited, uncited = parse_cited_sentences(answer)

    results = []
    if cited:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(_verify_one, c["sentence"], c["cited_sources"], content_by_chunk)
                for c in cited
            ]
            for f in as_completed(futures):
                results.append(f.result())

    total_citable = len(cited) + len(uncited)
    coverage = round(len(cited) / total_citable, 3) if total_citable else 0.0
    accuracy = round(sum(r["supported"] for r in results) / len(results), 3) if results else 0.0

    return {
        "citations": results,
        "uncited_sentences": uncited,
        "coverage": coverage,
        "accuracy": accuracy,
    }
